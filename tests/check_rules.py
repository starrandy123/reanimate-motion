"""rmo check <work dir> [--character <glb>] [--baked <glb>]: every rule the kit holds, each proven red by a planted defect.
<work dir> holds track.json, clean.json and solve/motion.json (rmo all). A rule HOLDS only when the real reading is green AND its plant
turns it red: a check that cannot fail is no check.
  R1 TRACKED  both legs and feet held (score >= 0.3) in >= 90 % of frames             plant: a quarter of the frames blanked
  R2 RIGID    every limb bone keeps its length within 5 % (std / median)              plant: the depth stretched x3
  R3 ANATOMY  facing the camera, a thigh far out of the picture has its knee IN FRONT; no knee bends backward past 15° (98 % of frames)
                                                                                      plant: the knees' depth mirrored about the hips
  R4 CLOCK    a body falling at 9.81 m/s², filmed 10x slow, reads back 10x (3 %)    plant: it falls at half gravity
  R5 SOLVED   the written skeleton puts the knees and ankles back within 3 cm        plant: one thigh turned 25°
  R6 SMOOTH   no bone turns faster than 1,500°/s of real time (99th percentile)     plant: the thighs swapped for one frame
  R7 HOSTS    Blender's keyed character and three.js's retarget agree within 2 cm    plant: Blender's keys one frame late
  R8 SCALE    the rider's leg is a person's (0.60-1.15 m), and a shoe the detector calls a board in 31 frames does not set the scale
  R9 HANDS    no wrist inside the trunk in more than 2 % of frames (7 Nov: "the hands are going in body")
                                                                                      plant: the wrists put in the trunk's middle for a tenth of the clip
  R10 TWIST   the chest turns no more than 45° against the pelvis in 98 % of frames (the spine: White & Panjabi)
                                                                                      plant: the shoulder line turned 90° for a third of the clip
  R11 PLACED  the skeleton's shoulders within 3 cm and its wrists within 6 cm of the tracked ones (medians) plant: the spine leaning 20° off
  R12 STRAIGHT where the picture shows an arm straight, its 3D elbow bends 20° or less (median)   plant: each forearm's depth mirrored
                                                                                      plant: the board taken whatever leg it implies (before 7 Nov)"""
import sys, os, json, copy, subprocess
import numpy as np
from scipy.spatial.transform import Rotation as Rot
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, '..', 'mocap')); from clock import fit_slowmo; from scale import pick as pick_scale, LEG_RANGE
W = sys.argv[1]; opt = lambda k, d=None: sys.argv[sys.argv.index(k) + 1] if k in sys.argv else d
T = json.load(open(os.path.join(W, 'track.json'))); C = json.load(open(os.path.join(W, 'clean.json'))); M = json.load(open(os.path.join(W, 'solve', 'motion.json')))
X = np.array(C['X'], float)

def r1(T):
    held = [('k' in r) and np.mean(np.array(r['s'])[11:23] > 0.3) > 0.9 for r in T['frames']]; v = float(np.mean(held))
    return v >= 0.9, f'{100 * v:.0f}% of frames hold both legs and feet'
def p1(T):
    T = copy.deepcopy(T); n = len(T['frames'])
    for r in T['frames'][n // 3: n // 3 + n // 4]: r['s'] = [0.0] * 133
    return T
LIMBS = [(11, 13), (13, 15), (12, 14), (14, 16), (5, 7), (7, 9), (6, 8), (8, 10)]
def r2(X):
    sp = []
    for a, b in LIMBS:
        d = np.linalg.norm(X[:, a] - X[:, b], axis=-1); d = d[np.isfinite(d)]
        if len(d) > 10: sp.append(float(np.std(d) / np.median(d)))
    return max(sp) <= 0.05, f'the most a limb bone stretches: {100 * max(sp):.1f}%'
def p2(X): X = X.copy(); X[..., 2] *= 3; return X
def r3(X):
    lat = X[:, 11] - X[:, 12]; facing = np.sign(lat[:, 0]); bad = tot = 0; hyper = hyp_tot = 0
    for hp, kn, an in ((11, 13, 15), (12, 14, 16)):
        u = X[:, kn] - X[:, hp]; w = X[:, an] - X[:, kn]; L = np.nanmedian(np.linalg.norm(u, axis=-1))
        far = np.linalg.norm(u[:, :2], axis=-1) < 0.943 * L                       # its depth over a third of its length
        ok = far & np.isfinite(u[:, 2]); tot += ok.sum(); bad += int((np.sign(u[ok, 2]) != facing[ok]).sum())
        fl = np.einsum('ij,ij->i', np.cross(u, w), lat) / (np.linalg.norm(u, axis=-1) * np.linalg.norm(w, axis=-1) * np.linalg.norm(lat, axis=-1) + 1e-9)
        f2 = np.isfinite(fl); hyp_tot += f2.sum(); hyper += int((fl[f2] < -np.sin(np.radians(15))).sum())
    a, b = bad / max(1, tot), hyper / max(1, hyp_tot)
    return a <= 0.02 and b <= 0.02, f'knees behind the hip where the thigh points out: {100 * a:.1f}% of {tot}; knees bent backward: {100 * b:.1f}%'
def p3(X):
    X = X.copy()
    for hp, kn in ((11, 13), (12, 14)): X[:, kn, 2] = 2 * X[:, hp, 2] - X[:, kn, 2]
    return X
def r4(g):
    t = np.arange(240); tau = t / (59.94 * 10); y = 0.3 + 1.6 * tau - 0.5 * g * tau ** 2
    got = fit_slowmo(t, y, 59.94); ok = bool(got) and abs(got[0] / 10 - 1) < 0.03
    return ok, f'a fall filmed 10x slow reads {got[0]:.2f}x' if got else 'no fit'
def fk(M, W_):
    names, par, off = M['bones'], dict(zip(M['bones'], M['parent'])), dict(zip(M['bones'], M['rest_offset']))
    R = {b: Rot.from_quat(np.array(W_[b])).as_matrix() for b in names}; H = np.array(M['hips']); pos = {}
    for b in names: pos[b] = H if par[b] is None else pos[par[b]] + np.einsum('nij,j->ni', R[par[b]], np.array(off[b]))
    return pos
def tracked(M):
    X0 = X[M['clip_from']:M['clip_to']]; t0 = np.arange(len(X0)) / M['video_per_real_second']; t = np.arange(M['frames']) / M['fps']
    return {j: np.stack([np.interp(t, t0, X0[:, j, c]) for c in range(3)], -1) for j in (13, 14, 15, 16)}
def r5(Wd):
    pos = fk(M, Wd); J = tracked(M); e = [float(np.nanmean(np.linalg.norm(pos[b] - J[j], axis=-1))) for b, j in (('LeftLeg', 13), ('RightLeg', 14), ('LeftFoot', 15), ('RightFoot', 16))]
    return max(e) <= 0.03, f'knees and ankles back within {100 * max(e):.1f} cm (the worst)'
def p5(Wd):
    Wd = copy.deepcopy(Wd); q = Rot.from_quat(np.array(Wd['LeftUpLeg'])); Wd['LeftUpLeg'] = (Rot.from_euler('x', 25, degrees=True) * q).as_quat().tolist(); return Wd
def r6(Wd):
    names, par = M['bones'], dict(zip(M['bones'], M['parent'])); worst, at = 0, None
    for b in names:
        if par[b] is None or b.startswith(('LeftHand', 'RightHand')) and len(b) > 9: continue
        loc = Rot.from_quat(np.array(Wd[par[b]])).inv() * Rot.from_quat(np.array(Wd[b])); ang = (loc[1:] * loc[:-1].inv()).magnitude() * 180 / np.pi * M['fps']
        v = float(np.percentile(ang, 99))
        if v > worst: worst, at = v, b
    return worst <= 1500, f'the fastest bone ({at}) turns {worst:.0f}°/s at its 99th percentile'
def p6(Wd):
    Wd = copy.deepcopy(Wd); f = M['frames'] // 2
    for k in range(f, f + max(2, M['frames'] // 50)): Wd['LeftUpLeg'][k], Wd['RightUpLeg'][k] = Wd['RightUpLeg'][k], Wd['LeftUpLeg'][k]   # 2 % of the clip: the 99th percentile must see it
    return Wd
def r7(plant):
    baked = opt('--baked'); char = opt('--character', os.path.join(HERE, '..', 'examples', 'models', 'street-skater.glb'))
    if not baked or not os.path.exists(baked): return None, 'no Blender-keyed character given (--baked): not read'
    cmd = ['node', os.path.join(HERE, 'hosts.mjs'), os.path.join(W, 'solve', 'motion.json'), char, baked] + (['--plant-late'] if plant else [])
    out = subprocess.run(cmd, capture_output=True, text=True, timeout=300); d = json.loads(out.stdout.strip().splitlines()[-1] or '{}')
    return out.returncode == 0, f"the hosts agree within {d.get('worst_cm', float('nan')):.2f} cm over {d.get('bones')} bones"

XW = X[M['clip_from']:M['clip_to']]   # the frames that become the animation: what the kit ships is what the anatomy rule reads
def legs_px(T):
    """the pictured legs, hip to knee to ankle, in pixels: their median and 95th percentile (what clean reads)"""
    out = []
    for r in T['frames']:
        if 'k' not in r: continue
        k = np.array(r['k'], float); sc = np.array(r['s'])
        for h, kn, a in ((11, 13, 15), (12, 14, 16)):
            if min(sc[h], sc[kn], sc[a]) >= 0.3: out.append(np.linalg.norm(k[h] - k[kn]) + np.linalg.norm(k[kn] - k[a]))
    return float(np.median(out)), float(np.percentile(out, 95))
def r8(open_range=False):
    lens = C['bones_m']; leg = (lens['11-13'] + lens['13-15'] + lens['12-14'] + lens['14-16']) / 2
    med, p95 = legs_px(T); n = len(T['frames'])
    bw = np.full(n, np.nan); bw[np.linspace(0, n - 1, 31).astype(int)] = p95 / 3          # a dancer's shoes, called a board in 31 frames
    ppm, src, _ = pick_scale(bw, p95, leg_range=(0.0, np.inf) if open_range else LEG_RANGE)
    shoe = p95 / ppm; lo, hi = LEG_RANGE
    return lo <= leg <= hi and lo <= shoe <= hi, f"the rider's leg {leg:.2f} m; a shoe called a board in 31 frames: the scale from {src}, the leg {shoe:.2f} m"
def in_trunk_pct(X):
    sh = (X[:, 5] + X[:, 6]) / 2; hp = (X[:, 11] + X[:, 12]) / 2; up = sh - hp; L = np.linalg.norm(up, axis=-1)
    lat = X[:, 5] - X[:, 6]; hw = np.linalg.norm(lat, axis=-1) / 2; latn = lat / (2 * hw[:, None]); fw = np.cross(latn, up / L[:, None]); out = []
    for w in (9, 10):
        v = X[:, w] - hp; a = np.einsum('ij,ij->i', v, up) / L ** 2; r = v - a[:, None] * up
        out.append((a > 0.05) & (a < 1.0) & (np.abs(np.einsum('ij,ij->i', r, latn)) < 0.85 * hw) & (np.abs(np.einsum('ij,ij->i', r, fw)) < 0.6 * hw))
    return 100 * float(np.nanmean(np.any(np.stack(out), 0)))
def r9(X):
    v = in_trunk_pct(X); return v <= 2.0, f'a wrist inside the trunk in {v:.1f}% of frames'
def p9(X):
    X = X.copy(); n = len(X); k = np.arange(n // 3, n // 3 + n // 10); mid = (X[k, 5] + X[k, 6] + X[k, 11] + X[k, 12]) / 4
    X[k, 9] = mid; X[k, 10] = mid; return X
def twist_pct(X):
    up = (X[:, 5] + X[:, 6]) / 2 - (X[:, 11] + X[:, 12]) / 2; upn = up / np.linalg.norm(up, axis=-1, keepdims=True)
    def flat(v): v = v - np.einsum('ij,ij->i', v, upn)[:, None] * upn; return v / np.linalg.norm(v, axis=-1, keepdims=True)
    a, b = flat(X[:, 11] - X[:, 12]), flat(X[:, 5] - X[:, 6])
    ang = np.degrees(np.abs(np.arctan2(np.einsum('ij,ij->i', np.cross(a, b), upn), np.einsum('ij,ij->i', a, b))))
    return 100 * float(np.nanmean(ang > 45))
def r10(X):
    v = twist_pct(X); return v <= 2.0, f'the chest turned past 45° against the pelvis in {v:.1f}% of frames'
def p10(X):
    X = X.copy(); n = len(X); k = np.arange(n // 3, 2 * n // 3)          # the shoulder line turned 90° about the trunk's middle (a line flat in the
    m = (X[k, 5] + X[k, 6]) / 2; h = (X[k, 5] - X[k, 6]) / 2             # picture: swapping its two depths turned nothing)
    r = np.stack([-h[:, 2], h[:, 1], h[:, 0]], -1); X[k, 5] = m + r; X[k, 6] = m - r; return X
def r11(Wd):
    """THE ARMS START WHERE THE RIDER'S DO: the written skeleton's shoulder joints within 3 cm of the tracked shoulders and its wrists within
    6 cm of the tracked wrists (medians). The spine had aimed off the trunk and put them 6-16 cm away (7 Nov, cut 2: "the hands are in the
    wrong position")"""
    pos = fk(M, Wd); X0 = X[M['clip_from']:M['clip_to']]; t0 = np.arange(len(X0)) / M['video_per_real_second']; t = np.arange(M['frames']) / M['fps']
    J = lambda j: np.stack([np.interp(t, t0, X0[:, j, c]) for c in range(3)], -1)
    e = lambda b, j: 100 * float(np.nanmedian(np.linalg.norm(pos[b] - J(j), axis=-1)))
    sh, wr = max(e('LeftArm', 5), e('RightArm', 6)), max(e('LeftHand', 9), e('RightHand', 10))
    return sh <= 3 and wr <= 6, f'the shoulders {sh:.1f} cm and the wrists {wr:.1f} cm from the tracked ones (medians, the worse side)'
def p11(Wd):
    Wd = copy.deepcopy(Wd); tilt = Rot.from_euler('x', 20, degrees=True)                       # the whole spine leaning 20° off the trunk's line
    for b in ('Spine', 'Spine1', 'Spine2'): Wd[b] = (tilt * Rot.from_quat(np.array(Wd[b]))).as_quat().tolist()
    return Wd
def r12(X):
    """A STRAIGHT ARM STAYS STRAIGHT: where the picture shows an arm straight (its elbow bent under 15° in the picture), the 3D elbow's median
    bend there is 20° or less (each arm with 20 such frames or more). Each bone had chosen its depth's side alone, and Mixkit 34382's left
    arm bent 51° (median) where the video shows it straight"""
    worst, say = 0.0, []
    for a, b, c in ((5, 7, 9), (6, 8, 10)):
        u2, f2 = Kp[:, b] - Kp[:, a], Kp[:, c] - Kp[:, b]
        b2 = np.degrees(np.arccos(np.clip((u2 * f2).sum(-1) / (np.linalg.norm(u2, axis=-1) * np.linalg.norm(f2, axis=-1) + 1e-9), -1, 1)))
        up, fo = X[:, b] - X[:, a], X[:, c] - X[:, b]
        b3 = np.degrees(np.arccos(np.clip((up * fo).sum(-1) / (np.linalg.norm(up, axis=-1) * np.linalg.norm(fo, axis=-1) + 1e-9), -1, 1)))
        st = (b2 < 15) & np.isfinite(b3)
        if st.sum() >= 20: m = float(np.median(b3[st])); worst = max(worst, m); say.append(f"{'left' if a == 5 else 'right'} {m:.0f}° over {int(st.sum())}")
    if not say: return None, 'no arm straight in the picture in 20 frames: not read'
    return worst <= 20, 'the 3D elbow where the picture shows the arm straight: ' + ', '.join(say)
def p12(X):
    X = X.copy()
    for e, w in ((7, 9), (8, 10)): X[:, w, 2] = 2 * X[:, e, 2] - X[:, w, 2]                    # each forearm's depth on the other side of its elbow
    return X
Kp = np.array([r['k'] if 'k' in r else [[np.nan, np.nan]] * 133 for r in T['frames']], float)   # the picture, as tracked
CHAR = opt('--character', os.path.join(HERE, '..', 'examples', 'models', 'street-skater.glb')); HIDE = [opt('--hide')] if opt('--hide') else []
FITTED = os.path.join(W, 'solve', 'motion_' + os.path.splitext(os.path.basename(CHAR))[0] + '.json')
def r13(plant=False):
    """THE CHARACTER'S HANDS NEVER GO THROUGH ITS BODY (7 Nov, Randy: "The hands shouldn't move through the body either"): read on the
    character itself, with the motion rmo fit wrote for it. Each body part (the vertices its bone moves most: hips, spine, chest, neck,
    head, thighs) is its convex hull at rest; each frame every vertex of both hands and forearms, posed as web/retarget.js poses it, is
    taken into each part's rest frame; a frame fails when 5 or more are inside one. The fit's own bodies are elliptic cylinders: this
    check is the mesh's hull, so a fit that is too small is caught. Plant: the left wrist driven into the chest's middle for a tenth
    of the clip (two-bone IK)"""
    if not os.path.exists(FITTED): return None, f'no motion fitted to {os.path.basename(CHAR)} (rmo fit): not read'
    sys.path.insert(0, os.path.join(HERE, '..', 'mocap')); from character import Character; from scipy.spatial import Delaunay; import fit as FIT
    Ch = Character(CHAR, hide=HIDE); Mf = json.load(open(FITTED)); W0 = {i: Ch.world[i] for i in range(len(Ch.names))}
    hull, arms = {}, []
    for part in Ch.parts:
        V = Ch.skin(W0, part)
        for b in FIT.BODY:
            m = part['dom'] == Ch.bone.get(b, -1)
            if m.any(): hull.setdefault(b, []).append(V[m])
        sub = np.isin(part['dom'], [Ch.bone[s + k] for s in ('Left', 'Right') for k in ('Hand', 'ForeArm')])
        if sub.any(): arms.append({kk: (v[sub] if isinstance(v, np.ndarray) and len(v) == len(part['P']) else v) for kk, v in part.items()})
    hull = {b: Delaunay(np.concatenate(v)) for b, v in hull.items() if sum(len(x) for x in v) >= 12}
    n = Mf['frames']; bad = 0; plant_at = set(range(n // 3, n // 3 + max(2, n // 10))) if plant else set()
    for f in range(n):
        Wp = Ch.pose(Mf, f)
        if f in plant_at:                                                                          # the wrist into the chest's middle
            S_, E_, Wr = (Wp[Ch.bone['Left' + b]][:3, 3] for b in ('Arm', 'ForeArm', 'Hand'))
            E2, W2 = FIT.two_bone(S_, E_, Wr, Wp[Ch.bone['Spine1']][:3, 3]); Mp = dict(Mf); Mp['world'] = dict(Mf['world'])
            for bone, old, new in (('LeftArm', E_ - S_, E2 - S_), ('LeftForeArm', Wr - E_, W2 - E2)):
                q = list(Mp['world'][bone]); q[f] = list(Rot.from_matrix(FIT.swing(old, new) @ Rot.from_quat(q[f]).as_matrix()).as_quat()); Mp['world'][bone] = q
            Wp = Ch.pose(Mp, f)
        P = np.concatenate([Ch.skin(Wp, a) for a in arms]); Ph = np.c_[P, np.ones(len(P))].T
        for b, h in hull.items():
            i = Ch.bone[b]; Q = (Ch.world[i] @ np.linalg.solve(Wp[i], Ph)).T[:, :3]
            if (h.find_simplex(Q) >= 0).sum() >= 5: bad += 1; break
    return bad == 0, f'the hands or forearms of {os.path.basename(CHAR)} inside its body in {bad} of {n} frames'
def r14(plant=False):
    """THE HEAD TURNS WHERE THE FACE TURNS (7 Nov: "The head must turn to the back too when the characters' head turns"): where the picture
    shows the face to one side of the head (from the ear the model is sure of, or the ears' middle, to the eyes' middle, a direction more
    than 60 % sideways: a face in profile lies forward of its ear), the character's head (the motion it plays) faces that side in 90 % of
    those frames or more. Read from the picture alone, not from the solve's own fit of the face (the nose against the eyes was tried and
    read a profile looking down as turned the other way: frame 125 of Mixkit 34382). Plant: the head turned the other way (mirrored)"""
    Mm = json.load(open(FITTED)) if os.path.exists(FITTED) else M
    k = Kp; sc = np.array([r['s'] if 's' in r else [0] * 133 for r in T['frames']], float)
    eyes = (k[:, 1] + k[:, 2]) / 2; s3, s4 = sc[:, 3] > 0.5, sc[:, 4] > 0.5
    ear = np.where((s3 & s4)[:, None], (k[:, 3] + k[:, 4]) / 2, np.where(s3[:, None], k[:, 3], k[:, 4]))
    v = eyes - ear; L = np.linalg.norm(v, axis=-1); side = v[:, 0] / np.maximum(L, 1e-6)
    ok = (sc[:, [1, 2]] > 0.5).all(1) & (s3 | s4) & (L > 4) & (np.abs(side) > 0.6)
    real = Mm['video_per_real_second']; a0 = Mm['clip_from'] or 0
    idx = np.clip(np.round((np.arange(len(k)) - a0) / real * Mm['fps']).astype(int), 0, Mm['frames'] - 1); ok &= np.arange(len(k)) >= a0
    if ok.sum() < 10: return None, f'the face to one side in {int(ok.sum())} frames: not read'
    head = Rot.from_quat(np.array(Mm['world']['Head'])).apply([0, 0, 1.0])[idx]                     # where the head faces (x: the picture's right)
    hx = -head[:, 0] if plant else head[:, 0]
    agree = float((np.sign(hx[ok]) == np.sign(side[ok])).mean())
    return agree >= 0.9, f'the head faces the side the face is on in {100 * agree:.0f}% of {int(ok.sum())} frames'
RULES = [('R1 TRACKED', lambda: r1(T), lambda: r1(p1(T))), ('R2 RIGID', lambda: r2(X), lambda: r2(p2(X))), ('R3 ANATOMY', lambda: r3(XW), lambda: r3(p3(XW))),
         ('R4 CLOCK', lambda: r4(9.81), lambda: r4(9.81 / 2)), ('R5 SOLVED', lambda: r5(M['world']), lambda: r5(p5(M['world']))),
         ('R6 SMOOTH', lambda: r6(M['world']), lambda: r6(p6(M['world']))),
         ('R6 FITTED', lambda: r6(json.load(open(FITTED))['world']) if os.path.exists(FITTED) else (None, 'no fitted motion: not read'), lambda: r6(p6(json.load(open(FITTED))['world'])) if os.path.exists(FITTED) else (None, '')), ('R7 HOSTS', lambda: r7(False), lambda: r7(True)),
         ('R8 SCALE', lambda: r8(False), lambda: r8(True)), ('R9 HANDS', lambda: r9(X), lambda: r9(p9(X))), ('R10 TWIST', lambda: r10(X), lambda: r10(p10(X))),
         ('R11 PLACED', lambda: r11(M['world']), lambda: r11(p11(M['world']))), ('R12 STRAIGHT', lambda: r12(X), lambda: r12(p12(X))),
         ('R13 CLEAR', lambda: r13(False), lambda: r13(True)), ('R14 HEAD', lambda: r14(False), lambda: r14(True))]
held = True
for name, real, plant in RULES:
    ok, say = real(); pok, psay = plant()
    if ok is None: print(f'  -    {name:<11} {say}'); continue
    good = ok and not pok; held = held and good
    print(f"  {'HOLDS' if good else 'RED  '} {name:<11} {say}  |  plant: {'red, as it must be' if not pok else 'STILL GREEN: the check cannot fail'} ({psay})")
print('RULES HOLD' if held else 'RULES RED'); sys.exit(0 if held else 1)
