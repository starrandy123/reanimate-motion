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
  R7 HOSTS    Blender's keyed character and three.js's retarget agree within 2 cm    plant: Blender's keys one frame late"""
import sys, os, json, copy, subprocess
import numpy as np
from scipy.spatial.transform import Rotation as Rot
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, '..', 'mocap')); from clock import fit_slowmo
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
    for k in range(f, f + 2): Wd['LeftUpLeg'][k], Wd['RightUpLeg'][k] = Wd['RightUpLeg'][k], Wd['LeftUpLeg'][k]
    return Wd
def r7(plant):
    baked = opt('--baked'); char = opt('--character', os.path.join(HERE, '..', 'examples', 'models', 'street-skater.glb'))
    if not baked or not os.path.exists(baked): return None, 'no Blender-keyed character given (--baked): not read'
    cmd = ['node', os.path.join(HERE, 'hosts.mjs'), os.path.join(W, 'solve', 'motion.json'), char, baked] + (['--plant-late'] if plant else [])
    out = subprocess.run(cmd, capture_output=True, text=True, timeout=300); d = json.loads(out.stdout.strip().splitlines()[-1] or '{}')
    return out.returncode == 0, f"the hosts agree within {d.get('worst_cm', float('nan')):.2f} cm over {d.get('bones')} bones"

XW = X[M['clip_from']:M['clip_to']]   # the frames that become the animation: what the kit ships is what the anatomy rule reads
RULES = [('R1 TRACKED', lambda: r1(T), lambda: r1(p1(T))), ('R2 RIGID', lambda: r2(X), lambda: r2(p2(X))), ('R3 ANATOMY', lambda: r3(XW), lambda: r3(p3(XW))),
         ('R4 CLOCK', lambda: r4(9.81), lambda: r4(9.81 / 2)), ('R5 SOLVED', lambda: r5(M['world']), lambda: r5(p5(M['world']))),
         ('R6 SMOOTH', lambda: r6(M['world']), lambda: r6(p6(M['world']))), ('R7 HOSTS', lambda: r7(False), lambda: r7(True))]
held = True
for name, real, plant in RULES:
    ok, say = real(); pok, psay = plant()
    if ok is None: print(f'  -    {name:<11} {say}'); continue
    good = ok and not pok; held = held and good
    print(f"  {'HOLDS' if good else 'RED  '} {name:<11} {say}  |  plant: {'red, as it must be' if not pok else 'STILL GREEN: the check cannot fail'} ({psay})")
print('RULES HOLD' if held else 'RULES RED'); sys.exit(0 if held else 1)
