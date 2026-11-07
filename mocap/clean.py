"""rmo clean <track.json> <clean.json> [--slowmo F] [--cutoff HZ]

THE BLEND AND THE CLEANING (Sports2D's order, BSD-3: a keypoint it is unsure of is a gap, Hampel, gaps filled, Butterworth), plus
what a trick needs:
  1  a keypoint the model scores under 0.3 is a gap (Sports2D's keypoint_likelihood_threshold)
  2  left and right put back: a limb that sits where the other side's limb was is swapped back (legs cross in a flip)
  3  the feet blended with the shoe tracker: where the model is unsure of a foot, or puts it away from the shoe the tracker holds,
     the foot is the tracker's, its shape carried from the last frame the two agreed
  4  Hampel: a value outside 1.96 robust sigmas of the median of its 7 neighbours is a gap (Sports2D's reject_outliers)
  5  gaps up to 100 frames filled (PCHIP: no overshoot)
  6  the depth's scale: the model gives each joint's depth in metres and its place in pixels; the pixels per metre are the ones that
     keep every limb bone the same length in every frame
  7  the clock: in the air the body's centre of mass (Dempster's segments, Winter 2009 table 4.1) falls at 9.81 m/s²; the fall it
     takes in the video's frames gives the slow-motion factor (or --slowmo when it is known)
  8  Butterworth, 4th order, zero-phase, at --cutoff Hz of REAL time (default 12: Sports2D's 6 Hz is for walking; a flick is faster)
Writes the 133 joints in metres in the camera's frame (x to the picture's right, y up, z toward the camera), the board, the clock."""
import sys, json, argparse
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view as win
from scipy.signal import butter, filtfilt
from scipy.interpolate import PchipInterpolator
import os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); from clock import fit_slowmo; from scale import pick as pick_scale   # the kit's own modules (python -I leaves the script's folder off the path)

ap = argparse.ArgumentParser(); ap.add_argument('track'); ap.add_argument('out'); ap.add_argument('--slowmo', type=float)
ap.add_argument('--cutoff', type=float, default=12.0); ap.add_argument('--thr', type=float, default=0.3)
ap.add_argument('--2d', dest='two_d', action='store_true', help='ignore any model depth: every depth from bone lengths and anatomy (Taylor)'); A = ap.parse_args()
D = json.load(open(A.track)); fr = D['frames']; n, J = len(fr), 133; fps = D['fps']
K = np.full((n, J, 2), np.nan); Z = np.full((n, J), np.nan); S = np.zeros((n, J)); board = np.full((n, 4), np.nan); shoe = {s: np.full((n, 2), np.nan) for s in 'LR'}
for t, r in enumerate(fr):
    if 'k' in r: K[t] = r['k']; S[t] = r['s']; Z[t] = r['z'] if ('z' in r and not A.two_d) else 0.0   # 2D: a flat model depth, every sign from anatomy
    if 'board' in r: board[t] = r['board']
    for s in 'LR':
        if 'shoe' + s in r: shoe[s][t] = r['shoe' + s]
K[S < A.thr] = np.nan; Z[S < A.thr] = np.nan
# a point the model puts outside the picture (his shoulders above its top at the landing, scored up to 0.47) is its guess, not what the
# camera showed: unseen, and bridged like any other gap
Wd, Hd = D['size']; mg = 0.01 * min(Wd, Hd)
out_ = (K[..., 0] < mg) | (K[..., 0] > Wd - mg) | (K[..., 1] < mg) | (K[..., 1] > Hd - mg)
K[out_] = np.nan; Z[out_] = np.nan
report = {'frames': n, 'unsure': int(np.isnan(Z[:, :23]).sum()), 'depth_from': 'bone lengths and anatomy only (2D)' if A.two_d else 'bone lengths, anatomy, and the model for the rest'}

# 2 left and right put back, limb by limb
LEG = ([11, 13, 15, 17, 18, 19], [12, 14, 16, 20, 21, 22]); ARM = ([5, 7, 9] + list(range(91, 112)), [6, 8, 10] + list(range(112, 133)))
swaps = 0
for L, R in (LEG, ARM):
    last = {}
    for t in range(n):
        if t and all(k in last for k in 'LR'):
            d = lambda P, Q: np.nanmean(np.linalg.norm(P - Q, axis=-1)) if np.isfinite(P - Q).any() else np.inf
            same, cross = d(K[t, L], last['L']) + d(K[t, R], last['R']), d(K[t, L], last['R']) + d(K[t, R], last['L'])
            if np.isfinite(cross) and cross < 0.6 * same:
                for arr in (K, Z, S): arr[t, L], arr[t, R] = arr[t, R].copy(), arr[t, L].copy()
                swaps += 1
        for side, ids in (('L', L), ('R', R)):
            ok = np.isfinite(K[t, ids, 0])
            if ok.sum() >= 2: prev = last.get(side, np.full((len(ids), 2), np.nan)).copy(); prev[ok] = K[t, ids][ok]; last[side] = prev
report['swaps'] = swaps

# 3 the feet blended with the shoe tracker
legpx = np.nanmedian(np.linalg.norm(K[:, 11] - K[:, 13], axis=-1) + np.linalg.norm(K[:, 13] - K[:, 15], axis=-1)); size = 0.32 * legpx
from_shoe = {}
for s, ids in (('L', [15, 17, 18, 19]), ('R', [16, 20, 21, 22])):
    off, used = None, 0
    for t in range(n):
        c = shoe[s][t]
        if not np.isfinite(c).all(): continue
        kp = K[t, ids]; ok = np.isfinite(kp[:, 0])
        centre = kp[ok].mean(0) if ok.sum() >= 2 else None
        if ok.sum() >= 3 and np.linalg.norm(centre - c) < 0.6 * size: off = kp - c
        elif off is not None:
            K[t, ids] = c + np.where(np.isfinite(off), off, 0); Z[t, ids] = np.nan; used += 1   # the depth comes back by the fill
    from_shoe[s] = used
report['feet_from_the_shoe_tracker'] = from_shoe

seen0 = np.isfinite(K[..., 0]).copy()                    # what the camera really showed (before outliers and gaps are touched)
# 4 Hampel, 7 frames (Sports2D)
def hampel(x):
    if len(x) < 7: return x
    w = win(np.pad(x, 3, mode='edge'), 7); med = np.nanmedian(w, 1); mad = 1.4826 * np.nanmedian(np.abs(w - med[:, None]), 1)
    y = x.copy(); y[np.abs(x - med) > 1.96 * np.maximum(mad, 1e-6)] = np.nan; return y
# 5 gaps up to 100 frames filled
def fill(x, most=100):
    ok = np.isfinite(x)
    if ok.sum() < 4: return x
    y = x.copy(); idx = np.arange(len(x)); f = PchipInterpolator(idx[ok], x[ok], extrapolate=False); gap = ~ok
    runs = np.split(idx[gap], np.where(np.diff(idx[gap]) != 1)[0] + 1) if gap.any() else []
    for r in runs:
        if len(r) and len(r) <= most and r[0] > idx[ok][0] and r[-1] < idx[ok][-1]: y[r] = f(r)
    return y
for j in range(J):
    for c in range(2): K[:, j, c] = fill(hampel(K[:, j, c]))
    Z[:, j] = fill(hampel(Z[:, j]))
for c in range(4): board[:, c] = fill(board[:, c], 30)

BONES = [(5, 7), (7, 9), (6, 8), (8, 10), (11, 13), (13, 15), (12, 14), (14, 16), (5, 6), (11, 12), (5, 11), (6, 12)]
# which way the model's depth runs: the nose is nearer the camera than the ears when the face is seen
face = np.isfinite(Z[:, [0, 3, 4]]).all(1)
zs = -1.0 if A.two_d or face.sum() == 0 or np.nanmedian(Z[face, 0] - Z[face][:, [3, 4]].mean(1)) < 0 else 1.0   # the nose's depth below the ears': it grows away from the camera
root = np.nanmean(Z[:, [11, 12]], 1)
# 6 THE SCALE: the board when the leg it implies is a person's, else an adult's leg (mocap/scale.py). A box refused as the rider's board
# is not the rider's board anywhere after: not the scale, not the flight, not in the output
bw = board[:, 2] - board[:, 0]
legs2 = np.concatenate([np.linalg.norm(K[:, h_] - K[:, k_], axis=-1) + np.linalg.norm(K[:, k_] - K[:, a_], axis=-1) for h_, k_, a_ in ((11, 13, 15), (12, 14, 16))])
ppm, scale_from, refused = pick_scale(bw, np.nanpercentile(legs2[np.isfinite(legs2)], 95))
if refused: report['board_refused'] = refused; board[:] = np.nan
# 7 the clock, from gravity: the centre of mass's fall in the air
X = np.zeros((n, J, 3)); X[..., 0] = (K[..., 0] - np.nanmedian(K[:, [11, 12], 0])) / ppm; X[..., 1] = -(K[..., 1]) / ppm   # heights: all the clock reads
def com(P):
    mid = lambda a, b: (P[:, a] + P[:, b]) / 2
    sh, hp = mid(5, 6), mid(11, 12)
    # a part out of the picture (his head leaves the top of the frame in the jump) is placed from the trunk's own line, never dropped:
    # dropping it would move the centre of mass by its share and bend the fall
    head = np.where(np.isfinite(mid(3, 4)), mid(3, 4), sh + 0.35 * (sh - hp))
    near = lambda j, alt: np.where(np.isfinite(P[:, j]), P[:, j], alt)
    seg = [(0.081, head), (0.497, hp + 0.5 * (sh - hp))]
    for s_, e, m, f in ((5, 7, 0.028, 0.436), (6, 8, 0.028, 0.436), (7, 9, 0.016, 0.430), (8, 10, 0.016, 0.430), (11, 13, 0.100, 0.433), (12, 14, 0.100, 0.433), (13, 15, 0.0465, 0.433), (14, 16, 0.0465, 0.433)):
        a_ = near(s_, sh if s_ in (5, 6) else hp); e_ = near(e, a_)
        seg.append((m, a_ + f * (e_ - a_)))
    seg += [(0.0145, mid(15, 17)), (0.0145, mid(16, 20)), (0.006, near(9, sh)), (0.006, near(10, sh))]
    M = sum(m for m, _ in seg); return sum(m * p for m, p in seg) / M
C = com(X)
# the body's centre with the upper body CARRIED by the hips (68 % of the mass: trunk, head, arms) and the legs as tracked: in a trick the
# trunk's top leaves the picture, and the hips alone are no clock (the knees come up as he rises and go down to land: their arc flattens)
hipm = (X[:, 11] + X[:, 12]) / 2
legs = sum(m * (X[:, s_] + f * (X[:, e] - X[:, s_])) for s_, e, m, f in ((11, 13, 0.100, 0.433), (12, 14, 0.100, 0.433), (13, 15, 0.0465, 0.433), (14, 16, 0.0465, 0.433)))
legs = legs + 0.0145 * (X[:, 15] + X[:, 17]) / 2 + 0.0145 * (X[:, 16] + X[:, 20]) / 2
C2 = 0.682 * hipm + legs
feetlow = np.nanmin(X[:, [15, 16, 17, 18, 19, 20, 21, 22], 1], 1)
ground = np.nanmedian(feetlow[feetlow < np.nanpercentile(feetlow, 60)])
bot = board[:, 3]; bot_m = -bot / ppm                    # the board's lowest edge (its wheels), metres up
if np.isfinite(bot_m).sum() > 20:
    rest_b = np.nanmedian(bot_m[feetlow < ground + 0.03]); air = bot_m > rest_b + 0.04; flight_by = 'the board'
else:
    air = feetlow > ground + 0.12; flight_by = 'the feet'
air = np.nan_to_num(air, nan=0).astype(bool)
runs = np.split(np.arange(n)[air], np.where(np.diff(np.arange(n)[air]) != 1)[0] + 1) if air.any() else []
runs = [r for r in runs if len(r) >= 8]
slowmo, fit = A.slowmo, None
if runs and slowmo is None:
    r = max(runs, key=len); t = r.astype(float)
    whole = np.isfinite(C[r, 1]).mean() >= 0.6
    y = C[r, 1] if whole else C2[r, 1]; ok = np.isfinite(y); body = 'the centre of mass' if whole else 'the centre of mass, the upper body carried by the hips'
    report['clock_frames_held'] = int(ok.sum()); report['clock_body'] = body; report['flight_by'] = flight_by
    got = fit_slowmo(t, y, fps)
    if got:
        measured, slowmo, rms = got
        fit = {'from': int(r[0]), 'to': int(r[-1]), 'measured_slowmo': round(measured, 2), 'taken': slowmo, 'rms_cm': round(rms * 100, 2)}
slowmo = slowmo or 1.0
report.update(slowmo=round(slowmo, 3), clock_fit=fit)

# 8 Butterworth at --cutoff Hz of real time, on the PICTURE and the model's depth: the 3D is built last, from smooth inputs,
#   so nothing after it bends the bones it holds rigid or the knees it settled
wn = A.cutoff / slowmo / (fps / 2)
if wn < 1:
    b_, a_ = butter(4, wn)
    def bw_(x):
        ok = np.isfinite(x)
        if ok.sum() > 15:
            idx = np.where(ok)[0]
            for seg in np.split(idx, np.where(np.diff(idx) != 1)[0] + 1):
                if len(seg) > 15: x[seg] = filtfilt(b_, a_, x[seg])
    for j in range(J):
        for c in range(2): bw_(K[:, j, c])
        bw_(Z[:, j])
report['cutoff_hz_real'] = A.cutoff
# 6 DEPTH, TAYLOR'S WAY (C. J. Taylor, "Reconstruction of articulated objects from point correspondences in a single uncalibrated
#   image", CVPR 2000): under a scaled-orthographic camera a bone of length L that shows l pixels long reaches sqrt(L² - l²) toward or
#   away from the camera. L is the bone's length when it lies across the picture (the 95th percentile of its pictured length); the model's
#   own depth says only WHICH END is nearer (a monocular model's depth runs compressed: a thigh pointing at the camera showed half its
#   length and the model made up a third of it). The trunk and the head keep the model's depth; the hands and toes ride their joint's.
root = np.nanmean(Z[:, [11, 12]], 1)
Dm = Z - root[:, None]                                  # the model's depth from the mid-hip, metres
Dep = Dm.copy()
TREE = [(11, 13), (13, 15), (15, 19), (15, 17), (15, 18), (12, 14), (14, 16), (16, 22), (16, 20), (16, 21), (5, 7), (7, 9), (6, 8), (8, 10)]
def near_end(model_dz, dz, L):
    """which end of a bone is nearer the camera, frame by frame: the model's vote smoothed over 7 frames, and a change allowed only
    while the bone lies in the picture's plane (its depth under a tenth of its length): a bone pointing at the camera cannot swap
    ends in one frame, and a monocular model's sign flips there"""
    v = np.pad(model_dz, 3, mode='edge'); sm = np.nanmedian(win(v, 7), 1)
    out = np.ones(len(sm)); fin = sm[np.isfinite(sm) & (sm != 0)]; cur = np.sign(fin[0]) if len(fin) else 1.0
    for t in range(len(sm)):
        st = np.sign(sm[t]) if np.isfinite(sm[t]) and sm[t] != 0 else cur
        if st != cur and (not np.isfinite(dz[t]) or dz[t] < L / 10): cur = st   # a real swap passes through the picture's plane: at most ~11° of jump
        out[t] = cur
    return out
# which way he faces, from the picture: his left hip on the picture's right means he faces the camera. "In front of him" in the model's
# depth units is then facing * zs (X_z = zs * depth is toward the camera)
lat2 = K[:, 11, 0] - K[:, 12, 0]; facing = np.sign(lat2); face_ok = np.abs(lat2) > 0.5 * np.nanpercentile(np.abs(lat2), 95)
front = facing * zs
def carry(rule, sure, fallback):
    """a sign the rule is sure of where it is; elsewhere the side the bone had the frame before (a sign never flips on noise); before the
    first sure frame, the fallback's"""
    rule = np.broadcast_to(rule, sure.shape); out = np.array(fallback, float); last = None
    for t in range(len(out)):
        if sure[t]: last = rule[t]
        if last is not None: out[t] = last
    return out
# the trunk is a rigid bone too (mid-hip to mid-shoulder): its depth from its length, leaning where he faces (a rider's trunk bends
# forward); the model's own mid-shoulder depth lurched 12 cm as his shoulders came back into the frame
msh = (K[:, 5] + K[:, 6]) / 2; mhp = (K[:, 11] + K[:, 12]) / 2; d2t = np.linalg.norm(msh - mhp, axis=-1); Lt = np.nanpercentile(d2t[np.isfinite(d2t)], 95)
dzt = np.sqrt(np.maximum(0, Lt ** 2 - d2t ** 2)) / ppm; trunk_dep = carry(front, face_ok, np.ones(n)) * dzt
reach = {}
for a_, b_ in ((11, 12), (5, 6)):                         # the pelvis and the shoulders are rigid lines too, about their middles
    d2 = np.linalg.norm(K[:, a_] - K[:, b_], axis=-1); Lb = np.nanpercentile(d2[np.isfinite(d2)], 95)
    dz = np.sqrt(np.maximum(0, Lb ** 2 - d2 ** 2)) / ppm; sgn = near_end(Dm[:, a_] - Dm[:, b_], dz, Lb / ppm)
    c = 0.0 if a_ == 11 else trunk_dep
    Dep[:, a_] = c + sgn * dz / 2; Dep[:, b_] = c - sgn * dz / 2; reach[f'{a_}-{b_}'] = round(float(Lb / ppm), 3)
P3 = lambda j, dep: np.stack([K[:, j, 0] / ppm, -K[:, j, 1] / ppm, zs * dep], -1)
ANAT = {'hip': 0, 'knee': 0, 'toe': 0, 'heel': 0, 'arm': 0}
for p_, c_ in TREE:
    d2 = np.linalg.norm(K[:, c_] - K[:, p_], axis=-1)
    Lb = np.nanpercentile(d2[np.isfinite(d2)], 95); L_ = Lb / ppm
    dz = np.sqrt(np.maximum(0, Lb ** 2 - d2 ** 2)) / ppm
    sgn = near_end(Dm[:, c_] - Dm[:, p_], dz, L_)            # the model's vote, by default
    far = np.isfinite(dz) & (dz > L_ / 3) & face_ok
    kind = {13: 'hip', 14: 'hip', 15: 'knee', 16: 'knee', 17: 'toe', 18: 'toe', 20: 'toe', 21: 'toe', 19: 'heel', 22: 'heel'}.get(c_)
    if kind is None and A.two_d and c_ in (7, 8, 9, 10): kind = 'arm'   # no model to ask: an arm pointing at the camera reaches forward
    if kind == 'hip':                                        # the hip extends at most ~20° and a rider's hips are always flexed: the knee is IN FRONT
        sgn = carry(front, face_ok, sgn)
    elif kind == 'knee':                                     # the knee bends one way: of the two ankles, the one the knee flexes BACK toward
        hp_ = 11 if c_ == 15 else 12; kn = 13 if c_ == 15 else 14
        lat = P3(11, Dep[:, 11]) - P3(12, Dep[:, 12]); u = P3(kn, Dep[:, kn]) - P3(hp_, Dep[:, hp_])
        fl = [np.einsum('ij,ij->i', np.cross(u, P3(c_, Dep[:, p_] + sg * dz) - P3(kn, Dep[:, kn])), lat) for sg in (1.0, -1.0)]
        # a knee never bends backward: where exactly one of the two ankles bends it forward, that one. Where BOTH do (a thigh pointing at
        # the camera: 60° or 120° of bend look the same in the picture) the anatomy cannot tell, and the ankle keeps the side it had
        n_ = np.linalg.norm(u, axis=-1) * L_ * np.linalg.norm(lat, axis=-1) + 1e-9; ok1, ok2 = fl[0] / n_ > -0.05, fl[1] / n_ > -0.05
        sgn = carry(np.where(ok1, 1.0, -1.0), ok1 != ok2, sgn)
    elif kind == 'toe': sgn = carry(front, face_ok, sgn)     # the toes in front of the ankle
    elif kind == 'heel': sgn = carry(-front, face_ok, sgn)   # the heel behind it
    elif kind == 'arm': sgn = carry(front, face_ok, sgn)
    if kind: ANAT[kind] += int(far.sum())
    Dep[:, c_] = Dep[:, p_] + sgn * dz
    reach[f'{p_}-{c_}'] = round(float(L_), 3)
report['depth_set_by_anatomy'] = ANAT
face = list(range(0, 5)) + list(range(23, 91))           # the head and face ride the shoulders: only the model's small offsets from them
Dep[:, face] = trunk_dep[:, None] + (Dm[:, face] - (Dm[:, 5:6] + Dm[:, 6:7]) / 2)
for w_, base in ((9, 91), (10, 112)):                     # the hand's points ride the wrist
    Dep[:, base:base + 21] = Dep[:, w_:w_ + 1] + (Dm[:, base:base + 21] - Dm[:, w_:w_ + 1])
X = np.zeros((n, J, 3)); X[..., 0] = (K[..., 0] - np.nanmedian(K[:, [11, 12], 0])) / ppm; X[..., 1] = -(K[..., 1]) / ppm
X[..., 2] = zs * Dep                                    # toward the camera (zs: +1 when the model's depth grows toward it, -1 away)
# A JOINT THE CAMERA DID NOT SHOW is not given a depth by any rule: across its gap the bone's 3D direction turns evenly (slerp) from the
# last way it was seen to the next, at its own length (GEM-X and GLAMR infill unseen frames; a short gap wants no more than this).
# Parents first, so a hand follows its forearm through the gap
def bridge(p_, c_, L_):
    gap = ~seen0[:, c_]
    if not gap.any() or gap.all(): return
    v = X[:, c_] - X[:, p_]; u = v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), 1e-9); idx = np.arange(n)
    for run in np.split(idx[gap], np.where(np.diff(idx[gap]) != 1)[0] + 1):
        a0, b0 = run[0] - 1, run[-1] + 1
        ok_a = a0 >= 0 and bool(np.isfinite(u[a0]).all()); ok_b = b0 < n and bool(np.isfinite(u[b0]).all())
        if not (ok_a or ok_b): continue                            # neither end of the gap holds a bone to turn from: left unseen
        ua = u[a0] if ok_a else u[b0]; ub = u[b0] if ok_b else ua   # a gap at the clip's start or end holds the one end it has
        om = np.arccos(np.clip(ua @ ub, -1, 1))
        for k, t in enumerate(run):
            f = (k + 1) / (len(run) + 1)
            w = ua * (1 - f) + ub * f if om < 1e-4 else (np.sin((1 - f) * om) * ua + np.sin(f * om) * ub) / np.sin(om)
            X[t, c_] = X[t, p_] + L_ * w / max(np.linalg.norm(w), 1e-9)
for p_, c_ in TREE: bridge(p_, c_, reach[f'{p_}-{c_}'])
# every bone held at its own length (a frame where the picture shows it longer than it is: the detector's error, not the bone's): the
# child moved along the bone, parents first, so the skeleton the solver writes puts the joints back where they are
for p_, c_ in TREE:
    v = X[:, c_] - X[:, p_]; d = np.linalg.norm(v, axis=-1, keepdims=True); L_ = reach[f'{p_}-{c_}']
    moved = X[:, p_] + v / np.maximum(d, 1e-9) * L_; fin = np.isfinite(moved).all(-1); X[fin, c_] = moved[fin]
lens = {f'{a_}-{b_}': float(np.nanmedian(np.linalg.norm(X[:, a_] - X[:, b_], axis=-1))) for a_, b_ in BONES[:8]}
rig = [float(np.nanstd(np.linalg.norm(X[:, a_] - X[:, b_], axis=-1)) / max(1e-6, lens[f'{a_}-{b_}'])) for a_, b_ in BONES[:8]]
report.update(px_per_m=round(ppm, 1), scale_from=scale_from, depth_sign=zs,
              bone_length_spread=round(float(np.mean(rig)), 3), bones_m=reach)

Bm = np.full((n, 4), np.nan); Bm[:, 0] = ((board[:, 0] + board[:, 2]) / 2 - np.nanmedian(K[:, [11, 12], 0])) / ppm; Bm[:, 1] = -((board[:, 1] + board[:, 3]) / 2) / ppm
Bm[:, 2] = (board[:, 2] - board[:, 0]) / ppm; Bm[:, 3] = (board[:, 3] - board[:, 1]) / ppm
X[..., 1] -= ground; Bm[:, 1] -= ground
out = {'video': D['video'], 'fps': fps, 'slowmo': slowmo, 'real_fps': fps * slowmo, 'first_frame': D['from'], 'joints': 'COCO-WholeBody 133', 'units': 'metres; x right in the picture, y up from the ground, z toward the camera',
       'report': report, 'bones_m': lens,
       'X': np.round(X, 4).tolist(),                     # NaN where a joint was never seen near (read with Python's json: it keeps NaN)
       'board': np.round(np.nan_to_num(Bm, nan=0.0), 4).tolist(), 'board_seen': np.isfinite(Bm[:, 0]).astype(int).tolist(), 'air': air.astype(int).tolist()}
json.dump(out, open(A.out, 'w'), separators=(',', ':'), allow_nan=True)
print(json.dumps(report))
