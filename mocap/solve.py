"""rmo solve <clean.json> <out dir> [--fps 60] [--from F] [--to F]

THE TRACKED BODY AS ANIMATION. Each bone's turn from its T-pose, the way every mocap solver does it: the bone aims at its child joint
(swing) and a second direction fixes its twist (the hips' and chest's left-right line, the plane an elbow or knee bends in, the
hand's knuckle line, the foot's toe line). The skeleton is Mixamo's (its bone names, a T-pose facing +Z, +X its left, +Y up) with this
rider's measured bone lengths. Writes:
  motion.bvh    local rotations (Z X Y, degrees) + the hips' place (metres): Blender imports it as it is, three.js reads it with BVHLoader
  motion.json   each bone's turn in the WORLD from its T-pose (quaternions x y z w) + the hips' place: web/retarget.js puts it on any
                T-posed character (Mixamo, a Polyfork or Blender rig) whatever its bones' own axes
The real clock (clean.json's slow-motion factor) is resampled to --fps."""
import sys, os, json, argparse
import numpy as np
from scipy.spatial.transform import Rotation as Rot, Slerp

ap = argparse.ArgumentParser(); ap.add_argument('clean'); ap.add_argument('out'); ap.add_argument('--fps', type=float, default=60)
ap.add_argument('--from', dest='a', type=int, default=0); ap.add_argument('--to', dest='b', type=int, default=None)
ap.add_argument('--hands', action='store_true', help='turn the wrists and fingers by the hand keypoints (only where the hands are large and sharp in the picture)'); A = ap.parse_args()
D = json.load(open(A.clean)); X0 = np.array(D['X'])[A.a:A.b]; real = D['real_fps']
t0 = np.arange(len(X0)) / real; T = np.arange(0, t0[-1] + 1e-9, 1 / A.fps)
X = np.stack([np.stack([np.interp(T, t0, X0[:, j, c]) for c in range(3)], -1) for j in range(133)], 1)   # (frames, 133, 3)
n = len(T); os.makedirs(A.out, exist_ok=True)
nz = lambda v: v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), 1e-9)
UNSEEN = {}
def fillv(v, tag=None):
    """a direction the camera did not see (his shoulders above the frame in the jump) carried from the last frame it was seen to the
    next, component by component, held at the ends; the share of frames bridged is reported per bone"""
    v = np.array(v, float); ok = np.isfinite(v).all(-1)
    if tag: UNSEEN[tag] = round(float(1 - ok.mean()), 3)
    if ok.all() or not ok.any(): return v
    idx = np.arange(len(v))
    for c in range(v.shape[-1]): v[:, c] = np.interp(idx, idx[ok], v[ok, c])
    return v
mid = lambda a, b: (X[:, a] + X[:, b]) / 2
HAND = {'L': 91, 'R': 112}
def P(name):                                             # a named point of the tracked body
    s = {'L': 0, 'R': 1}
    table = {'hips': mid(11, 12), 'chest': mid(5, 6), 'head': mid(3, 4), 'nose': X[:, 0]}
    if name in table: return table[name]
    side, part = name[0], name[1:]
    k = {'shoulder': (5, 6), 'elbow': (7, 8), 'wrist': (9, 10), 'hip': (11, 12), 'knee': (13, 14), 'ankle': (15, 16), 'bigtoe': (17, 20), 'smalltoe': (18, 21), 'heel': (19, 22)}
    if part in k: return X[:, k[part][s[side]]]
    if part == 'toe': return (P(side + 'bigtoe') + P(side + 'smalltoe')) / 2
    if part == 'ball': return P(side + 'heel') + 0.7 * (P(side + 'toe') - P(side + 'heel'))
    if part.startswith('h'): return X[:, HAND[side] + int(part[1:])]   # hand keypoint 0-20
    raise KeyError(name)

def frame_from(a, b, tag=None):                           # columns: a, b made perpendicular to a, a x b
    a = nz(fillv(a, tag)); b = fillv(b); b = b - (b * a).sum(-1, keepdims=True) * a
    strength = np.linalg.norm(b, axis=-1) / np.maximum(np.nanmedian(np.linalg.norm(b, axis=-1)), 1e-9)
    weak = np.linalg.norm(b, axis=-1) < 1e-3
    b = nz(b)
    for t in range(1, len(b)):                            # a nearly straight limb's bend plane may read on either side: it keeps the side it had
        if strength[t] < 0.35 and (b[t] * b[t - 1]).sum() < 0: b[t] = -b[t]
    for t in range(1, len(b)):                            # a straight limb has no bend plane: keep the last one
        if weak[t] or not np.isfinite(b[t]).all(): q = b[t - 1] - (b[t - 1] @ a[t]) * a[t]; b[t] = q / max(np.linalg.norm(q), 1e-9)
    return np.stack([a, b, np.cross(a, b)], -1)           # (n, 3, 3), columns
def rest_frame(a, b):
    a = np.asarray(a, float); a = a / np.linalg.norm(a); b = np.asarray(b, float); b = b - (b @ a) * a; b = b / np.linalg.norm(b)
    return np.stack([a, b, np.cross(a, b)], -1)
def delta(M, R0): return np.einsum('nij,kj->nik', M, R0)   # M · R0^T: the world turn from the rest frame

med = lambda v: float(np.nanmedian(np.linalg.norm(v, axis=-1)))
L = {'thigh': (med(P('Lknee') - P('Lhip')) + med(P('Rknee') - P('Rhip'))) / 2, 'shin': (med(P('Lankle') - P('Lknee')) + med(P('Rankle') - P('Rknee'))) / 2,
     'upper': (med(P('Lelbow') - P('Lshoulder')) + med(P('Relbow') - P('Rshoulder'))) / 2, 'fore': (med(P('Lwrist') - P('Lelbow')) + med(P('Rwrist') - P('Relbow'))) / 2,
     'hiphalf': med(P('Lhip') - P('Rhip')) / 2, 'shhalf': med(P('Lshoulder') - P('Rshoulder')) / 2, 'trunk': med(P('chest') - P('hips')), 'neck': med(P('head') - P('chest')),
     'foot': (med(P('Ltoe') - P('Lheel')) + med(P('Rtoe') - P('Rheel'))) / 2, 'ankle_h': 0.075, 'hand': (med(P('Lh9') - P('Lh0')) + med(P('Rh9') - P('Rh0'))) / 2}

# the world frames of the trunk and head
x_hips = P('Lhip') - P('Rhip'); up = fillv(P('chest') - P('hips'))
Mhips = frame_from(x_hips, up, 'Hips'); Mhips[:, :, 2] = np.cross(Mhips[:, :, 0], Mhips[:, :, 1])        # the hip line exact (x, his left), up made square to it, z forward
Mchest = frame_from(P('Lshoulder') - P('Rshoulder'), up, 'Spine2'); Mchest[:, :, 2] = np.cross(Mchest[:, :, 0], Mchest[:, :, 1])
# THE HEAD (7 Nov, his word on the MOTION cut: "the character is flailing in the head"). Its line across was the ears', and ears hidden,
# swapped or out of the frame flipped it between frames: the head turned 8,627°/s of real time on Mixkit 34382 (rule R6 reads it now on the
# free clips). A head turns at most about 80° either way of the chest, bends about 50° forward and 60° back, and 45° to a side (the cervical
# spine's range of motion: AAOS, "Joint Motion: Method of Measuring and Recording", 1965; Youdas et al., Phys Ther 72(11), 1992). So the ears'
# line is turned to the chest's side when it points the other way (the ears swapped: no neck turns 90°), the head's turn against the chest
# is held inside that range, its spikes taken out (a median of 5) and low-passed at 3 Hz of real time, and the head is put back on the chest.
ear = X[:, 3] - X[:, 4]
ear[np.nan_to_num(np.einsum('ij,ij->i', ear, Mchest[:, :, 0])) < 0] *= -1
Mhead = frame_from(P('head') - P('chest'), ear, 'Head')[:, :, [1, 0, 2]]; Mhead[:, :, 2] = np.cross(Mhead[:, :, 0], Mhead[:, :, 1])
from scipy.signal import medfilt as _medfilt, butter as _bw, filtfilt as _ff
HEAD_RANGE = np.radians([[-80, 80], [-50, 50], [-45, 45]])     # yaw (about the chest's up), pitch (about its side; 50 the lesser of 50/60), roll
_e = Rot.from_matrix(np.einsum('nji,njk->nik', Mchest, Mhead)).as_euler('YXZ')
for _k in range(3):
    _v = _e[:, _k]; _ok = np.isfinite(_v)
    if _ok.sum() >= 2: _v = np.interp(np.arange(n), np.flatnonzero(_ok), _v[_ok])
    _v = np.clip(np.unwrap(np.nan_to_num(_v)), *HEAD_RANGE[_k])
    if n >= 9:
        _v = _medfilt(_v, 5); _b, _a = _bw(2, min(0.99, 3.0 / (A.fps / 2))); _v = _ff(_b, _a, _v)
    _e[:, _k] = np.clip(_v, *HEAD_RANGE[_k])
# where the face is not seen (the nose or an ear out of the picture or hidden: Mixkit 34382's last third, 12-27 % of its frames) its turn is a
# guess over bridged points, and the head tipped 45° there: it follows the chest instead, weighted by how much of the face the last quarter
# second saw
_vis = np.isfinite(X[:, [0, 3, 4], 0]).all(1).astype(float); _w = max(1, int(round(0.25 * A.fps)))
_vis = np.convolve(np.pad(_vis, _w // 2, mode='edge'), np.ones(_w) / _w, 'same')[_w // 2: _w // 2 + n]
_e = _e * _vis[:, None]
Mhead = np.einsum('nij,njk->nik', Mchest, Rot.from_euler('YXZ', _e).as_matrix())
I3 = np.eye(3)
W = {'Hips': Mhips, 'Spine2': Mchest, 'Head': Mhead}                                       # rest frames are the identity
qs = lambda M: Rot.from_matrix(M)
for name, f in (('Spine', 1 / 3), ('Spine1', 2 / 3)):
    W[name] = np.stack([Slerp([0, 1], qs(np.stack([Mhips[t], Mchest[t]])))(f).as_matrix() for t in range(n)])
W['Neck'] = np.stack([Slerp([0, 1], qs(np.stack([Mchest[t], Mhead[t]])))(0.5).as_matrix() for t in range(n)])
REST = {}
for s, sx in (('L', 1.0), ('R', -1.0)):
    W[('Left' if s == 'L' else 'Right') + 'Shoulder'] = Mchest
    side = 'Left' if s == 'L' else 'Right'
    # the arm: aims at the elbow; twist from the bend toward the wrist (rest: the forearm bends forward, +Z)
    fwd = Mchest[:, :, 2] * 0.3 * L['fore']                                                 # a straight arm has no bend plane: the chest's forward steadies it
    W[side + 'Arm'] = delta(frame_from(P(s + 'elbow') - P(s + 'shoulder'), fillv(P(s + 'wrist') - P(s + 'elbow')) + fwd, side + 'Arm'), rest_frame([sx, 0, 0], [0, 0, 1]))
    # the forearm twists with the elbow's own bend (the elbow plane's normal); the knuckle line of a small, blurred hand spun it at 8,800°/s
    up_, fo_ = fillv(P(s + 'elbow') - P(s + 'shoulder')), fillv(P(s + 'wrist') - P(s + 'elbow'))
    W[side + 'ForeArm'] = delta(frame_from(fo_, np.cross(up_, fo_ + fwd), side + 'ForeArm'), rest_frame([sx, 0, 0], [0, -sx, 0]))
    knuck = P(s + 'h5') - P(s + 'h17')                                                       # index knuckle minus little-finger knuckle: forward in the T-pose
    if A.hands:                                                                              # wrists and fingers by the hand keypoints; else they stay with the forearm
        W[side + 'Hand'] = delta(frame_from(P(s + 'h9') - P(s + 'h0'), knuck), rest_frame([sx, 0, 0], [0, 0, 1]))
        for fi, (fname, base) in enumerate((('Thumb', 1), ('Index', 5), ('Middle', 9), ('Ring', 13), ('Pinky', 17))):
            for k in range(3):
                a = P(s + f'h{base + k + 1}') - P(s + f'h{base + k}')
                rest_dir = [sx, 0, 0.6] if fname == 'Thumb' else [sx, 0, 0]
                W[f'{side}Hand{fname}{k + 1}'] = delta(frame_from(a, knuck), rest_frame(rest_dir, [0, 0, 1] if fname != 'Thumb' else [-sx, 0, 1]))
    else:
        # THE PALMS (7 Nov, his word on the MOTION cut: "The hands and palms in the wrong position too"). Hands 35-65 px tall cannot be read,
        # and the hand had copied the forearm's twist (the elbow's bend plane): the palms faced anywhere. A hand that cannot be seen is held in
        # the forearm's neutral turn, the palm toward the trunk's middle line, as a relaxed arm holds it (the T-pose rest: palms down), its
        # twist low-passed like the other bones' (below)
        palm = fillv((P('chest') + P('hips')) / 2 - P(s + 'wrist'))
        W[side + 'Hand'] = delta(frame_from(fo_, palm, side + 'Hand'), rest_frame([sx, 0, 0], [0, -1, 0]))
    # the leg: aims at the knee; twist from the shin's bend (rest: a bent knee sends the shin back, -Z)
    W[side + 'UpLeg'] = delta(frame_from(fillv(P(s + 'knee') - P(s + 'hip'), side + 'UpLeg'), P(s + 'ankle') - P(s + 'knee') - (P(s + 'toe') - P(s + 'heel')) * 0.05), rest_frame([0, -1, 0], [0, 0, -1]))
    # the shin twists with the knee's hinge: its reference is the thigh's direction (square to the shin it points forward); the foot's line
    # only steadies it once the knee is straight (at a flick the foot points down the shin and its line says nothing)
    W[side + 'Leg'] = delta(frame_from(P(s + 'ankle') - P(s + 'knee'), fillv(P(s + 'knee') - P(s + 'hip')) + 0.3 * fillv(P(s + 'toe') - P(s + 'heel')), side + 'Leg'), rest_frame([0, -1, 0], [0, 0, 1]))
    # the foot rolls with the leg (an ankle turns in or out about 20° at most): its twist from the knee's hinge line, which points to his
    # left for both legs; two toe points 2-3 cm apart swap between frames and spun the foot half a turn
    hinge = np.cross(fillv(P(s + 'knee') - P(s + 'hip')), fillv(P(s + 'ankle') - P(s + 'knee')) + 0.3 * fillv(P(s + 'toe') - P(s + 'heel')))
    across = -sx * hinge                                                                      # toward the other foot: -X for the left in the T-pose
    foot_rest = [0, -(L['ankle_h'] - 0.02), 0.7 * L['foot'] - 0.05]
    W[side + 'Foot'] = delta(frame_from(P(s + 'ball') - P(s + 'ankle'), across), rest_frame(foot_rest, [-sx, 0, 0]))
    W[side + 'ToeBase'] = delta(frame_from(P(s + 'toe') - P(s + 'ball'), across), rest_frame([0, 0, 1], [-sx, 0, 0]))

# A LIMB'S TWIST, SMOOTHED ON ITS OWN: each limb's world turn split into swing (where the bone points: measured, kept exactly) and twist
# about the bone's own axis (read from a bend plane or a small, blurred hand: the noisy part); the twist unwrapped, its single-frame
# spikes taken out (median of 5) and low-passed at 6 Hz of real time. The bone's line is untouched, so every joint stays where it was.
from scipy.signal import medfilt, butter as _butter, filtfilt as _filtfilt
def smooth_twist(Wb, a0):
    if os.environ.get("RMO_NO_TWIST_SMOOTH"): return Wb
    q = Rot.from_matrix(Wb); a0 = np.asarray(a0, float) / np.linalg.norm(a0); cur = q.apply(a0)
    ax = np.cross(a0, cur); s_ = np.linalg.norm(ax, axis=1); c_ = cur @ a0
    swing = Rot.from_rotvec(ax / np.maximum(s_, 1e-9)[:, None] * np.arctan2(s_, c_)[:, None])
    th = np.unwrap((swing.inv() * q).as_rotvec() @ a0)
    if len(th) > 15:
        th = medfilt(th, 5); b_, a_ = _butter(2, min(0.99, 6.0 / (A.fps / 2))); th = _filtfilt(b_, a_, th)
    return (swing * Rot.from_rotvec(np.outer(th, a0))).as_matrix()
for side, sx in (('Left', 1.0), ('Right', -1.0)):
    foot_rest = np.array([0, -(L['ankle_h'] - 0.02), 0.7 * L['foot'] - 0.05])
    for bone, a0 in (('Arm', [sx, 0, 0]), ('ForeArm', [sx, 0, 0]), ('Hand', [sx, 0, 0]), ('UpLeg', [0, -1, 0]), ('Leg', [0, -1, 0]), ('Foot', foot_rest), ('ToeBase', [0, 0, 1])):
        W[side + bone] = smooth_twist(W[side + bone], a0)

# the skeleton (Mixamo's names and hierarchy) in its T-pose with this rider's lengths
tr = L['trunk']; SK = [('Hips', None, [0, 0, 0]), ('Spine', 'Hips', [0, 0.10 * tr, 0]), ('Spine1', 'Spine', [0, 0.25 * tr, 0]), ('Spine2', 'Spine1', [0, 0.25 * tr, 0]),
      ('Neck', 'Spine2', [0, 0.40 * tr, 0]), ('Head', 'Neck', [0, 0.5 * L['neck'], 0])]
for s, sx in (('Left', 1), ('Right', -1)):
    SK += [(s + 'Shoulder', 'Spine2', [sx * 0.02, 0.36 * tr, 0]), (s + 'Arm', s + 'Shoulder', [sx * (L['shhalf'] - 0.02), 0.04 * tr, 0]),
           (s + 'ForeArm', s + 'Arm', [sx * L['upper'], 0, 0]), (s + 'Hand', s + 'ForeArm', [sx * L['fore'], 0, 0])]
    for fname, base in (('Thumb', 1), ('Index', 5), ('Middle', 9), ('Ring', 13), ('Pinky', 17)):
        sd = 'L' if s == 'Left' else 'R'; seg = [med(P(sd + f'h{base + k + 1}') - P(sd + f'h{base + k}')) for k in range(3)]   # a finger's 4 points: 3 segments
        start = [sx * 0.03, 0, 0.02] if fname == 'Thumb' else [sx * L['hand'] * 0.9, 0, {'Index': 0.025, 'Middle': 0.008, 'Ring': -0.01, 'Pinky': -0.028}[fname]]
        prev = s + 'Hand'
        for k in range(3):
            d = np.array([sx, 0, 0.6 if fname == 'Thumb' else 0.0]); d = d / np.linalg.norm(d)
            off = start if k == 0 else list(d * seg[k - 1])
            SK.append((f'{s}Hand{fname}{k + 1}', prev, off)); prev = f'{s}Hand{fname}{k + 1}'
    # the root IS the hip joints' midpoint: the thighs start beside it, level
    SK += [(s + 'UpLeg', 'Hips', [sx * L['hiphalf'], 0, 0]), (s + 'Leg', s + 'UpLeg', [0, -L['thigh'], 0]), (s + 'Foot', s + 'Leg', [0, -L['shin'], 0]),
           (s + 'ToeBase', s + 'Foot', [0, -(L['ankle_h'] - 0.02), 0.7 * L['foot'] - 0.05])]
names = [b for b, _, _ in SK]; parent = {b: p for b, p, _ in SK}; off = {b: o for b, _, o in SK}
for b in names:                                          # a bone the body does not move by itself turns with its parent (names are parent-first)
    if b not in W: W[b] = W[parent[b]]
root = fillv(P('hips'))

# BVH: local = parent world^T · world, Z X Y (scipy's intrinsic 'ZXY' is Rz·Rx·Ry, BVH's order)
kids = {b: [c for c in names if parent[c] == b] for b in names}
lines = ['HIERARCHY']
def emit(b, depth):
    ind = '  ' * depth; o = off[b]
    lines.append(f"{ind}{'ROOT' if parent[b] is None else 'JOINT'} {b}"); lines.append(ind + '{')
    lines.append(f"{ind}  OFFSET {o[0]:.5f} {o[1]:.5f} {o[2]:.5f}")
    lines.append(f"{ind}  CHANNELS " + ('6 Xposition Yposition Zposition Zrotation Xrotation Yrotation' if parent[b] is None else '3 Zrotation Xrotation Yrotation'))
    for c in kids[b]: emit(c, depth + 1)
    if not kids[b]: lines.append(f"{ind}  End Site"); lines.append(ind + '  {'); lines.append(f"{ind}    OFFSET 0 {0.08 if b == 'Head' else 0} {0.05 if 'Toe' in b else 0}"); lines.append(ind + '  }')
    lines.append(ind + '}')
emit('Hips', 0)
eul = {}
for b in names:
    loc = W[b] if parent[b] is None else np.einsum('nji,njk->nik', W[parent[b]], W[b])
    e = Rot.from_matrix(loc).as_euler('ZXY', degrees=False); eul[b] = np.degrees(np.unwrap(e, axis=0))
lines += ['MOTION', f'Frames: {n}', f'Frame Time: {1 / A.fps:.6f}']
for t in range(n):
    row = [*root[t]]
    for b in names: row += list(eul[b][t])
    lines.append(' '.join(f'{v:.4f}' for v in row))
open(os.path.join(A.out, 'motion.bvh'), 'w').write('\n'.join(lines) + '\n')
json.dump({'fps': A.fps, 'frames': n, 'source': D['video'], 'slowmo': D['slowmo'], 'clip_from': A.a, 'clip_to': A.b, 'video_per_real_second': real, 'bones': names, 'parent': [parent[b] for b in names], 'rest_offset': [off[b] for b in names],
           'hips': np.round(root, 4).tolist(), 'world': {b: np.round(Rot.from_matrix(W[b]).as_quat(), 5).tolist() for b in names}, 'lengths': L},
          open(os.path.join(A.out, 'motion.json'), 'w'), separators=(',', ':'))
# the check: forward kinematics of what was written puts the joints back where they were tracked
def fk(t):
    G, pos = {}, {}
    for b in names:
        p = parent[b]; Rw = W[b][t]
        pos[b] = root[t] if p is None else pos[p] + W[p][t] @ np.array(off[b]); G[b] = Rw
    return pos
tracked = {'LeftLeg': 'Lknee', 'RightLeg': 'Rknee', 'LeftFoot': 'Lankle', 'RightFoot': 'Rankle', 'LeftForeArm': 'Lelbow', 'RightForeArm': 'Relbow', 'LeftHand': 'Lwrist', 'RightHand': 'Rwrist'}
err = {k: float(np.nanmean([np.linalg.norm(fk(t)[k] - P(v)[t]) for t in range(0, n, max(1, n // 60))])) for k, v in tracked.items()}
print(json.dumps({'frames': n, 'fps': A.fps, 'bones': len(names), 'fk_error_cm': {k: round(v * 100, 1) for k, v in err.items()}, 'unseen_share': {k: v for k, v in UNSEEN.items() if v > 0}}))
