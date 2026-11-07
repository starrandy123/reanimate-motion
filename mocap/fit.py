"""rmo fit <work> --character <glb> [--hide <node>] [--margin 0.015]   -> <work>/solve/motion_<character>.json

THE CHARACTER'S HANDS NEVER GO THROUGH ITS OWN BODY (7 Nov, Randy: "The hands shouldn't move through the body either. All this must be
handled"). A motion put on a character with other proportions (a wider body, shorter arms) can sink its hands into its own chest or hips
even where the rider's never touched his. The fix is the one Unreal's IK Retargeter ships as its Body Intersect Goals op (the Spatially
Aware Retarget Ops plugin, UE 5.7+: "moves specified IK goals based on collisions to prevent a character's body parts from intersecting";
its bodies are the target's physics-asset spheres and capsules), built here from its documented idea, not its code:
  1  THE BODIES: one per body bone (hips, the three spine bones, neck, head, both thighs), fitted to the vertices that bone moves most, as
     Unreal's physics-asset tool fits a body to a bone's dominant-weight vertices: their convex hull (a capsule round a boxy shirt left
     its corners out: 8 of 86 frames of Mixkit 34382 still had a hand in the body by the hull)
  2  THE GOALS: each hand's own vertices and its forearm's, posed as web/retarget.js poses them; a vertex inside a body asks the wrist to
     move by what takes it out through the facet it is least deep behind (a point's way out of a convex body), plus a margin
  3  THE ARM: solved again for the moved wrist with the analytic two-bone IK (law of cosines), the elbow kept in the plane it bent in;
     the hand keeps its own turn; where it is the elbow's end of the forearm that is in, the elbow is the goal and the upper arm swings it
     out. Five passes a round, twenty in the last. Every other bone is untouched.
Writes the character's own motion (world turns from its T-pose, as rmo solve writes them) beside motion.json; rmo film --motion reads it."""
import sys, os, json, argparse
import numpy as np
from scipy.spatial.transform import Rotation as Rot
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); from character import Character

BODY = ('Hips', 'Spine', 'Spine1', 'Spine2', 'Neck', 'Head', 'LeftUpLeg', 'RightUpLeg')
BAND = ('Hips', 'Spine', 'Spine1', 'Spine2', 'LeftUpLeg', 'RightUpLeg')
CHILD = {'Hips': 'Spine', 'Spine': 'Spine1', 'Spine1': 'Spine2', 'Spine2': 'Neck', 'Neck': 'Head', 'Head': None, 'LeftUpLeg': 'LeftLeg', 'RightUpLeg': 'RightLeg'}


def bodies(C):
    """each body bone's body: the convex hull of the vertices that bone moves most (Unreal's physics-asset tool fits a bone's body to its
    dominant-weight vertices), at rest, as its facets' planes (n . p + d <= 0 inside)"""
    from scipy.spatial import ConvexHull
    pts = {b: [] for b in BODY}; W0 = {i: C.world[i] for i in range(len(C.names))}
    for part in C.parts:
        V = C.skin(W0, part)
        for b in BODY:
            if b in C.bone: pts[b].append(V[part['dom'] == C.bone[b]])
    out = {}
    for b in BODY:
        P = np.concatenate(pts[b]) if pts[b] else np.zeros((0, 3))
        if len(P) < 12: continue
        h = ConvexHull(P); o = C.world[C.bone[b]][:3, 3]; ch = CHILD[b]
        ax = (C.world[C.bone[ch]][:3, 3] - o) if ch and ch in C.bone else np.array([0, 1.0, 0]); ax = ax / np.linalg.norm(ax)
        # the torso and the thighs are bands stacked end to end: a point leaves one through its SIDES (a facet within 60° of square to the
        # bone), never through a top or bottom that only opens into the next band (frame 76 of Mixkit 34382: an elbow in the spine's band
        # was pushed up into the chest's, then down into the hips', and back)
        side = np.abs(h.equations[:, :3] @ ax) < 0.5 if b in BAND else np.ones(len(h.equations), bool)
        out[b] = dict(eq=h.equations, side=side, n=len(P), extent=np.round(P.max(0) - P.min(0), 3))
    return out


def push_out(C, Bd, W, pts, margin):
    """the largest move that takes any of the points (posed, world) out of every body by the margin, through the nearest facet (a point's
    way out of a convex body: the facet it is least deep behind), the body it came from, and the point"""
    best, who, at = np.zeros(3), None, None
    Ph = np.c_[pts, np.ones(len(pts))].T
    for b, d in Bd.items():
        M = W[C.bone[b]]; R0 = C.world[C.bone[b]]
        Q = (R0 @ np.linalg.solve(M, Ph)).T[:, :3]                                                  # into the body's rest frame
        E = Q @ d['eq'][:, :3].T + d['eq'][:, 3]                                                    # each point's signed distance to each facet
        inside = (E < margin).all(1)
        if not inside.any(): continue
        k = np.flatnonzero(inside); depth = np.where(d['side'][None, :], margin - E[k], np.inf); j = depth.argmin(1); dd = depth[np.arange(len(k)), j]
        i = int(np.argmax(dd)); mv = d['eq'][j[i], :3] * dd[i]                                       # out through that facet
        mv_w = M[:3, :3] @ np.linalg.solve(R0[:3, :3], mv)                                          # rest frame -> posed world
        if np.linalg.norm(mv_w) > np.linalg.norm(best): best, who, at = mv_w, b, pts[k[i]]
    return best, who, at


def swing(a, b):
    a = a / np.linalg.norm(a); b = b / np.linalg.norm(b); ax = np.cross(a, b); s = np.linalg.norm(ax)
    if s < 1e-9: return np.eye(3)
    return Rot.from_rotvec(ax / s * np.arctan2(s, a @ b)).as_matrix()


def two_bone(S, E, Wr, T):
    """the elbow and wrist for a wrist target T, the arm's lengths kept, the elbow in the plane it bent in"""
    a, b = np.linalg.norm(E - S), np.linalg.norm(Wr - E); d = T - S; L = np.linalg.norm(d)
    L = np.clip(L, abs(a - b) + 1e-4, a + b - 1e-4); u = d / max(np.linalg.norm(d), 1e-9)
    pole = (E - S) - ((E - S) @ u) * u
    if np.linalg.norm(pole) < 1e-6: pole = np.cross(u, [0, 0, 1.0]) if abs(u[2]) < 0.9 else np.cross(u, [1.0, 0, 0])
    pole /= np.linalg.norm(pole); ca = np.clip((a * a + L * L - b * b) / (2 * a * L), -1, 1)
    E2 = S + a * (ca * u + np.sqrt(1 - ca * ca) * pole); return E2, S + L * u


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('work'); ap.add_argument('--character', required=True); ap.add_argument('--hide', action='append', default=[])
    ap.add_argument('--margin', type=float, default=0.015); ap.add_argument('--motion', default='motion.json')
    ap.add_argument('--goals', type=float, default=1.0, help='how far each hand goes to where the rider\'s hand is against the body (0: the turns only)'); A = ap.parse_args()
    M = json.load(open(os.path.join(A.work, 'solve', A.motion))); C = Character(A.character, hide=A.hide); Bd = bodies(C)
    W0 = {i: C.world[i] for i in range(len(C.names))}
    sel = {}                                                                                    # each side's hand and forearm vertices
    for s in ('Left', 'Right'):
        idx = []
        for k, part in enumerate(C.parts):
            m = np.isin(part['dom'], [C.bone[s + 'Hand'], C.bone[s + 'ForeArm']])
            if m.any(): idx.append((k, np.flatnonzero(m)))
        sel[s] = idx
    def posed_pts(W, s):
        out = []
        for k, ix in sel[s]:
            part = C.parts[k]; sub = {kk: (v[ix] if isinstance(v, np.ndarray) and len(v) == len(part['P']) else v) for kk, v in part.items()}
            out.append(C.skin(W, sub))
        return np.concatenate(out)
    world = {b: [list(q) for q in M['world'][b]] for b in M['bones']}
    Mf = dict(M); Mf['world'] = world; n = M['frames']; pushed = {'Left': 0, 'Right': 0}; worst = 0.0; hits = {}
    def frame_pass(f, s, count, passes=5):
        nonlocal worst
        for _ in range(passes):
            W = C.pose(Mf, f); mv, who, at = push_out(C, Bd, W, posed_pts(W, s), A.margin)
            if who is None: return
            if _ == 0 and count: pushed[s] += 1; hits[who] = hits.get(who, 0) + 1
            worst = max(worst, float(np.linalg.norm(mv)))
            S_, E_, Wr = (W[C.bone[s + b]][:3, 3] for b in ('Arm', 'ForeArm', 'Hand'))
            if (at - E_) @ (Wr - E_) < 0.35 * ((Wr - E_) @ (Wr - E_)):
                # the ELBOW's end of the forearm in the body (an elbow tucked into the side): the elbow is the goal; the upper arm swings
                # it out, and the forearm and hand, keeping their own turns, ride with it
                moves = ((s + 'Arm', E_ - S_, E_ + mv - S_),)
            else:
                E2, W2 = two_bone(S_, E_, Wr, Wr + mv); moves = ((s + 'Arm', E_ - S_, E2 - S_), (s + 'ForeArm', Wr - E_, W2 - E2))
            for bone, old, new in moves:
                R = swing(old, new) @ Rot.from_quat(world[bone][f]).as_matrix(); world[bone][f] = list(Rot.from_matrix(R).as_quat())
    # THE HANDS WHERE THE RIDER'S ARE (7 Nov, cut 2: "the hands are in the wrong position"). Turns copied bone by bone keep the arms' angles,
    # not the hands' places: Polyfork's skater has shoulders as wide as its trunk is long (the rider's are two thirds of it), so a karate
    # guard's fists, in front of the rider's sternum, landed out at the skater's sides. Unreal's IK Retargeter places each hand goal toward
    # where the source's hand is (its "Blend to Source", scaled to the target) and solves the arm to reach it: here each hand goes to the
    # skater's mid-shoulder plus the rider's hand's offset from his, scaled by the two reaches (half the shoulders plus the arm), the elbow
    # in the plane it bent in, by --goals of the way; the hand keeps its own turn. Then the body is kept clear, below
    if A.goals > 0:
        X0 = np.array(json.load(open(os.path.join(A.work, 'clean.json')))['X'], float)[M['clip_from']:M['clip_to']]
        t0 = np.arange(len(X0)) / M['video_per_real_second']; tt = np.arange(n) / M['fps']
        Xr = {j: np.stack([np.interp(tt, t0, X0[:, j, c]) for c in range(3)], -1) for j in (5, 6, 9, 10)}
        Lg = M['lengths']; reach_r = Lg['shhalf'] + Lg['upper'] + Lg['fore']
        pr = lambda nm: C.world[C.bone[nm]][:3, 3]
        reach_c = np.linalg.norm(pr('LeftArm') - pr('RightArm')) / 2 + np.linalg.norm(pr('LeftForeArm') - pr('LeftArm')) + np.linalg.norm(pr('LeftHand') - pr('LeftForeArm'))
        k_ = reach_c / reach_r
        for f in range(n):
            W = C.pose(Mf, f); cc = (W[C.bone['LeftArm']][:3, 3] + W[C.bone['RightArm']][:3, 3]) / 2; cr = (Xr[5][f] + Xr[6][f]) / 2
            for s_, wj in (('Left', 9), ('Right', 10)):
                if not np.isfinite(Xr[wj][f]).all() or not np.isfinite(cr).all(): continue
                S_, E_, Wr = (W[C.bone[s_ + b]][:3, 3] for b in ('Arm', 'ForeArm', 'Hand'))
                E2, W2 = two_bone(S_, E_, Wr, Wr + A.goals * (cc + k_ * (Xr[wj][f] - cr) - Wr))
                for bone, old, new in ((s_ + 'Arm', E_ - S_, E2 - S_), (s_ + 'ForeArm', Wr - E_, W2 - E2)):
                    R = swing(old, new) @ Rot.from_quat(world[bone][f]).as_matrix(); world[bone][f] = list(Rot.from_matrix(R).as_quat())
        # the goals ride the tracked wrists, whose depth can turn over between two frames: the placing is eased like the push below
        from scipy.signal import butter as _b, filtfilt as _f
        if n > 15:
            for b in [s_ + k for s_ in ('Left', 'Right') for k in ('Arm', 'ForeArm')]:
                Q0 = Rot.from_quat(np.array(M['world'][b])); dv = (Rot.from_quat(np.array(world[b])) * Q0.inv()).as_rotvec()
                world[b] = [list(q) for q in (Rot.from_rotvec(_f(*_b(2, min(0.99, 4.0 / (M['fps'] / 2))), dv, axis=0)) * Q0).as_quat()]
    # EASED, NOT SNAPPED: a correction made frame by frame switches on and off as a hand meets the body, and Mixkit 488's right arm turned
    # 3,537°/s doing it (rule R6). So each arm bone's correction (the turn from its solved world turn to its moved one) is low-passed at 4 Hz
    # of real time, the arm posed by the smoothed correction, the frames still in the body moved again from there, and the whole smoothed
    # again: four rounds, as an animator eases into a contact; then one last unsmoothed pass, so nothing stays inside
    ARM = [s + b for s in ('Left', 'Right') for b in ('Arm', 'ForeArm')]
    R0 = {b: Rot.from_quat(np.array(world[b])) for b in ARM}; D = {b: np.zeros((n, 3)) for b in ARM}   # from the turns with the hands placed
    from scipy.signal import butter, filtfilt
    lp = (lambda v: filtfilt(*butter(2, min(0.99, 4.0 / (M['fps'] / 2))), v, axis=0)) if n > 15 else (lambda v: v)
    for rnd in range(5):
        for b in ARM: world[b] = [list(q) for q in (Rot.from_rotvec(D[b]) * R0[b]).as_quat()]
        for f in range(n):
            for s in ('Left', 'Right'): frame_pass(f, s, rnd == 0, 5 if rnd < 4 else 20)   # the last pass, unsmoothed, until clear
        if rnd == 4: break
        for b in ARM: D[b] = lp((Rot.from_quat(np.array(world[b])) * R0[b].inv()).as_rotvec())
    left = sum(1 for f in range(n) for s in ('Left', 'Right') if push_out(C, Bd, C.pose(Mf, f), posed_pts(C.pose(Mf, f), s), 0.0)[1])
    Mf['world'] = {b: np.round(np.array(v), 5).tolist() for b, v in world.items()}
    Mf['fit'] = {'character': os.path.basename(A.character), 'margin_m': A.margin, 'frames_moved': pushed, 'by_body': hits, 'largest_move_cm': round(100 * worst, 1),
                 'bodies': {b: {'vertices': d['n'], 'extent_m': d['extent'].tolist()} for b, d in Bd.items()}}
    name = 'motion_' + os.path.splitext(os.path.basename(A.character))[0] + '.json'
    json.dump(Mf, open(os.path.join(A.work, 'solve', name), 'w'), separators=(',', ':'))
    print(json.dumps({'wrote': name, 'frames': n, 'moved': pushed, 'by_body': hits, 'largest_move_cm': round(100 * worst, 1), 'still_inside_frames': left}))


if __name__ == '__main__':
    main()
