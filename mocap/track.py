"""rmo track <video> <out dir> [--from N] [--to M]

EVERY FRAME OF THE VIDEO, MAPPED:
  the whole body   RTMW3D-x through rtmlib (Apache-2.0): 133 COCO-WholeBody keypoints (body 0-16, feet 17-22, face 23-90,
                   hands 91-132), each with its place in the picture and its depth (metres, relative), behind a person detector
                   (YOLOX-m, Human-Art) that needs no face, run every 4th frame as Sports2D does; between, the box follows the body
  the board        YOLOX-s on COCO's 80 classes (Megvii, Apache-2.0), class 36 'skateboard', the one nearest the feet
  each shoe        OpenCV's CSRT tracker, seeded from the feet keypoints and seeded again when they are sure and it has drifted
Writes <out>/track.json and <out>/track_overlay.mp4 (the mapping drawn, to judge by eye)."""
import sys, os, json, argparse
import cv2, numpy as np
from rtmlib import YOLOX, RTMPose3d, draw_skeleton

ap = argparse.ArgumentParser(); ap.add_argument('video'); ap.add_argument('out'); ap.add_argument('--from', dest='a', type=int, default=0)
ap.add_argument('--to', dest='b', type=int, default=10 ** 9); ap.add_argument('--det-every', type=int, default=4); ap.add_argument('--device', default='cpu', help="cpu, or mps (Apple's GPU through CoreML)")
ap.add_argument('--model', default='rtmw3d', choices=['rtmw3d', 'halpe26'], help='rtmw3d: 133 points with depth; halpe26: RTMPose body+feet, 2D only, never trained on Human3.6M (rmo clean --2d)')
A = ap.parse_args()
os.makedirs(A.out, exist_ok=True)
DEV = dict(backend='onnxruntime', device=A.device)
det = YOLOX('https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/yolox_m_8xb8-300e_humanart-c2c7a14a.zip', model_input_size=(640, 640), **DEV)
from rtmlib import RTMPose
if A.model == 'rtmw3d': pose = RTMPose3d('https://huggingface.co/Soykaf/RTMW3D-x/resolve/main/onnx/rtmw3d-x_8xb64_cocktail14-384x288-b0a0eab7_20240626.onnx', model_input_size=(288, 384), **DEV)
else: pose = RTMPose('https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/rtmpose-x_simcc-body7_pt-body7-halpe26_700e-384x288-7fb6e239_20230606.zip', model_input_size=(288, 384), **DEV)
HALPE = {**{i: i for i in range(17)}, 20: 17, 22: 18, 24: 19, 21: 20, 23: 21, 25: 22}   # Halpe26 -> COCO-WholeBody: body, then big toe, small toe, heel
objs = YOLOX('https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_s.onnx', model_input_size=(640, 640), det_mode='multiclass', score_thr=0.3, **DEV)
SKATEBOARD = 36
FEET = {'L': (15, 17, 18, 19), 'R': (16, 20, 21, 22)}            # ankle, big toe, small toe, heel

def iou(a, b):
    x1, y1, x2, y2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    i = max(0, x2 - x1) * max(0, y2 - y1); u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i
    return i / u if u > 0 else 0

def box_of(k, s, W, H, grow=1.25):
    ok = s > 0.3
    if ok.sum() < 6: return None
    p = k[ok]; c = (p.min(0) + p.max(0)) / 2; h = (p.max(0) - p.min(0)) * grow / 2
    return [max(0, c[0] - h[0]), max(0, c[1] - h[1]), min(W, c[0] + h[0]), min(H, c[1] + h[1])]

def foot_box(k, s, side, size):
    ids = list(FEET[side]); w = s[ids]
    if w.mean() < 0.5: return None
    c = (k[ids] * w[:, None]).sum(0) / w.sum(); r = size / 2
    return (int(c[0] - r), int(c[1] - r), int(2 * r), int(2 * r))

cap = cv2.VideoCapture(A.video); W, H = int(cap.get(3)), int(cap.get(4)); fps = cap.get(cv2.CAP_PROP_FPS); N = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
a, b = max(0, A.a), min(N - 1, A.b); cap.set(cv2.CAP_PROP_POS_FRAMES, a)
vw = cv2.VideoWriter(os.path.join(A.out, 'track_overlay.mp4'), cv2.VideoWriter_fourcc(*'mp4v'), fps, (W, H))
frames, prev, shoes = [], None, {}
for i in range(a, b + 1):
    ok, im = cap.read()
    if not ok: break
    row = {'i': i}
    if prev is None or (i - a) % A.det_every == 0:
        boxes = det(im)
        if len(boxes):
            boxes = [list(map(float, bb)) for bb in boxes]
            prev = max(boxes, key=(lambda bb: iou(bb, prev)) if prev else (lambda bb: (bb[2] - bb[0]) * (bb[3] - bb[1])))
    if prev is not None:
        if A.model == 'rtmw3d':
            k3, sc, _, k2 = pose(im, bboxes=[prev]); k3, sc, k2 = k3[0], sc[0], k2[0]; zz = np.round(k3[:, 2], 3).tolist()
        else:                                              # 2D only: the 26 points in the 133 layout, the rest unseen
            kk, ss = pose(im, bboxes=[prev]); k2 = np.zeros((133, 2)); sc = np.zeros(133); zz = None
            for h_, w_ in HALPE.items(): k2[w_] = kk[0][h_]; sc[w_] = ss[0][h_]
        row.update(k=np.round(k2, 1).tolist(), s=np.round(sc, 2).tolist(), box=[round(v, 1) for v in prev])
        if zz is not None: row['z'] = zz
        nb = box_of(k2, sc, W, H); prev = nb if nb else prev
        # the board: the skateboard box nearest the feet
        bx, cl = objs(im)
        feet = k2[[15, 16, 17, 20]].mean(0)
        cands = [list(map(float, bx[j])) for j in range(len(cl)) if int(cl[j]) == SKATEBOARD]
        if cands:
            bb = min(cands, key=lambda q: np.hypot((q[0] + q[2]) / 2 - feet[0], (q[1] + q[3]) / 2 - feet[1]))
            row['board'] = [round(v, 1) for v in bb]
        # the shoes, each followed as a picture; seeded again when the body model is sure of the foot and the tracker has wandered
        leg = np.linalg.norm(k2[11] - k2[13]) + np.linalg.norm(k2[13] - k2[15]); size = max(24, 0.32 * leg)
        for side in ('L', 'R'):
            fb = foot_box(k2, sc, side, size); t = shoes.get(side)
            if t is not None:
                okt, tb = t.update(im)
                if okt: row['shoe' + side] = [round(tb[0] + tb[2] / 2, 1), round(tb[1] + tb[3] / 2, 1)]
                if fb and (not okt or np.hypot(tb[0] + tb[2] / 2 - fb[0] - fb[2] / 2, tb[1] + tb[3] / 2 - fb[1] - fb[3] / 2) > 0.6 * size) and sc[list(FEET[side])].mean() > 0.6:
                    t = None
            if t is None and fb:
                t = cv2.TrackerCSRT_create(); t.init(im, fb); shoes[side] = t; row['shoe' + side] = [fb[0] + fb[2] / 2, fb[1] + fb[3] / 2]; row['seed' + side] = 1
        pic = draw_skeleton(im.copy(), k2[None], sc[None], kpt_thr=0.3)
        if 'board' in row: q = list(map(int, row['board'])); cv2.rectangle(pic, (q[0], q[1]), (q[2], q[3]), (0, 140, 255), 2)
        for side, col in (('L', (0, 255, 255)), ('R', (255, 0, 255))):
            if 'shoe' + side in row: c = row['shoe' + side]; cv2.circle(pic, (int(c[0]), int(c[1])), int(size / 2), col, 2)
    else: pic = im.copy()
    cv2.putText(pic, str(i), (10, 34), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2); vw.write(pic); frames.append(row)
    if (i - a) % 100 == 0: print(f'{i}/{b}', flush=True)
vw.release()
held = sum(1 for r in frames if 'k' in r and np.mean(np.array(r['s'])[11:23] > 0.3) > 0.9)
json.dump({'video': os.path.basename(A.video), 'fps': fps, 'size': [W, H], 'from': a, 'frames': frames,
           'models': {'body': 'RTMW3D-x (rtmw3d-x_8xb64_cocktail14-384x288-b0a0eab7_20240626)', 'person': 'YOLOX-m Human-Art', 'board': 'YOLOX-s COCO (Megvii 0.1.1rc0)', 'shoes': 'OpenCV CSRT'}},
          open(os.path.join(A.out, 'track.json'), 'w'), separators=(',', ':'))
print(f'{os.path.basename(A.video)}: {len(frames)} frames, legs and feet held in {held} ({100 * held / max(1, len(frames)):.0f}%), board in {sum(1 for r in frames if "board" in r)}')
