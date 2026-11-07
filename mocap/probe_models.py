"""Which tracker holds the whole body in the hard frames: rtmlib's BodyWithFeet (RTMPose-x, Halpe26, 2D) and Wholebody3d (RTMW3D-x,
133 keypoints with depth), each frame on its own (a person detector on every frame), drawn side by side.
   .venv/bin/python -I mocap/probe_models.py <video> <first> <last> <n> <out.jpg>"""
import sys, json
import cv2, numpy as np
from rtmlib import BodyWithFeet, Wholebody3d, draw_skeleton
video, a, b, n, out = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), sys.argv[5]
m2 = BodyWithFeet(mode='performance', backend='onnxruntime', device='cpu')
m3 = Wholebody3d(mode='balanced', backend='onnxruntime', device='cpu')
cap = cv2.VideoCapture(video); rows = [[], []]; held = {'2d': [], '3d': []}
for i in np.linspace(a, b, n).astype(int):
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(i)); ok, im = cap.read()
    if not ok: continue
    k2, s2 = m2(im)
    k3, s3, _, k3_2d = m3(im)
    for r, (k, s, tag) in enumerate(((k2, s2, '2d'), (k3_2d, s3, '3d'))):
        pic = im.copy()
        if len(k):
            j = int(np.argmax(s.mean(1))); pic = draw_skeleton(pic, k[j:j + 1], s[j:j + 1], kpt_thr=0.3)
            legs = [11, 12, 13, 14, 15, 16] + ([20, 21, 24, 25] if tag == '2d' else [17, 18, 19, 20, 21, 22])
            held[tag].append(float(np.mean(s[j][legs] > 0.3)))
        else: held[tag].append(0.0)
        cv2.putText(pic, f'{tag} {i}', (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3)
        rows[r].append(cv2.resize(pic, (320, 180)))
cv2.imwrite(out, np.vstack([np.hstack(rows[0]), np.hstack(rows[1])]), [cv2.IMWRITE_JPEG_QUALITY, 80])
print(json.dumps({k: [round(v, 2) for v in vs] for k, vs in held.items()}))
if len(k3): print('3d sample (frame last, keypoints 11-16 legs):', np.round(k3[0][11:17], 3).tolist())
