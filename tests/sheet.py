"""The video above, the viewer below (one row per view dir), at the same moments: tests/sheet.py <video> <clean.json> <solve from> <out.jpg> <view dir> [<view dir> ...]"""
import sys, json, os
import cv2, numpy as np
video, clean, start, out, vdirs = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4], sys.argv[5:]; vdir = vdirs[0]
C = json.load(open(clean)); V = json.load(open(os.path.join(vdir, 'view.json'))); per = C['fps'] * C['slowmo']
cap = cv2.VideoCapture(video); top, rows = [], [[] for _ in vdirs]
for i, t in enumerate(V['times']):
    f = int(round(start + t * per)); cap.set(cv2.CAP_PROP_POS_FRAMES, f); ok, im = cap.read()
    h, w = im.shape[:2]; side = min(h, w); x0 = (w - side) // 2
    a = cv2.resize(im[:, x0:x0 + side], (300, 300)); cv2.putText(a, f'video {f}', (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
    top.append(a)
    for r, vd in enumerate(vdirs):
        v = cv2.imread(os.path.join(vd, f'view_{i}.png')); vh, vw = v.shape[:2]; s = min(vh, vw); v = cv2.resize(v[(vh - s) // 2:(vh + s) // 2, (vw - s) // 2:(vw + s) // 2], (300, 300))
        cv2.putText(v, f'{t:.2f} s', (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (40, 40, 200), 1); rows[r].append(v)
cv2.imwrite(out, np.vstack([np.hstack(top)] + [np.hstack(r) for r in rows]), [cv2.IMWRITE_JPEG_QUALITY, 82]); print('sheet', out)
