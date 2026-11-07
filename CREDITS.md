# Credits and licences

The kit's own code is MIT (LICENSE). What it runs or learned from:

| Piece | What it does here | Licence | Source |
|---|---|---|---|
| rtmlib 0.0.16 | runs the trackers (ONNX) | Apache-2.0 | github.com/Tau-J/rtmlib |
| RTMW3D-x (rtmw3d-x_8xb64_cocktail14-384x288) | 133 keypoints with depth, every frame | code and weights Apache-2.0 (OpenMMLab); **its 3D training data includes H3WB, built on Human3.6M, whose EULA is academic use only: see below** | open-mmlab/mmpose · ONNX via huggingface.co/Soykaf/RTMW3D-x |
| YOLOX-m Human-Art | finds the person (no face needed) | Apache-2.0 | open-mmlab/mmpose (rtmposev1 onnx_sdk) |
| YOLOX-s COCO | finds the skateboard (class 36) | Apache-2.0 | github.com/Megvii-BaseDetection/YOLOX |
| OpenCV CSRT (opencv-contrib) | follows each shoe as a picture | Apache-2.0 | opencv.org |
| Sports2D | the cleaning order and its defaults (Hampel 7, gaps, Butterworth 4th order) — method, no code copied | BSD-3 | github.com/davidpagnon/Sports2D |
| C. J. Taylor, CVPR 2000 | depth from known bone lengths under a scaled-orthographic camera | paper | "Reconstruction of articulated objects from point correspondences in a single uncalibrated image" |
| D. A. Winter, *Biomechanics and Motor Control of Human Movement* (2009), table 4.1 (Dempster) | segment masses and centres for the gravity clock | book | |
| NVIDIA GEM-X | the reference pipeline this kit learns from (contacts and IK, the camera from the background) — not bundled | Apache-2.0 code; weights NVIDIA Open Model Licence (commercial use) | github.com/NVlabs/GEM-X |
| three.js r183 | the web example and retarget.js | MIT | threejs.org |
| Blender | the Blender example (bpy) | GPL (the tool); the script is MIT | blender.org |
| Polyfork "Street Skater" | the example character (examples/models/street-skater.glb) | CC BY 4.0 — credit Polyfork, polyfork.dev | polyfork.dev |
| scipy · numpy · onnxruntime | maths · runtime | BSD-3 · BSD-3 · MIT | |

**Before you ship motion made with it in a commercial product:** RTMW3D's depth was learned partly from Human3.6M, whose licence is
academic use only, and most pose models (RTMPose's Body7 too) were trained on academic datasets. A lawyer's reading decides what that
means for motion you derive. Cleaner for commercial work: track with `--model halpe26`
and clean with `--2d` (no Human3.6M in its models, depth by Taylor and anatomy alone; its 2D model still learned partly from academic
sets, like most). The one route clean on paper is NVIDIA's GEM-X (trained on NVIDIA's own motion capture, its weights licensed for
commercial use).

Reference videos: copy the motion of a video you may use (your own, or one whose licence allows it). Footage studied only for its
mechanics is never redistributed and never shown.
