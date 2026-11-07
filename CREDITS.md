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
| OpenCV solvePnP (SQPnP) | the head's turn from the face points: a generic 3D face fitted to the picture | Apache-2.0 | opencv.org; G. Terzakis & M. Lourakis, "A Consistently Fast and Globally Optimal Solution to the Perspective-n-Point Problem", ECCV 2020 |
| LearnOpenCV, "Head Pose Estimation using OpenCV and Dlib" | the generic 6-point 3D face (nose tip, chin, eye and mouth corners) and the focal length taken as the picture's width — numbers, no code copied | article | learnopencv.com |
| Unreal Engine IK Retargeter (Epic) | the idea of `rmo fit`: hand goals moved out of the target's body colliders (its "Body Intersect Goals" op) and placed toward the source's hands ("Blend to Source"); bodies fitted to a bone's dominant-weight vertices as Unreal's physics-asset tool does — from Epic's documentation, no code | docs | dev.epicgames.com (IK Retargeter, retargeting operation stack) |
| KevinLTT/video2bvh | keypoints to BVH: each bone aims at its child joint — the method, no code copied | MIT | github.com/KevinLTT/video2bvh |
| Physio-Pedia, "Sternoclavicular Joint"; GPnotebook | the clavicle's range (up 45°, down 10°, forward and back about 15°) | articles | physio-pedia.com |
| White & Panjabi, *Clinical Biomechanics of the Spine* (1990); AAOS, *Joint Motion* (1965); Youdas et al., *Phys Ther* 72(11) 1992 | the spine's twist and the neck's range | books / paper | |
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
