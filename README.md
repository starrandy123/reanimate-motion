# reanimate-motion

Copy a body's **motion** from a video onto any character: every frame of the video mapped (the whole body, the feet, a board), cleaned,
turned into animation, and played on a Mixamo, Blender or three.js character. The sequel to
[reanimate-video](https://github.com/starrandy123/reanimate-video), which copies what a video *looks like*; this one copies how a
body *moves*.

```
video ──► rmo track ──► rmo clean ──► rmo solve ──► motion.bvh   ──► Blender (File > Import), three.js BVHLoader
           every frame:   blend, depth,   each bone's    motion.json ──► web/retarget.js on any T-posed three.js character
           body, feet,    clock, smooth   turn from its             ──► examples/blender/apply_motion.py on any Blender rig
           board, shoes                   T-pose                    ──► your game (feet and hips as IK targets)
```

## Measured on a real clip

A slow-motion kickflip, 981 frames at 60 fps, head out of the picture for most of the jump:

| | |
|---|---|
| legs and feet held | 98 % of frames (the shoe tracker covers the blurred ones) |
| the board found | 958 of 981 frames |
| the clip's clock, from gravity | 10.33x slow measured, 10x taken (a 240 fps recording slowed to 24) — fit error 0.56 cm |
| bones held rigid | every limb bone within 0.0 % of its length |
| the written skeleton against what was tracked | knees within 1.2 cm, ankles within 2.5 cm |
| Blender and three.js, the same motion on the same character | 0.00 cm apart over 22 bones, every frame |

`rmo check <work>` proves each of these and proves each check can fail (below).

## Install

```sh
git clone https://github.com/starrandy123/reanimate-motion && cd reanimate-motion
./rmo setup      # once: the kit's own Python (3.12) with rtmlib, onnxruntime, OpenCV, scipy
./rmo doctor     # what is installed
```
The models download on first use (about 650 MB, from OpenMMLab, Hugging Face and the YOLOX release). Node is needed for the viewer and
the checks (`npm i -g playwright` or a local install), Blender only for `rmo blender`.

## The commands

```
rmo all <video> <work> [--from F --to F]     track, clean and solve in one go
rmo track <video> <work> [--model halpe26]   every frame: 133 body points with depth (or 26, 2D only), the board, each shoe
rmo clean <track.json> <clean.json> [--2d]   left/right fixed, shoes blended, outliers, gaps, the clock, smoothing, depth
rmo solve <clean.json> <out> [--fps 60]      motion.bvh (Mixamo bone names) + motion.json (each bone's world turn)
rmo view <solve dir> <out dir>               the three.js example, stilled at chosen times (examples/threejs/index.html)
rmo film <work> [--video V] [--character G]  the character beside the video, frame for frame: film/side.mp4, character.mp4
rmo blender <motion.json> <rig> <out>        a Blender character (.glb .fbx .blend) keyed with the motion
rmo check <work> [--baked <glb>]             every rule, each proven red by a planted defect
```

## Try it on free clips

Any video you may use works. These free clips from Mixkit (the Mixkit Stock Video Free License: commercial use, edits and social
posts allowed, no credit required) were run through the kit on 7 Nov 2026; download them from their pages (they are not bundled:
Mixkit's terms forbid handing the files on):

| Clip | What it moves |
|---|---|
| [Boy doing a pirouette with a skateboard in a park](https://mixkit.co/free-stock-video/boy-doing-a-pirouette-with-a-skateboard-in-a-park-34382/) | a board trick, filmed slow (the clock reads 6.3x) |
| [A young man performs a karate kick to the air](https://mixkit.co/free-stock-video/a-young-man-performs-a-karate-kick-to-the-air-49634/) | a high kick on black |

```sh
./rmo all 34382-1080.mp4 work/skate && ./rmo film work/skate --video 34382-1080.mp4 --hide attach-righthand-mesh
```

## How each step works, and where it comes from

| Step | How | Source |
|---|---|---|
| the body | RTMW3D-x: 133 COCO-WholeBody points (feet, hands, face) with relative depth, behind a person detector that needs no face (MediaPipe loses a rider whose head leaves the frame) | rtmlib (Apache-2.0), OpenMMLab |
| the board | YOLOX on COCO's "skateboard" class, the box nearest the feet | Megvii YOLOX (Apache-2.0) |
| the shoes | each followed as a picture (OpenCV CSRT), seeded from the feet points, blended in where the model is unsure | OpenCV |
| the cleaning | a point under 0.3 is a gap; left/right swaps undone; Hampel over 7 frames; gaps up to 100 filled; Butterworth 4th order at 12 Hz of real time | Sports2D's order and defaults (BSD-3) |
| unseen points | a point the model puts outside the picture is unseen, never trusted; across a gap a bone turns evenly from the last way it was seen to the next | the infilling idea of NVIDIA GEM-X and GLAMR |
| the clock | in the air the body's centre falls at 9.81 m/s²; with the upper body carried by the hips when it leaves the picture | Dempster's segments (Winter 2009, table 4.1) |
| depth | each bone's depth from its own length under a scaled-orthographic camera; which end is nearer by anatomy (hips flexed, knees bend one way, toes ahead), else the model, a sign never flipping on noise | C. J. Taylor, CVPR 2000 |
| the scale | the board's length (a deck is about 0.81 m), else an adult leg | |
| the turns | each bone aims at its child joint; its twist from a bend plane, the hip line or the shoulders; limb twist low-passed on its own | the standard swing-twist solve |
| retarget | a bone's world rotation = the motion's world turn × the bone's own world rotation at rest: any T-posed rig, whatever its bones' axes | |

## Using the output

- **Blender:** `File > Import > Motion Capture (.bvh)` opens `motion.bvh` as the rider's own skeleton. To move *your* character:
  `rmo blender motion.json my_character.fbx out.glb` (a Mixamo download, a Blender rig with Mixamo, Unreal or VRM bone names).
- **three.js:** `bindMotion(characterRoot, motion)` from `web/retarget.js`, then `m.apply(seconds)` each frame (`examples/threejs`).
- **A game:** the feet, hips and knees in `clean.json` are metres in the camera's frame; a game that keeps feet on a board or the ground
  takes them as IK targets.

## The rules (`rmo check`)

R1 legs and feet held in ≥ 90 % of frames · R2 every limb bone within 5 % of its length · R3 knees in front where a thigh points out,
none bent backward · R4 a fall filmed 10x slow reads 10x · R5 the written skeleton puts knees and ankles back within 3 cm · R6 no bone
faster than 1,500°/s · R7 Blender and three.js agree within 2 cm · R8 the rider's leg is a person's: a shoe the detector calls a board never sets the scale. Each has a planted defect that must turn it red.

## What it does not do yet

One person; a camera that holds still (a moving camera needs its own motion taken from the background: next), and the person at
about one distance from it (the scale is one for the whole clip: someone who runs at the camera reads wrong where they are far; a
scale for each frame is next). Parts out
of the picture are bridged, not seen. Depth from one camera is reconstructed, not measured. Hands and fingers turn only with
`--hands`, for clips where they are large and sharp. Licences, and what they mean for commercial use: [CREDITS.md](CREDITS.md).

## Why not NVIDIA's GEM-X?

[GEM-X](https://github.com/NVlabs/GEM-X) turns a video into a 3D body too, and it is where this kit learned to bridge what the camera
loses, to lock feet and to take a moving camera from the background. It is not bundled:

- it reads the same video: another reader, not more footage;
- what leaves the picture is a guess either way (GEM-X fills it from what it learned, this kit bridges it evenly): a wider shot is
  what measures it;
- for a moving camera it runs ViPE, which is built on CUDA, so it needs an NVIDIA card. This kit runs on a laptop's CPU and is headed
  for the browser;
- its weights are about 7 GB.

**InstantHMR** ([Naturalpad-studio/InstantHMR](https://github.com/Naturalpad-studio/InstantHMR), tried 7 Nov 2026) recovers a whole
3D body per frame and runs on a Mac through CoreML (about 110 ms a frame here). Its `medium-commercial` model (Apache-2.0) drew both legs
standing at the top of the free karate clip's high kick (the right foot 737 px from where it was); its `medium` model followed the kick
(38 px) but is CC BY-NC 4.0, so it cannot carry a commercial game. Not wired in until a commercial model holds the rules.

Where GEM-X is ahead is the licence: its weights are licensed for commercial use and it learned from NVIDIA's own motion capture
([CREDITS.md](CREDITS.md)). If your project needs that on paper, GEM-X is the route.
