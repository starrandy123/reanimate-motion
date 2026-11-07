---
name: reanimate-motion
description: Copy a body's motion from a video onto a character (Mixamo, Blender, three.js, a game): every frame tracked, cleaned, solved to bone turns, checked by rules that can fail. Use when someone wants motion capture from a video, a trick or move from footage on their character, or BVH/animation from a clip.
---

# reanimate-motion

1. **The video.** One person, a camera that holds still. Use a video you may use (yours, or one whose licence allows it); footage
   studied only for its mechanics is never redistributed or shown.
2. `./rmo doctor`; `./rmo setup` if anything is missing.
3. `./rmo track <video> <work>` — then LOOK at `<work>/track_overlay.mp4` (or a sheet of it): the skeleton on the body, the board's
   box, the shoe circles. Read its last line: the share of frames that hold the legs and feet.
4. `./rmo clean <work>/track.json <work>/clean.json` — read the report: `slowmo` and `clock_fit` (is the measured factor near the one
   taken?), `px_per_m` and `scale_from`, `bone_length_spread`. Pass `--slowmo F` when the clip's speed is known.
5. `./rmo solve <work>/clean.json <work>/solve --from F --to F` with the frames of the move; read `fk_error_cm` and `unseen_share`.
6. `./rmo check <work>` — every rule must HOLD, and every plant must go red. A red rule is a defect to fix at its source, never a
   bound to move.
7. Show it: `./rmo view <work>/solve <out>` and `tests/sheet.py` (the video above, the character below, front and side).
8. Put it on the character: `./rmo blender …` or `web/retarget.js`; for Blender, `./rmo check <work> --baked <out.glb>` proves the
   two hosts agree.
