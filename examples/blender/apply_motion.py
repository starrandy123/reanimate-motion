"""PUT A MOTION ON A BLENDER CHARACTER, headless (Blender 3.6+ / 4.x; MIT like the kit).

  blender --background --python examples/blender/apply_motion.py -- <motion.json> <character .glb/.gltf/.fbx/.blend> <out .glb/.fbx/.blend> [--in-place]

Any rig in a T-pose works, whatever its bones' own axes (a Mixamo download, a Polyfork model, your own Blender rig with Mixamo, Unreal
or VRM bone names): each bone's world rotation = the motion's world turn x the bone's own world rotation at rest (the rule
web/retarget.js uses in three.js). The motion is Y-up and faces +Z (glTF's frame); Blender is Z-up and faces -Y, so every turn is
carried across: C · turn · C^-1. The hips' place is scaled by the character's legs against the rider's. Keys every frame, then exports.
(Only want the rider's own skeleton? File > Import > Motion Capture (.bvh) opens motion.bvh as it is.)"""
import sys, json, re
import bpy
from mathutils import Matrix, Quaternion, Vector

argv = sys.argv[sys.argv.index('--') + 1:] if '--' in sys.argv else []
if len(argv) < 3: raise SystemExit(__doc__)
motion_path, char_path, out_path = argv[:3]; in_place = '--in-place' in argv
M = json.load(open(motion_path))

bpy.ops.wm.read_factory_settings(use_empty=True)
ext = char_path.lower().rsplit('.', 1)[-1]
if ext in ('glb', 'gltf'): bpy.ops.import_scene.gltf(filepath=char_path)
elif ext == 'fbx': bpy.ops.import_scene.fbx(filepath=char_path)
elif ext == 'blend':
    with bpy.data.libraries.load(char_path) as (src, dst): dst.objects = src.objects
    for o in dst.objects: bpy.context.scene.collection.objects.link(o)
arm = next(o for o in bpy.context.scene.objects if o.type == 'ARMATURE')

ALIASES = {'Hips': ['pelvis', 'hips', 'j_bip_c_hips'], 'Spine': ['spine_01', 'spine'], 'Spine1': ['spine_02', 'chest'], 'Spine2': ['spine_03', 'upperchest'],
           'Neck': ['neck_01', 'neck'], 'Head': ['head'], 'LeftShoulder': ['clavicle_l', 'leftshoulder'], 'LeftArm': ['upperarm_l', 'leftupperarm'],
           'LeftForeArm': ['lowerarm_l', 'leftlowerarm'], 'LeftHand': ['hand_l', 'lefthand'], 'LeftUpLeg': ['thigh_l', 'leftupperleg'],
           'LeftLeg': ['calf_l', 'leftlowerleg'], 'LeftFoot': ['foot_l', 'leftfoot'], 'LeftToeBase': ['ball_l', 'lefttoes']}
for k in [k for k in ALIASES if k.startswith('Left')]: ALIASES['Right' + k[4:]] = [a.replace('_l', '_r').replace('left', 'right') for a in ALIASES[k]]
plain = lambda n: re.sub(r'[^a-z0-9_]', '', re.sub(r'^mixamorig\d*[:_]?', '', n.lower()))
have = {plain(b.name): b.name for b in arm.data.bones}
bones = {}
for want in M['bones']:
    for t in [want] + ALIASES.get(want, []):
        if plain(t) in have: bones[want] = have[plain(t)]; break

C = Matrix(((1, 0, 0), (0, 0, -1), (0, 1, 0))).to_quaternion()          # glTF (x, y, z) -> Blender (x, -z, y)
Ci = C.inverted(); AW = arm.matrix_world.copy(); AWq = AW.to_quaternion(); AWqi = AWq.inverted()
rest_world = {w: (AW @ arm.data.bones[n].matrix_local).to_quaternion() for w, n in bones.items()}
def leg_world():
    if 'LeftUpLeg' not in bones or 'LeftFoot' not in bones: return None
    return ((AW @ arm.data.bones[bones['LeftUpLeg']].head_local) - (AW @ arm.data.bones[bones['LeftFoot']].head_local)).length
k = (leg_world() or 1.0) / (M['lengths']['thigh'] + M['lengths']['shin'])
hips0 = M['hips'][0]; hips_rest_world = AW @ arm.data.bones[bones['Hips']].head_local if 'Hips' in bones else Vector()

scene = bpy.context.scene; scene.render.fps = int(round(M['fps'])); scene.frame_start = 0; scene.frame_end = M['frames'] - 1   # glTF writes a key at frame / fps: frame 0 is time 0
order = [b for b in M['bones'] if b in bones]
for pb in arm.pose.bones: pb.rotation_mode = 'QUATERNION'
def pose_rot(bone, done):
    """a bone's rotation in armature space this frame: the one just set, or (unmoved) its parent's times its rest offset from it"""
    if bone.name in done: return done[bone.name]
    if bone.parent is None: r = bone.matrix_local.to_quaternion()
    else: r = pose_rot(bone.parent, done) @ (bone.parent.matrix_local.inverted() @ bone.matrix_local).to_quaternion()
    done[bone.name] = r; return r
for f in range(M['frames']):
    arm_rot = {}
    for want in order:
        name = bones[want]; bone = arm.data.bones[name]; pb = arm.pose.bones[name]
        x, y, z, w = M['world'][want][f]; turn = C @ Quaternion((w, x, y, z)) @ Ci
        r_arm = AWqi @ (turn @ rest_world[want])                          # this bone's rotation in armature space
        arm_rot[name] = r_arm
        if bone.parent is not None:
            pr = pose_rot(bone.parent, arm_rot)              # a parent the motion does not move still turns with ITS parent
            rel = (bone.parent.matrix_local.inverted() @ bone.matrix_local).to_quaternion()
            pb.rotation_quaternion = (pr @ rel).inverted() @ r_arm
        else:
            pb.rotation_quaternion = bone.matrix_local.to_quaternion().inverted() @ r_arm
        if want == 'Hips':
            # the character starts where it stands and moves as the rider moved; its hips' height is the rider's, scaled to its legs
            h = M['hips'][f]; d = Vector((h[0] - hips0[0], 0, h[2] - hips0[2])) * k
            if in_place: d.x = 0; d.z = 0
            target = hips_rest_world + (C @ d); target.z = AW.translation.z + h[1] * k
            L = bone.matrix_local; t_arm = AW.inverted() @ target
            pb.location = L.to_quaternion().inverted() @ (t_arm - L.translation)
            pb.keyframe_insert('location', frame=f)
        pb.keyframe_insert('rotation_quaternion', frame=f)
o = out_path.lower()
if o.endswith('.glb'): bpy.ops.export_scene.gltf(filepath=out_path, export_format='GLB', export_animations=True, export_force_sampling=True)
elif o.endswith('.fbx'): bpy.ops.export_scene.fbx(filepath=out_path, bake_anim=True, add_leaf_bones=False)
else: bpy.ops.wm.save_as_mainfile(filepath=out_path)
print(f'apply_motion: {len(order)} bones of {len(M["bones"])} moved, {M["frames"]} frames at {M["fps"]} fps, legs x{k:.3f} -> {out_path}')
