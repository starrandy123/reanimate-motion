/* PUT A MOTION ON ANY T-POSED CHARACTER (three.js, MIT like the rest of the kit).
 * motion.json (rmo solve) holds each bone's turn in the WORLD from its T-pose. On a character whose rest pose is a T-pose, that turn
 * is the same whatever axes its own bones were built with: a bone's world rotation = the turn × the bone's own world rotation at rest,
 * and its local rotation = its parent's world rotation⁻¹ × that. The same rule moves a Mixamo download, a Polyfork or Blender rig
 * and the three.js examples' characters. The hips' place is scaled by the character's leg against the rider's.
 *   import { bindMotion } from './retarget.js';
 *   const m = bindMotion(characterRoot, await (await fetch('motion.json')).json());
 *   renderer.setAnimationLoop((ms) => { m.apply(ms / 1000 % m.duration); renderer.render(scene, camera); });
 * Bone names: Mixamo's (with or without "mixamorig:", "mixamorig1_" ...), Unreal's mannequin and VRM's humanoid names are read; pass
 * { names: { Hips: 'pelvis_bone', ... } } for any other rig. */
import * as THREE from 'three';

const ALIASES = {
  Hips: ['pelvis', 'hips', 'j_bip_c_hips'], Spine: ['spine_01', 'spine', 'j_bip_c_spine'], Spine1: ['spine_02', 'chest', 'j_bip_c_chest'],
  Spine2: ['spine_03', 'upperchest', 'j_bip_c_upperchest'], Neck: ['neck_01', 'neck', 'j_bip_c_neck'], Head: ['head', 'j_bip_c_head'],
  LeftShoulder: ['clavicle_l', 'leftshoulder', 'j_bip_l_shoulder'], LeftArm: ['upperarm_l', 'leftupperarm', 'j_bip_l_upperarm'],
  LeftForeArm: ['lowerarm_l', 'leftlowerarm', 'j_bip_l_lowerarm'], LeftHand: ['hand_l', 'lefthand', 'j_bip_l_hand'],
  LeftUpLeg: ['thigh_l', 'leftupperleg', 'j_bip_l_upperleg'], LeftLeg: ['calf_l', 'leftlowerleg', 'j_bip_l_lowerleg'],
  LeftFoot: ['foot_l', 'leftfoot', 'j_bip_l_foot'], LeftToeBase: ['ball_l', 'lefttoes', 'j_bip_l_toebase'],
};
for (const k of Object.keys(ALIASES)) if (k.startsWith('Left')) ALIASES['Right' + k.slice(4)] = ALIASES[k].map((a) => a.replace(/_l$/, '_r').replace(/^left/, 'right').replace('_l_', '_r_'));
const plain = (n) => n.toLowerCase().replace(/^mixamorig\d*[:_]?/, '').replace(/[^a-z0-9_]/g, '');

export function findBones(root, names = {}) {
  const all = new Map(); root.traverse((o) => { if (o.isBone || o.type === 'Bone') all.set(plain(o.name), o); });
  const out = {};
  for (const want of Object.keys(ALIASES).concat(Object.keys(names))) {
    const tries = [names[want], want, ...(ALIASES[want] || [])].filter(Boolean).map(plain);
    for (const t of tries) if (all.has(t)) { out[want] = all.get(t); break; }
  }
  root.traverse((o) => {                                   // fingers and anything else spelled the Mixamo way
    if (!(o.isBone || o.type === 'Bone')) return;
    const p = plain(o.name); for (const want of [o.name.replace(/^mixamorig\d*[:_]?/, '')]) if (!out[want] && /^(left|right)hand(thumb|index|middle|ring|pinky)\d$/.test(p)) out[want] = o;
  });
  return out;
}

export function bindMotion(root, motion, opts = {}) {
  const bones = findBones(root, opts.names), fps = motion.fps, n = motion.frames;
  root.updateMatrixWorld(true);
  const rootQ = root.getWorldQuaternion(new THREE.Quaternion()), rootQi = rootQ.clone().invert();
  const rest = new Map();                                  // each bone's world rotation at rest, in the character root's own frame
  for (const [name, b] of Object.entries(bones)) rest.set(name, rootQi.clone().multiply(b.getWorldQuaternion(new THREE.Quaternion())));
  const hips = bones.Hips, hipsRest = hips ? hips.position.clone() : null;
  const rootPos = root.getWorldPosition(new THREE.Vector3()), hipsRestWorld = hips ? hips.getWorldPosition(new THREE.Vector3()) : rootPos.clone();
  const legOf = () => {                                    // the character's hip height from its feet, in its root's frame
    const a = new THREE.Vector3(), b = new THREE.Vector3();
    if (!bones.LeftUpLeg || !bones.LeftFoot) return null;
    bones.LeftUpLeg.getWorldPosition(a); bones.LeftFoot.getWorldPosition(b); return a.distanceTo(b);
  };
  const L = motion.lengths, k = (legOf() || 1) / ((L.thigh + L.shin) || 1) / (root.getWorldScale(new THREE.Vector3()).y || 1);
  const order = motion.bones.filter((b) => bones[b]);
  const q = (b, f) => { const v = motion.world[b][f]; return new THREE.Quaternion(v[0], v[1], v[2], v[3]); };
  const world = new Map(), tmp = new THREE.Quaternion(), pq = new THREE.Quaternion();
  return {
    duration: n / fps, bones, scale: k,
    apply(time) {
      const u = Math.max(0, Math.min(n - 1, time * fps)), f0 = Math.floor(u), f1 = Math.min(n - 1, f0 + 1), w = u - f0;
      world.clear();
      for (const name of order) {
        const b = bones[name];
        tmp.copy(q(name, f0)).slerp(q(name, f1), w).multiply(rest.get(name));           // world (root frame) = turn × rest
        world.set(b, tmp.clone());
        const parent = b.parent; let pw = world.get(parent);
        if (!pw) { parent.updateWorldMatrix(true, false); pw = rootQi.clone().multiply(parent.getWorldQuaternion(pq)); }
        b.quaternion.copy(pw.clone().invert().multiply(tmp));
        b.updateMatrixWorld(false);
      }
      if (hips && motion.hips) {                           // the hips' place: the rider's, scaled to this character's legs, in the hips' parent's frame
        /* the character starts where it stands and moves as the rider moved; its hips' height is the rider's, scaled to its legs */
        const a = motion.hips[f0], c = motion.hips[f1], h0 = motion.hips[0];
        const p = new THREE.Vector3((a[0] + (c[0] - a[0]) * w - h0[0]) * k, (a[1] + (c[1] - a[1]) * w) * k, (a[2] + (c[2] - a[2]) * w - h0[2]) * k);
        if (opts.inPlace) { p.x = 0; p.z = 0; }
        p.applyQuaternion(rootQ).add(hipsRestWorld);
        p.y = rootPos.y + (a[1] + (c[1] - a[1]) * w) * k;
        hips.parent.updateWorldMatrix(true, false); hips.parent.worldToLocal(p);
        hips.position.copy(opts.keepHeightOnly ? hipsRest.clone().setY(p.y) : p);
      }
    },
  };
}
