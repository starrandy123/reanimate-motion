"""A T-posed, skinned glTF character read in Python, and posed the way web/retarget.js poses it (a bone's world turn = the motion's turn x
the bone's own world turn at rest; its rest translation kept). For rmo fit and the checks: where the character's own hands and body are,
frame by frame, without a browser. Reads .glb only (one binary chunk), float positions and weights, the joints as bytes or shorts."""
import json, struct, numpy as np
from scipy.spatial.transform import Rotation as Rot

CT = {5120: np.int8, 5121: np.uint8, 5122: np.int16, 5123: np.uint16, 5125: np.uint32, 5126: np.float32}
NC = {'SCALAR': 1, 'VEC2': 2, 'VEC3': 3, 'VEC4': 4, 'MAT4': 16}
ALIAS = {'hips': 'Hips', 'pelvis': 'Hips', 'spine': 'Spine', 'spine_01': 'Spine', 'spine1': 'Spine1', 'spine_02': 'Spine1', 'chest': 'Spine1',
         'spine2': 'Spine2', 'spine_03': 'Spine2', 'upperchest': 'Spine2', 'neck': 'Neck', 'neck_01': 'Neck', 'head': 'Head'}
for s, S in (('l', 'Left'), ('r', 'Right')):
    side = 'left' if s == 'l' else 'right'
    for mix, ue in (('shoulder', 'clavicle'), ('arm', 'upperarm'), ('forearm', 'lowerarm'), ('hand', 'hand'), ('upleg', 'thigh'), ('leg', 'calf'), ('foot', 'foot'), ('toebase', 'ball')):
        name = S + {'shoulder': 'Shoulder', 'arm': 'Arm', 'forearm': 'ForeArm', 'hand': 'Hand', 'upleg': 'UpLeg', 'leg': 'Leg', 'foot': 'Foot', 'toebase': 'ToeBase'}[mix]
        ALIAS[side + mix] = name; ALIAS[f'{ue}_{s}'] = name
def plain(n): import re; n = re.sub(r'^mixamorig\d*[:_]?', '', n.lower()); return re.sub(r'[^a-z0-9_]', '', n)


class Character:
    def __init__(self, path, hide=()):
        f = open(path, 'rb').read(); n0 = struct.unpack('<I', f[12:16])[0]; self.J = J = json.loads(f[20:20 + n0])
        b0 = 20 + n0; self.bin = f[b0 + 8: b0 + 8 + struct.unpack('<I', f[b0:b0 + 4])[0]]
        N = J['nodes']; self.names = [n.get('name', f'node{i}') for i, n in enumerate(N)]
        self.parent = {c: i for i, n in enumerate(N) for c in n.get('children', [])}
        self.local = []
        for n in N:
            if 'matrix' in n: m = np.array(n['matrix'], float).reshape(4, 4).T
            else:
                m = np.eye(4); m[:3, :3] = Rot.from_quat(n.get('rotation', [0, 0, 0, 1])).as_matrix() * np.array(n.get('scale', [1, 1, 1])); m[:3, 3] = n.get('translation', [0, 0, 0])
            self.local.append(m)
        self.world = [None] * len(N)
        for i in range(len(N)): self._w(i)
        # the bones a motion names, found by name as retarget.js finds them
        self.bone = {}
        for i, nm in enumerate(self.names):
            p = plain(nm); want = ALIAS.get(p) or next((k for k in ALIAS.values() if plain(k) == p), None)
            if want and want not in self.bone: self.bone[want] = i
        # the skinned vertices at rest, per mesh, with each vertex's dominant bone
        self.parts = []
        for i, n in enumerate(N):
            if 'mesh' not in n or 'skin' not in n or self.names[i] in hide: continue
            sk = J['skins'][n['skin']]; joints = sk['joints']; ibm = self.acc(sk['inverseBindMatrices']).reshape(-1, 4, 4).transpose(0, 2, 1)
            for prim in J['meshes'][n['mesh']]['primitives']:
                a = prim['attributes']; P = self.acc(a['POSITION']).astype(float); Jt = self.acc(a['JOINTS_0']).astype(int); Wt = self.acc(a['WEIGHTS_0']).astype(float)
                if Wt.max() > 1.5: Wt = Wt / Wt.sum(1, keepdims=True)                      # normalised integer weights
                nodes = np.array(joints)[Jt]                                                  # (v, 4) node indices
                self.parts.append({'mesh': self.names[i], 'P': P, 'nodes': nodes, 'w': Wt, 'ibm': {j: ibm[k] for k, j in enumerate(joints)}, 'dom': nodes[np.arange(len(P)), Wt.argmax(1)]})

    def _w(self, i):
        if self.world[i] is None: self.world[i] = self.local[i] if i not in self.parent else self._w(self.parent[i]) @ self.local[i]
        return self.world[i]

    def acc(self, k):
        a = self.J['accessors'][k]; v = self.J['bufferViews'][a['bufferView']]; off = v.get('byteOffset', 0) + a.get('byteOffset', 0)
        dt = np.dtype(CT[a['componentType']]); nc = NC[a['type']]; stride = v.get('byteStride', 0)
        if stride and stride != dt.itemsize * nc:
            raw = np.frombuffer(self.bin, np.uint8, count=stride * a['count'], offset=off).reshape(a['count'], stride)
            return raw[:, :dt.itemsize * nc].copy().view(dt).reshape(a['count'], nc)
        return np.frombuffer(self.bin, dt, count=a['count'] * nc, offset=off).reshape(a['count'], nc)

    def pose(self, motion, f):
        """every node's world matrix at frame f of a motion.json (rmo solve), the hips held where they rest: bones the motion names take its
        world turn x their own rest turn; every other node rides its parent with its rest local transform"""
        W = {}; Rrest = self.world
        named = {self.bone[b]: b for b in motion['bones'] if b in self.bone}
        if not hasattr(self, '_order'): self._order = sorted(range(len(self.names)), key=self.depth)
        for i in self._order:
            p = self.parent.get(i)
            M = np.eye(4)
            if i in named:
                q = motion['world'][named[i]][f]; Rw = Rot.from_quat(q).as_matrix() @ Rrest[i][:3, :3]
                base = self.world[i] if p is None else W[p] @ self.local[i]                # the rest translation, carried by the posed parent
                M[:3, :3] = Rw; M[:3, 3] = base[:3, 3]                                      # Rw keeps the rest matrix's own scale
            else: M = self.world[i] if p is None else W[p] @ self.local[i]
            W[i] = M
        return W

    def depth(self, i):
        d = 0
        while i in self.parent: i = self.parent[i]; d += 1
        return d

    def skin(self, W, part):
        """a mesh part's vertices posed by the node matrices W (linear blend skinning, glTF's rule)"""
        out = np.zeros((len(part['P']), 3)); Ph = np.c_[part['P'], np.ones(len(part['P']))]
        for k in range(part['nodes'].shape[1]):
            js = part['nodes'][:, k]; w = part['w'][:, k]
            for j in np.unique(js[w > 0]):
                sel = (js == j) & (w > 0); S = W[j] @ part['ibm'][j]
                out[sel] += w[sel, None] * (Ph[sel] @ S.T)[:, :3]
        return out
