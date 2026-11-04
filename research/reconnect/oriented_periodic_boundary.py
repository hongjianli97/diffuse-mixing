"""Opt-in exact isometry boundary adapter; original geometry/flux backend unchanged."""
from dataclasses import dataclass
from fractions import Fraction as F
import hashlib,math
import numpy as np
from periodic_voronoi_fv import build_periodic_faces,coefficient_packet


def rational(value):
    if isinstance(value,(bool,np.bool_)):raise ValueError('boolean is not an exact coordinate')
    if isinstance(value,(float,np.floating)):
        if not math.isfinite(float(value)):raise ValueError('finite exact coordinate required')
        return F(float(value))
    if isinstance(value,np.integer):return F(int(value))
    try:return F(value)
    except (ValueError,TypeError,ZeroDivisionError,OverflowError) as e:raise ValueError('finite rational coordinate required') from e


def lossless_float(q):
    q=rational(q)
    try:x=float(q)
    except OverflowError as e:raise ValueError('coordinate not representable by original float backend') from e
    if not math.isfinite(x) or F(x)!=q:raise ValueError('lossy coordinate conversion rejected; no rounding repair')
    return x


@dataclass(frozen=True)
class OrientedPeriodicRectangle:
    origin: tuple
    lengths: tuple
    orientation: tuple

    def __post_init__(self):
        try:
            o=tuple(map(rational,self.origin));L=tuple(map(rational,self.lengths));E=tuple(tuple(map(rational,row)) for row in self.orientation)
        except TypeError as e:raise ValueError('complete exact rectangular boundary required') from e
        if len(o)!=2 or len(L)!=2 or len(E)!=2 or any(len(row)!=2 for row in E) or min(L)<=0:raise ValueError('complete positive 2D boundary required')
        if any(sum(E[k][i]*E[k][j] for k in range(2))!=int(i==j) for i in range(2) for j in range(2)) or E[0][0]*E[1][1]-E[0][1]*E[1][0]!=1:
            raise ValueError('exact proper orthogonality required; no orthonormalization')
        for v in L:lossless_float(v)
        object.__setattr__(self,'origin',o);object.__setattr__(self,'lengths',L);object.__setattr__(self,'orientation',E)

    def push(self,q):
        return tuple(self.origin[i]+sum(self.orientation[i][j]*q[j] for j in range(2)) for i in range(2))

    def pull(self,p):
        return tuple(sum(self.orientation[k][i]*(p[k]-self.origin[k]) for k in range(2)) for i in range(2))

    def packet(self):
        return {'origin_exact':list(map(str,self.origin)),'periods_exact':list(map(str,self.lengths)),
          'orientation_rows_exact':[[str(v) for v in row] for row in self.orientation],
          'basis_columns_exact':[[str(self.orientation[i][j]*self.lengths[j]) for i in range(2)] for j in range(2)]}

    @classmethod
    def from_packet(cls,p):
        b=cls(p['origin_exact'],p['periods_exact'],p['orientation_rows_exact'])
        if 'basis_columns_exact' in p:
            supplied=tuple(tuple(map(rational,row)) for row in p['basis_columns_exact'])
            expected=tuple(tuple(map(F,row)) for row in b.packet()['basis_columns_exact'])
            if supplied!=expected:raise ValueError('period basis inconsistent with given boundary')
        return b


def build_oriented_periodic_faces(ids,world_positions_exact,rho,D,boundary,progress=None):
    if not isinstance(boundary,OrientedPeriodicRectangle):raise ValueError('given exact boundary required')
    try:points=tuple(tuple(map(rational,row)) for row in world_positions_exact)
    except TypeError as e:raise ValueError('complete 2D world points required') from e
    ident=np.asarray(ids);density=np.asarray(rho,dtype=float);n=len(points)
    if ident.shape!=(n,) or ident.dtype.kind not in 'iu' or len(set(map(int,ident)))!=n or n<4:raise ValueError('complete unique integer IDs required')
    if any(len(p)!=2 for p in points) or len(set(points))!=n:raise ValueError('distinct complete world points required')
    if density.shape!=(n,) or not np.isfinite(density).all() or np.any(density<=0):raise ValueError('complete positive given point density required')
    local=tuple(boundary.pull(p) for p in points)
    if any(not 0<=q[k]<boundary.lengths[k] for q in local for k in range(2)):raise ValueError('world point outside given oriented fundamental cell; no wrapping repair')
    converted=np.array([[lossless_float(v) for v in q] for q in local])
    graph=build_periodic_faces(ident,converted,density,D,tuple(lossless_float(v) for v in boundary.lengths),1,progress)
    packet=coefficient_packet(graph)
    world={k:v for k,v in packet.items() if k not in ('vertices_exact','periods_exact')}
    world.update(boundary.packet())
    world['points_world_exact']=[[str(v) for v in p] for p in points]
    world['vertices_world_exact']=[[str(v) for v in boundary.push(tuple(map(F,q)))] for q in packet['vertices_exact']]
    certificate={'exact_proper_isometry_verified':True,'local_coordinates_float64_lossless':True,'period_lengths_float64_lossless':True,
      'world_input_points_preserved':True,'points_wrapped_jittered_or_fitted':False,'provided_source_local_labels_or_mass_scalar_reference':False,
      'local_coordinate_C_order_sha256':hashlib.sha256(converted.tobytes(order='C')).hexdigest(),'nodes':n}
    return packet,world,certificate
