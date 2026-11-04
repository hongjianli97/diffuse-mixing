"""Explicit fixed-connectivity 1D material-face input and linear transport.

Validates declared identities and modeled geometry. It cannot prove that the
producer supplied actual material surfaces; that is a separate physical audit.
One square meter cross section, SI inputs, no remap or extrapolation.
"""
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
import hashlib,json
import numpy as np
from fv_density_geometry import cellwise_conductance

SI_UNITS={'time':'s','position':'m','mass':'kg','cross_section_area':'m^2'}


def _ids(value,name):
    ar=np.asarray(value)
    if ar.ndim!=1 or ar.dtype.kind not in 'iuU' or len(np.unique(ar))!=len(ar):
        raise ValueError(name+' must be unique string or integer identities')
    if ar.dtype.kind in 'US' and np.any(ar==''):
        raise ValueError(name+' contains an empty identity')
    return ar.copy()


@dataclass(frozen=True)
class MaterialFaceHistory:
    time: object
    positions: object
    face_ids: object
    cell_ids: object
    cell_faces: object
    mass: object
    c0: object
    units: object
    metadata: object
    frame_face_ids: object=None
    boundary_mode: str='material_no_flux'
    cross_section_area: float=1.

    def __post_init__(self):
        t=np.array(self.time,dtype=float,copy=True);x=np.array(self.positions,dtype=float,copy=True)
        f=_ids(self.face_ids,'face_ids');c=_ids(self.cell_ids,'cell_ids')
        cf=np.array(self.cell_faces,copy=True);m=np.array(self.mass,dtype=float,copy=True);c0=np.array(self.c0,dtype=float,copy=True)
        if t.ndim!=1 or len(t)<2 or not np.isfinite(t).all() or np.any(np.diff(t)<=0):
            raise ValueError('at least two finite strictly increasing sample times required')
        if len(f)<3 or len(c)!=len(f)-1 or x.shape!=(len(t),len(f)):
            raise ValueError('fixed face/cell count or position history shape mismatch')
        if not np.isfinite(x).all() or np.any(np.diff(x,axis=1)<=0):
            raise ValueError('invalid coordinate, identity crossing or nonpositive volume')
        expected=np.column_stack((f[:-1],f[1:]))
        if cf.dtype.kind not in 'iuU' or cf.shape!=expected.shape or not np.array_equal(cf,expected):
            raise ValueError('cell_faces must be the declared ordered fixed shared-face chain')
        if m.shape!=(len(c),) or c0.shape!=m.shape or not np.isfinite(m).all() or np.any(m<=0):
            raise ValueError('fixed positive cell masses and matching c0 required')
        if not np.isfinite(c0).all() or np.any(c0<0) or np.any(c0>1):
            raise ValueError('initial mass fractions must lie in [0,1], no repair')
        if dict(self.units)!=SI_UNITS or self.cross_section_area!=1.:
            raise ValueError('explicit SI units and one square meter cross section required')
        if self.boundary_mode!='material_no_flux':
            raise ValueError('only closed material_no_flux endpoints supported')
        metadata=json.loads(json.dumps(dict(self.metadata),allow_nan=False))
        if metadata.get('geometry_kind')!='material_faces':
            raise ValueError('geometry_kind=material_faces declaration required; not proof of materiality')
        frame_ids=None
        if self.frame_face_ids is not None:
            frame_ids=np.array(self.frame_face_ids,copy=True)
            if frame_ids.shape!=x.shape or not np.all(frame_ids==f[None,:]):
                raise ValueError('per-frame face identities changed, lost or reordered')
        for key,value in [('time',t),('positions',x),('face_ids',f),('cell_ids',c),('cell_faces',cf),('mass',m),('c0',c0)]:
            value.setflags(write=False);object.__setattr__(self,key,value)
        if frame_ids is not None:frame_ids.setflags(write=False)
        object.__setattr__(self,'frame_face_ids',frame_ids)
        object.__setattr__(self,'units',MappingProxyType(dict(SI_UNITS)))
        object.__setattr__(self,'metadata',MappingProxyType(metadata))

    @property
    def identity_digest(self):
        data={'face_ids':self.face_ids.tolist(),'cell_ids':self.cell_ids.tolist(),
              'cell_faces':self.cell_faces.tolist(),'mass':self.mass.tolist(),'c0':self.c0.tolist(),
              'units':dict(self.units),'boundary_mode':self.boundary_mode,'cross_section_area':self.cross_section_area}
        return hashlib.sha256(json.dumps(data,sort_keys=True,separators=(',',':')).encode()).hexdigest()

    def evaluate(self,query):
        q=np.asarray(query,dtype=float)
        if q.ndim>1 or not np.isfinite(q).all() or np.any(q<self.time[0]) or np.any(q>self.time[-1]):
            raise ValueError('query must be finite and within sampled interval; no extrapolation')
        flat=q.ravel();i=np.searchsorted(self.time,flat,side='right')-1
        i=np.where(flat==self.time[-1],len(self.time)-2,i)
        dt=self.time[i+1]-self.time[i];theta=(flat-self.time[i])/dt
        x=self.positions[i]+theta[:,None]*(self.positions[i+1]-self.positions[i])
        # Exactly reproduce the final input frame, including floating bits.
        end=flat==self.time[-1]
        x[end]=self.positions[-1]
        w=(self.positions[i+1]-self.positions[i])/dt[:,None]
        V=np.diff(x,axis=1)
        if np.any(V<=0):raise RuntimeError('linear interpolation lost positivity')
        shape=q.shape
        return {'positions':x.reshape(shape+(len(self.face_ids),)),
                'face_velocity':w.reshape(shape+(len(self.face_ids),)),
                'volume':V.reshape(shape+(len(self.cell_ids),)),
                'density':(self.mass/V).reshape(shape+(len(self.cell_ids),)),
                'interval_index':i.reshape(shape)}

    def conductance(self,query,D):
        geometry=self.evaluate(query)
        return cellwise_conductance(geometry['positions'],geometry['density'],D)

    def geometric_conservation(self):
        dt=np.diff(self.time);velocity=np.diff(self.positions,axis=0)/dt[:,None]
        dV=np.diff(np.diff(self.positions,axis=1),axis=0)
        residual=dV-dt[:,None]*np.diff(velocity,axis=1)
        total=np.diff(self.positions,axis=1).sum(axis=1)-(self.positions[:,-1]-self.positions[:,0])
        return {'maximum_volume_increment_residual':float(np.max(abs(residual))),
                'maximum_partition_volume_residual':float(np.max(abs(total))),
                'minimum_sampled_volume':float(np.min(np.diff(self.positions,axis=1))),
                'identity_digest':self.identity_digest,
                'materiality_physically_verified_by_this_check':False}

    def save(self,path):
        path=Path(path)
        if path.suffix!='.npz':raise ValueError('explicit .npz record path required')
        if path.exists():raise FileExistsError('refusing to overwrite a face record')
        np.savez_compressed(path,time=self.time,positions=self.positions,face_ids=self.face_ids,
                cell_ids=self.cell_ids,cell_faces=self.cell_faces,mass=self.mass,c0=self.c0,
                units_json=np.array(json.dumps(dict(self.units),sort_keys=True)),
                metadata_json=np.array(json.dumps(dict(self.metadata),sort_keys=True)),
                boundary_mode=np.array(self.boundary_mode),cross_section_area=np.array(self.cross_section_area),
                frame_face_ids=np.broadcast_to(self.face_ids,self.positions.shape))

    @classmethod
    def load(cls,path):
        with np.load(path,allow_pickle=False) as z:
            return cls(time=z['time'],positions=z['positions'],face_ids=z['face_ids'],cell_ids=z['cell_ids'],
                       cell_faces=z['cell_faces'],mass=z['mass'],c0=z['c0'],
                       units=json.loads(str(z['units_json'])),metadata=json.loads(str(z['metadata_json'])),
                       boundary_mode=str(z['boundary_mode']),cross_section_area=float(z['cross_section_area']),
                       frame_face_ids=z['frame_face_ids'])
