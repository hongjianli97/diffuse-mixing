"""Fixed 2D convex quadrilateral identities and shared-edge geometric flux.

No remeshing, area absolute value, clipping, fitted F/C or physical exchange
operator. Validates local convexity through each linear interpolation interval;
global non-overlap and fluid materiality are separate physical/geometric audits.
"""
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
import hashlib,json
import numpy as np
from material_face_history import _ids

POLYGON_UNITS={'time':'s','position':'m','mass':'kg','thickness':'m'}


def cross2(a,b):
    return a[...,0]*b[...,1]-a[...,1]*b[...,0]


def signed_area(p):
    v=p-p[...,:1,:]
    return .5*np.sum(cross2(v,np.roll(v,-1,axis=-2)),axis=-1)


def quadratic_minimum(a,b,c):
    """Minimum of a+b*t+c*t^2 on [0,1], with no sign repair."""
    result=np.minimum(a,a+b+c)
    tau=np.zeros_like(result)
    np.divide(-b,2*c,out=tau,where=c>0)
    inner=(c>0)&(tau>0)&(tau<1)
    result[inner]=np.minimum(result[inner],a[inner]+b[inner]*tau[inner]+c[inner]*tau[inner]**2)
    return result


def interpolation_minima(p0,p1):
    v=p0-p0[...,:1,:];d=(p1-p1[...,:1,:])-v
    vn=np.roll(v,-1,axis=-2);dn=np.roll(d,-1,axis=-2)
    area=quadratic_minimum(.5*np.sum(cross2(v,vn),axis=-1),
             .5*np.sum(cross2(d,vn)+cross2(v,dn),axis=-1),.5*np.sum(cross2(d,dn),axis=-1))
    e=np.roll(p0,-1,axis=-2)-p0;f=(np.roll(p1,-1,axis=-2)-p1)-e
    en=np.roll(e,-1,axis=-2);fn=np.roll(f,-1,axis=-2)
    turns=quadratic_minimum(cross2(e,en),cross2(f,en)+cross2(e,fn),cross2(f,fn))
    return area,turns


def mesh_edge_area_flux(positions,velocity,edge_vertices):
    a,b=edge_vertices.T
    tangent=positions[...,b,:]-positions[...,a,:]
    mean_velocity=(velocity[...,a,:]+velocity[...,b,:])/2
    return mean_velocity[...,0]*tangent[...,1]-mean_velocity[...,1]*tangent[...,0]


@dataclass(frozen=True)
class MaterialPolygonHistory:
    time: object
    positions: object
    vertex_ids: object
    cell_ids: object
    cell_vertices: object
    mass: object
    c0: object
    units: object
    metadata: object
    frame_vertex_ids: object=None
    thickness: float=1.
    boundary_mode: str='material_no_flux'

    def __post_init__(self):
        t=np.array(self.time,dtype=float,copy=True);x=np.array(self.positions,dtype=float,copy=True)
        vertices=_ids(self.vertex_ids,'vertex_ids');cells=_ids(self.cell_ids,'cell_ids')
        cv=np.array(self.cell_vertices,copy=True);m=np.array(self.mass,dtype=float,copy=True);c0=np.array(self.c0,dtype=float,copy=True)
        if t.ndim!=1 or len(t)<2 or not np.isfinite(t).all() or np.any(np.diff(t)<=0):
            raise ValueError('strictly increasing finite sample times required')
        if x.shape!=(len(t),len(vertices),2) or not np.isfinite(x).all():
            raise ValueError('finite positions[S,V,2] required')
        if not len(cells) or cv.shape!=(len(cells),4) or cv.dtype.kind not in 'iuU':
            raise ValueError('fixed four-vertex cell connectivity required')
        lookup={key:i for i,key in enumerate(vertices.tolist())}
        try:indices=np.array([[lookup[key] for key in row] for row in cv.tolist()],dtype=int)
        except KeyError as error:raise ValueError('cell references unknown vertex identity') from error
        if any(len(set(row))!=4 for row in indices.tolist()) or len(set(map(frozenset,indices.tolist())))!=len(cells):
            raise ValueError('duplicate vertex in cell or duplicate cell geometry')
        if len(np.unique(indices))!=len(vertices):raise ValueError('unused vertex identities unsupported')
        if m.shape!=(len(cells),) or c0.shape!=m.shape or not np.isfinite(m).all() or np.any(m<=0):
            raise ValueError('fixed positive mass per cell required')
        if not np.isfinite(c0).all() or np.any(c0<0) or np.any(c0>1):raise ValueError('valid initial mass fractions required')
        if dict(self.units)!=POLYGON_UNITS or self.thickness!=1. or self.boundary_mode!='material_no_flux':
            raise ValueError('SI, unit thickness and material_no_flux boundary required')
        meta=json.loads(json.dumps(dict(self.metadata),allow_nan=False))
        if meta.get('geometry_kind')!='material_quadrilateral_cells':raise ValueError('explicit material-cell declaration required')
        frame_ids=None
        if self.frame_vertex_ids is not None:
            frame_ids=np.array(self.frame_vertex_ids,copy=True)
            if frame_ids.shape!=x.shape[:2] or not np.all(frame_ids==vertices[None,:]):
                raise ValueError('vertex identities changed, disappeared or were reordered')
        pairs=sorted({tuple(sorted((int(a),int(b)))) for row in indices for a,b in zip(row,np.roll(row,-1))})
        edge_vertices=np.array(pairs,dtype=int);edge_map={pair:i for i,pair in enumerate(pairs)}
        ce=np.empty((len(cells),4),dtype=int);signs=np.empty_like(ce)
        for i,row in enumerate(indices):
            for k,(a,b) in enumerate(zip(row,np.roll(row,-1))):
                ce[i,k]=edge_map[tuple(sorted((int(a),int(b))))];signs[i,k]=1 if a<b else -1
        incidence=np.zeros((len(cells),len(pairs)),dtype=int)
        incidence[np.arange(len(cells))[:,None],ce]=signs
        counts=np.count_nonzero(incidence,axis=0);orientation=incidence.sum(axis=0)
        if np.any(counts>2) or np.any(orientation[counts==2]!=0):raise ValueError('nonmanifold edge or inconsistent shared-edge orientation')
        boundary=counts==1
        degree=np.bincount(edge_vertices[boundary].ravel(),minlength=len(vertices))
        if np.any((degree!=0)&(degree!=2)) or len(vertices)-len(pairs)+len(cells)!=1:
            raise ValueError('only a connected disk-like quadrilateral complex is supported')
        reached={0};pending=[0]
        while pending:
            i=pending.pop()
            neighbours=np.flatnonzero(np.any(incidence[:,ce[i]]!=0,axis=1))
            for j in neighbours:
                if int(j) not in reached:reached.add(int(j));pending.append(int(j))
        if len(reached)!=len(cells):raise ValueError('disconnected cell complex')
        areas,turns=interpolation_minima(x[:-1,indices],x[1:,indices])
        if np.any(areas<=0) or np.any(turns<=0):raise ValueError('cell loses positive orientation or strict convexity inside a sampling interval')
        extra={'cell_vertex_indices':indices,'edge_vertex_indices':edge_vertices,
               'edge_ids':np.array([f'edge-{i:04d}' for i in range(len(pairs))]),
               'cell_edge_indices':ce,'cell_edge_signs':signs,'incidence':incidence,
               'boundary_edge_mask':boundary,'minimum_interval_area':float(areas.min()),
               'minimum_interval_corner_orientation':float(turns.min())}
        for key,value in [('time',t),('positions',x),('vertex_ids',vertices),('cell_ids',cells),('cell_vertices',cv),('mass',m),('c0',c0)]+list(extra.items()):
            if isinstance(value,np.ndarray):value.setflags(write=False)
            object.__setattr__(self,key,value)
        if frame_ids is not None:frame_ids.setflags(write=False)
        object.__setattr__(self,'frame_vertex_ids',frame_ids)
        object.__setattr__(self,'units',MappingProxyType(dict(POLYGON_UNITS)))
        object.__setattr__(self,'metadata',MappingProxyType(meta))

    @property
    def identity_digest(self):
        data={'vertex_ids':self.vertex_ids.tolist(),'cell_ids':self.cell_ids.tolist(),'cell_vertices':self.cell_vertices.tolist(),
              'mass':self.mass.tolist(),'c0':self.c0.tolist(),'units':dict(self.units),'thickness':self.thickness,'boundary_mode':self.boundary_mode}
        return hashlib.sha256(json.dumps(data,sort_keys=True,separators=(',',':')).encode()).hexdigest()

    def evaluate(self,query):
        q=np.asarray(query,dtype=float)
        if q.ndim>1 or not np.isfinite(q).all() or np.any(q<self.time[0]) or np.any(q>self.time[-1]):
            raise ValueError('finite scalar/vector query inside sampling interval required')
        flat=q.ravel();i=np.searchsorted(self.time,flat,side='right')-1
        i=np.where(flat==self.time[-1],len(self.time)-2,i)
        dt=self.time[i+1]-self.time[i];theta=(flat-self.time[i])/dt
        x=self.positions[i]+theta[:,None,None]*(self.positions[i+1]-self.positions[i])
        x[flat==self.time[-1]]=self.positions[-1]
        velocity=(self.positions[i+1]-self.positions[i])/dt[:,None,None]
        area=signed_area(x[:,self.cell_vertex_indices]);flux=mesh_edge_area_flux(x,velocity,self.edge_vertex_indices)
        if np.any(area<=0):raise RuntimeError('invalid signed area; no repair permitted')
        shape=q.shape
        return {'positions':x.reshape(shape+(len(self.vertex_ids),2)),
                'vertex_velocity':velocity.reshape(shape+(len(self.vertex_ids),2)),
                'area':area.reshape(shape+(len(self.cell_ids),)),
                'density':(self.mass/area).reshape(shape+(len(self.cell_ids),)),
                'edge_area_flux':flux.reshape(shape+(len(self.edge_ids),)),
                'cell_area_rate':(flux@self.incidence.T).reshape(shape+(len(self.cell_ids),))}

    def geometric_conservation(self):
        dt=np.diff(self.time);velocity=np.diff(self.positions,axis=0)/dt[:,None,None]
        midpoint=(self.positions[:-1]+self.positions[1:])/2
        flux=mesh_edge_area_flux(midpoint,velocity,self.edge_vertex_indices)
        areas=signed_area(self.positions[:,self.cell_vertex_indices]);increment=np.diff(areas,axis=0)
        predicted=dt[:,None]*(flux@self.incidence.T)
        left_flux=mesh_edge_area_flux(self.positions[:-1],velocity,self.edge_vertex_indices)
        boundary_sign=self.incidence.sum(axis=0)
        return {'maximum_area_increment_residual':float(np.max(abs(increment-predicted))),
                'left_endpoint_sweep_residual_negative_control':float(np.max(abs(increment-dt[:,None]*(left_flux@self.incidence.T)))),
                'maximum_global_boundary_sweep_residual':float(np.max(abs(increment.sum(axis=1)-dt*(flux@boundary_sign)))),
                'internal_edge_orientation_sum_max_abs':int(np.max(abs(boundary_sign[~self.boundary_edge_mask]))) if np.any(~self.boundary_edge_mask) else 0,
                'minimum_area_over_linear_intervals':self.minimum_interval_area,
                'minimum_corner_orientation_over_intervals':self.minimum_interval_corner_orientation,
                'global_nonoverlap_certified':False,'physical_materiality_verified':False}

    def save(self,path):
        path=Path(path)
        if path.suffix!='.npz':raise ValueError('explicit .npz path required')
        if path.exists():raise FileExistsError('refusing to overwrite polygon record')
        np.savez_compressed(path,time=self.time,positions=self.positions,vertex_ids=self.vertex_ids,cell_ids=self.cell_ids,
                cell_vertices=self.cell_vertices,mass=self.mass,c0=self.c0,units_json=np.array(json.dumps(dict(self.units))),
                metadata_json=np.array(json.dumps(dict(self.metadata))),thickness=np.array(self.thickness),boundary_mode=np.array(self.boundary_mode),
                frame_vertex_ids=np.broadcast_to(self.vertex_ids,self.positions.shape[:2]),
                edge_vertices=self.vertex_ids[self.edge_vertex_indices],cell_edge_ids=self.edge_ids[self.cell_edge_indices],cell_edge_signs=self.cell_edge_signs)

    @classmethod
    def load(cls,path):
        with np.load(path,allow_pickle=False) as z:
            result=cls(time=z['time'],positions=z['positions'],vertex_ids=z['vertex_ids'],cell_ids=z['cell_ids'],cell_vertices=z['cell_vertices'],
                     mass=z['mass'],c0=z['c0'],units=json.loads(str(z['units_json'])),metadata=json.loads(str(z['metadata_json'])),
                     frame_vertex_ids=z['frame_vertex_ids'],thickness=float(z['thickness']),boundary_mode=str(z['boundary_mode']))
            if not (np.array_equal(result.vertex_ids[result.edge_vertex_indices],z['edge_vertices'])
                    and np.array_equal(result.edge_ids[result.cell_edge_indices],z['cell_edge_ids'])
                    and np.array_equal(result.cell_edge_signs,z['cell_edge_signs'])):
                raise ValueError('stored shared-edge identities or orientations disagree with connectivity')
            return result
