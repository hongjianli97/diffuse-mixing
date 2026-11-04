"""One fixed 2D polygon mesh: area GCL, shared edges and fluid mass audit.

No 2D scalar diffusion solver or ORC is enabled by this geometry experiment.
"""
from pathlib import Path
import argparse,json,platform
import numpy as np
import scipy
from becker import Becker
from becker_polygon_benchmark import quad_connectivity,to_world,true_vertices,actual_polygon_mass,direct_density_mass,physical_edge_mass_flux
from kernel_joint_study import verify
from material_polygon_history import MaterialPolygonHistory,POLYGON_UNITS,signed_area


def rejected_half_turn():
    x=np.array([[0.,0.],[1.,0.],[1.,1.],[0.,1.]])
    vertices=np.array(['v0','v1','v2','v3']);positions=np.stack((x,-x))
    try:
        MaterialPolygonHistory([0.,1.],positions,vertices,['c0'],vertices[None,:],[1.],[.5],
                               POLYGON_UNITS,{'geometry_kind':'material_quadrilateral_cells'})
        accepted=True;reason=None
    except ValueError as error:accepted=False;reason=str(error)
    return {'both_endpoint_signed_areas':signed_area(positions).tolist(),
            'midpoint_signed_area':float(signed_area((positions[0]+positions[1])/2)),
            'rejected':not accepted,'reason':reason},positions


def main(source,output):
    root=Path(__file__).resolve().parents[2];source=Path(source).resolve();output=Path(output).resolve();verify(root,source)
    old=json.loads((source/'config.json').read_text());p=old['physical_parameters'];T=p['T']
    density=root/'runs'/Path(old['source']).name;verify(root,density)
    with np.load(density/'geometry-audit.npz') as z:
        source_times=z['time'];source_faces=z['material_faces'];mass1=z['mass'];initial1=z['c0'];edges=z['mass_edges']
    flow=Becker(delta=p['delta'],mach=p['Mach'],gamma=p['gamma'],rho1=p['rho1'],p1=p['p1'],R=p['R'])
    nx=len(mass1);ny=2;y=np.linspace(-.5,.5,ny+1);dy=np.diff(y)
    cell_indices=quad_connectivity(nx,ny);vertex_ids=np.array([f'vertex-{i:03d}' for i in range((nx+1)*(ny+1))])
    cell_ids=np.array([f'cell-{i:03d}' for i in range(nx*ny)]);cv=vertex_ids[cell_indices]
    mass=np.concatenate([mass1*h for h in dy]);c0=np.tile(initial1,ny)
    # The producer transforms archived true face positions, not a reconstructed
    # node cloud. No analytic motion is called by the geometry interface.
    base=np.stack((np.tile(source_faces,(1,len(y))),np.broadcast_to(np.repeat(y,nx+1),(len(source_times),(nx+1)*len(y)))),axis=-1)
    supplied=to_world(base,source_times,T)
    output.mkdir(parents=True,exist_ok=False)
    cfg={'study':'one fixed 2D quadrilateral material mesh, geometry and physical mass only',
         'source':str(source),'inherited_physical_parameters':p,'time':T,'mesh_columns':nx,'mesh_rows':ny,
         'kinematic_motion':{'rotation':'theta(t)=pi*t/(3T)','shear':'k(t)=.6*sin(pi*t/T)',
                             'translation':'(.1*t/T,-.05*t/T)','area_factor':'det(R*shear)=1'},
         'sampling_intervals':[8,16,32],'audit_intervals':512,'audit_time_control_intervals':256,
         'mass_quadrature_orders':[16,32],'flux_quadrature_orders':[16,32],'mass_derivative_eps':1e-6,
         'units':POLYGON_UNITS,'input_geometry':'synthetic exact material vertices at knots, fixed CCW quad connectivity',
         'coefficient_inputs':'positions, identities, connectivity, fixed mass only; no local F/C fits',
         'fluid_scope':'prescribed 2D continuity-preserving transformation, not unforced 2D Navier-Stokes/DNS',
         'scalar_diffusion_solver_run':False,'physical_acceptance':False,'fluid_orc_enabled':False}
    (output/'config.json').write_text(json.dumps(cfg,indent=2)+'\n')
    times=np.linspace(0,T,cfg['audit_intervals']+1)
    true=true_vertices(flow,edges,y,times,T);true_area=np.tile(np.diff(flow.x_mu(edges[None,:],times[:,None]),axis=1)*dy[0],(1,ny))
    rows=[];identity=None
    for intervals in cfg['sampling_intervals']:
        take=np.arange(0,len(source_times),(len(source_times)-1)//intervals)
        history=MaterialPolygonHistory(source_times[take],supplied[take],vertex_ids,cell_ids,cv,mass,c0,
                   POLYGON_UNITS,{'geometry_kind':'material_quadrilateral_cells',
                                  'provenance':'known material vertices; straight edges remain material in this benchmark'})
        history.save(output/f'polygon-record-{intervals}.npz');history=MaterialPolygonHistory.load(output/f'polygon-record-{intervals}.npz')
        if identity is None:identity=history.identity_digest
        if history.identity_digest!=identity:raise RuntimeError('sampling changed physical identities, masses or connectivity')
        model=history.evaluate(times);area=model['area'];positions=model['positions']
        M16=actual_polygon_mass(flow,positions,cell_indices,times,T,order=16)
        M32=actual_polygon_mass(flow,positions,cell_indices,times,T,order=32)
        relative_mass=abs(M32-mass)/mass
        midpoint=(history.time[:-1]+history.time[1:])/2;eps=cfg['mass_derivative_eps']
        q16=physical_edge_mass_flux(history,flow,midpoint,T,order=16)
        q32=physical_edge_mass_flux(history,flow,midpoint,T,order=32)
        transport=-q32@history.incidence.T
        plus=history.evaluate(midpoint+eps)['positions'];minus=history.evaluate(midpoint-eps)['positions']
        Mplus=actual_polygon_mass(flow,plus,cell_indices,midpoint+eps,T,order=32)
        Mminus=actual_polygon_mass(flow,minus,cell_indices,midpoint-eps,T,order=32)
        finite=(Mplus-Mminus)/(2*eps)
        worst=np.unravel_index(np.argmax(relative_mass),relative_mass.shape)
        spot=direct_density_mass(flow,positions[worst[0],cell_indices[worst[1]]],times[worst[0]],T,order=32)
        gc=history.geometric_conservation();knots=np.arange(0,len(times),cfg['audit_intervals']//intervals)
        row={'sampling_intervals':intervals,'vertices':len(vertex_ids),'cells':len(cell_ids),'edges':len(history.edge_ids),
             'shared_edges':int(np.count_nonzero(~history.boundary_edge_mask)),
             'boundary_edges':int(np.count_nonzero(history.boundary_edge_mask)),
             'max_relative_area_error':float(np.max(abs(area-true_area)/true_area)),
             'max_vertex_position_error':float(np.max(np.linalg.norm(positions-true,axis=-1))),
             'max_actual_fluid_mass_relative_mismatch':float(relative_mass.max()),
             'max_knot_actual_fluid_mass_relative_mismatch':float(relative_mass[knots].max()),
             'audit_256_max_mass_mismatch':float(relative_mass[::2].max()),
             'audit_512_minus_256_max_mass_mismatch':float(relative_mass.max()-relative_mass[::2].max()),
             'max_mass_quadrature_relative_difference':float(np.max(abs(M16-M32)/mass)),
             'max_edge_flux_quadrature_absolute_difference':float(np.max(abs(q16-q32))),
             'max_Reynolds_mass_rate_residual':float(np.max(abs(finite-transport))),
             'max_observed_cell_mass_rate_abs':float(np.max(abs(transport))),
             'global_mass_rate_boundary_flux_residual':float(np.max(abs(transport.sum(axis=1)+q32@history.incidence.sum(axis=0)))),
             'independent_2D_density_spot_mass_difference':abs(spot-float(M32[worst])),
             'mass_spot_time':float(times[worst[0]]),'mass_spot_cell':int(worst[1]),**gc}
        rows.append(row)
        np.savez_compressed(output/f'audit-{intervals}.npz',time=times,positions=positions,true_positions=true,
                   modeled_area=area,true_area=true_area,fixed_mass=mass,c0=c0,actual_mass=M32,actual_mass_order16=M16,
                   modeled_density=model['density'],edge_area_flux=model['edge_area_flux'],cell_area_rate=model['cell_area_rate'],
                   vertex_ids=vertex_ids,cell_ids=cell_ids,cell_vertices=cell_indices,edge_vertices=history.edge_vertex_indices,
                   incidence=history.incidence,boundary_edge_mask=history.boundary_edge_mask)
        np.savez_compressed(output/f'reynolds-{intervals}.npz',time=midpoint,edge_outward_mass_flux=q32,
                   mass_rate_from_edge_flux=transport,mass_rate_from_independent_difference=finite,
                   cell_ids=cell_ids,edge_ids=history.edge_ids,incidence=history.incidence)
        print('2D polygon sampling',intervals,'area error',row['max_relative_area_error'],
              'fluid mass error',row['max_actual_fluid_mass_relative_mismatch'],flush=True)
    negative,negative_positions=rejected_half_turn()
    np.savez_compressed(output/'rejected-half-turn.npz',time=np.array([0.,1.]),positions=negative_positions)
    orders={}
    for key in ['max_relative_area_error','max_actual_fluid_mass_relative_mismatch']:
        values=[r[key] for r in rows];orders[key]=[float(np.log2(a/b)) for a,b in zip(values[:-1],values[1:])]
    result={'input_identity_digest':identity,'geometry_and_transport':rows,'pairwise_observed_orders':orders,
            'half_turn_negative_control':negative,'inherited_total_mass_difference':float(mass.sum()-mass1.sum()),
            'physical_acceptance':False,'scalar_diffusion_solver_run':False,
            'limits':['one prescribed continuity benchmark','straight convex quadrilateral local geometry only',
                      'general global nonoverlap not certified by input validator','not curved material-edge or 2D DNS acceptance']}
    (output/'results.json').write_text(json.dumps(result,indent=2)+'\n')
    (output/'environment.json').write_text(json.dumps({'python':platform.python_version(),'numpy':np.__version__,'scipy':scipy.__version__,'BLAS_threads_command':4},indent=2)+'\n')
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('source');p.add_argument('--output',required=True)
    a=p.parse_args();main(a.source,a.output)
