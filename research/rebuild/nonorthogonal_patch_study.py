"""One existing 2D mesh: linear mass-average face-flux patch and positivity.

Density, mass and geometry are inherited. Oracle mass centroids and face
coefficients remove density-closure ambiguity from this geometric diagnostic.
No variable-geometry diffusion solver or physical backend is promoted.
"""
from pathlib import Path
import argparse,json,platform
import numpy as np
import scipy
from scipy.special import roots_legendre
from becker import Becker
from becker_polygon_benchmark import to_world,density_velocity
from kernel_joint_study import verify
from material_polygon_history import MaterialPolygonHistory
from nonorthogonal_flux import internal_face_geometry,two_point_flux_matrix,scalar_reconstruction_reference,closed_operator_audit


def oracle_mass_centroids(flow,mass_edges,y_edges,t,T=.1,order=64):
    z,w=roots_legendre(order);a,b=mass_edges[:-1],mass_edges[1:]
    mu=(a+b)[:,None]/2+(b-a)[:,None]*z/2
    xi=(flow.x_mu(mu,t)@(w/2));ny=len(y_edges)-1
    canonical=np.column_stack((np.tile(xi,ny),np.repeat((y_edges[:-1]+y_edges[1:])/2,len(xi))))
    return to_world(canonical[None,...],np.array([t]),T)[0]


def oracle_face_coefficient(history,flow,positions,t,D,T=.1,order=64):
    edges=np.flatnonzero(~history.boundary_edge_mask);a,b=history.edge_vertex_indices[edges].T
    v=positions[b]-positions[a];length=np.linalg.norm(v,axis=1);z,w=roots_legendre(order);s=(z+1)/2
    points=positions[a,None,:]+s[None,:,None]*v[:,None,:]
    rho,_=density_velocity(flow,points[None,...],np.array([t]),T)
    return D*length*np.sum(rho[0]*(w/2),axis=1)


def patch_errors(F,states,exact,normalization):
    flux=F@states;residual=flux-exact
    return {'maximum_joint_basis_relative_flux_error':float(np.max(np.linalg.norm(residual,axis=1)/normalization)),
            'rms_joint_basis_relative_flux_error':float(np.sqrt(np.mean(np.sum(residual*residual,axis=1)/(normalization**2)))),
            'maximum_absolute_flux_error':float(np.max(abs(residual)))},flux


def main(source,output):
    root=Path(__file__).resolve().parents[2];source=Path(source).resolve();output=Path(output).resolve();verify(root,source)
    old=json.loads((source/'config.json').read_text());p=old['inherited_physical_parameters'];T=p['T'];D=p['D']
    history=MaterialPolygonHistory.load(source/'polygon-record-32.npz')
    face=root/'runs'/Path(old['source']).name;verify(root,face)
    facecfg=json.loads((face/'config.json').read_text());density=root/'runs'/Path(facecfg['source']).name
    with np.load(density/'geometry-audit.npz') as z:edges=z['mass_edges']
    y=np.linspace(-.5,.5,old['mesh_rows']+1)
    flow=Becker(delta=p['delta'],mach=p['Mach'],gamma=p['gamma'],rho1=p['rho1'],p1=p['p1'],R=p['R'])
    output.mkdir(parents=True,exist_ok=False)
    cfg={'study':'linear scalar mass-average flux consistency on the same frozen 2D mesh',
         'source':str(source),'record':'polygon-record-32.npz','physical_parameters':p,'times':[0.,T/2,T],
         'probe_fields':['.5+.1*x','.5+.1*y'],'gradient_amplitude_per_meter':.1,
         'density_and_state':'same physical density and fixed masses; exact linear averages use independently integrated mass centroids',
         'reference_face_coefficient':'D integral_edge rho ds; independently integrated 32/64 point control',
         'two_point_methods':['normal_distance','center_distance'],
         'reconstruction_reference':'fixed inverse-distance weighted linear scalar reconstruction; not fitted F/C or flux outcomes',
         'probe_time':1e-4,'patch_boundary_scope':'internal faces only; linear probes do not satisfy the closed no-flux boundary',
         'operator_boundary_scope':'separate frozen matrix audit uses zero exterior flux',
         'variable_geometry_diffusion_solver_run':False,'physical_acceptance':False,'fluid_orc_enabled':False}
    (output/'config.json').write_text(json.dumps(cfg,indent=2)+'\n')
    rows=[];amplitude=cfg['gradient_amplitude_per_meter']
    for t in cfg['times']:
        positions=history.evaluate(t)['positions']
        center32=oracle_mass_centroids(flow,edges,y,t,T,32);centers=oracle_mass_centroids(flow,edges,y,t,T,64)
        k32=oracle_face_coefficient(history,flow,positions,t,D,T,32);coefficient=oracle_face_coefficient(history,flow,positions,t,D,T,64)
        g=internal_face_geometry(history,positions,centers)
        states=.5+amplitude*centers;exact=-amplitude*coefficient[:,None]*g['normal']
        if np.any(states<0) or np.any(states>1):raise RuntimeError('manufactured mass fractions outside [0,1]')
        row={'time':t,'maximum_nonorthogonality_degrees':float(np.degrees(g['angle_radians']).max()),
             'maximum_any_scalar_conductance_error_lower_bound':float(g['any_nonnegative_scalar_relative_error_lower_bound'].max()),
             'mass_centroid_quadrature_max_abs_difference':float(np.max(abs(centers-center32))),
             'face_coefficient_quadrature_max_relative_difference':float(np.max(abs(coefficient-k32)/coefficient)),
             'maximum_mass_vs_geometric_centroid_distance':float(np.max(np.linalg.norm(centers-positions[history.cell_vertex_indices].mean(axis=1),axis=1))),
             'minimum_patch_mass_fraction':float(states.min()),'maximum_patch_mass_fraction':float(states.max()),'methods':{}}
        saved={'positions':positions,'mass':history.mass,'original_c0':history.c0,'mass_centroids':centers,'mass_centroids_order32':center32,
               'patch_states':states,'exact_internal_face_flux':exact,'integrated_face_coefficient':coefficient,
               'edge_indices':g['edge_indices'],'edge_owner':g['owner'],'edge_other':g['other'],'face_normal':g['normal'],
               'mass_centroid_displacement':g['displacement'],'scalar_flux_error_lower_bound':g['any_nonnegative_scalar_relative_error_lower_bound']}
        for method in cfg['two_point_methods']:
            F,weights=two_point_flux_matrix(g,coefficient,len(history.mass),method)
            patch,flux=patch_errors(F,states,exact,amplitude*coefficient)
            audit,arrays=closed_operator_audit(F,g,history.mass,cfg['probe_time'])
            row['methods'][method]={'linear_patch':patch,'closed_operator':audit}
            saved[method+'_flux_matrix']=F;saved[method+'_face_flux']=flux;saved[method+'_conductance']=weights
            for name,value in arrays.items():saved[method+'_'+name]=value
        F,gradient,geometry_checks=scalar_reconstruction_reference(g,coefficient,centers)
        patch,flux=patch_errors(F,states,exact,amplitude*coefficient)
        audit,arrays=closed_operator_audit(F,g,history.mass,cfg['probe_time'])
        row['methods']['linear_reconstruction_reference']={'linear_patch':patch,'closed_operator':audit,**geometry_checks}
        saved['reconstruction_gradient_operator']=gradient;saved['reconstruction_flux_matrix']=F;saved['reconstruction_face_flux']=flux
        for name,value in arrays.items():saved['reconstruction_'+name]=value
        np.savez_compressed(output/f'patch-t{t:g}.npz',**saved)
        rows.append(row)
        print('frozen time',t,'angle',row['maximum_nonorthogonality_degrees'],
              'two-point flux error',row['methods']['normal_distance']['linear_patch']['maximum_joint_basis_relative_flux_error'],
              'reference min Q offdiagonal',audit['minimum_offdiagonal_generator'],flush=True)
    result={'frozen_snapshots':rows,'physical_acceptance':False,'variable_geometry_diffusion_solver_run':False,
            'decision':'two-point face exchange fails on sheared cells; scalar reconstruction is an unaccepted reference requiring positivity work',
            'limits':['one instantaneous linear patch, not a transient convergence test','oracle mass centroids and face coefficients used to isolate geometry',
                      'failure applies to two-point face conductance, not every possible positive wider-stencil operator',
                      'reference exactness does not establish monotonicity or ORC admissibility']}
    (output/'results.json').write_text(json.dumps(result,indent=2)+'\n')
    (output/'environment.json').write_text(json.dumps({'python':platform.python_version(),'numpy':np.__version__,'scipy':scipy.__version__,'BLAS_threads_command':4},indent=2)+'\n')
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('source');p.add_argument('--output',required=True)
    a=p.parse_args();main(a.source,a.output)
