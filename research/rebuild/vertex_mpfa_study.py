"""One candidate on the unchanged 64-cell mesh; no transient parameter scan."""
from pathlib import Path
import argparse,json,platform
import numpy as np
import scipy
from scipy.special import roots_legendre
from becker import Becker
from becker_polygon_benchmark import density_velocity
from kernel_joint_study import verify
from material_polygon_history import MaterialPolygonHistory
from nonorthogonal_flux import internal_face_geometry,closed_operator_audit
from nonorthogonal_patch_study import patch_errors
from vertex_mpfa import build_vertex_mpfa,internal_flux


def oracle_half_coefficients(history,flow,positions,t,D,T=.1,order=32):
    a,b=history.edge_vertex_indices.T;v=positions[b]-positions[a];length=np.linalg.norm(v,axis=1)
    z,w=roots_legendre(order);s=np.array([0.,.5])[:,None]+.5*((z+1)/2)[None,:]
    p=positions[a,None,None,:]+s[None,:,:,None]*v[:,None,None,:]
    rho,_=density_velocity(flow,p[None,...],np.array([t]),T)
    return D*.5*length[:,None]*np.sum(rho[0]*(w/2),axis=-1)


def prescribed_linear_boundary(operator,half_coefficients,amplitude=.1):
    r=operator['raw'];e=r['boundary_half_edges'];v=r['boundary_half_endpoints']
    return -amplitude*half_coefficients[e,v,None]*r['boundary_outward_sign'][:,None]*r['canonical_normal'][e]


def main(source,output):
    root=Path(__file__).resolve().parents[2];source=Path(source).resolve();output=Path(output).resolve();verify(root,source)
    cfgold=json.loads((source/'config.json').read_text());p=cfgold['physical_parameters'];T=p['T'];D=p['D']
    polygon=root/'runs'/Path(cfgold['source']).name;verify(root,polygon)
    history=MaterialPolygonHistory.load(polygon/'polygon-record-32.npz')
    flow=Becker(delta=p['delta'],mach=p['Mach'],gamma=p['gamma'],rho1=p['rho1'],p1=p['p1'],R=p['R'])
    output.mkdir(parents=True,exist_ok=False)
    cfg={'study':'one vertex-local multipoint candidate on unchanged 64-cell mass-average input',
      'source':str(source),'polygon_source':str(polygon),'physical_parameters':p,'times':[0.,T/2,T],
      'candidate':'one vertex interaction, two local scalar gradient components per incident cell, half-face normal-flux continuity',
      'internal_continuity_point':'mass-centroid line intersection with edge; outside open segment rejected, no fallback',
      'boundary_continuity_point':'edge midpoint, fixed before computation',
      'auxiliary_trace_sharing':'one trace per (edge, vertex); separate vertex interactions, not a globally single edge trace',
      'state':'original fixed masses and mass averages; saved oracle mass centroids, no point-value reinterpretation',
      'face_coefficients':'independent density integrals over actual edge halves, 16/32 Gauss control',
      'patch_fields':['.5+.1*x','.5+.1*y'],'gradient_amplitude_per_meter':.1,
      'patch_boundary':'prescribed exact outward Neumann flux on all boundary half-faces',
      'closed_operator_boundary':'homogeneous Neumann data; audited separately, same elimination matrices',
      'probe_time':1e-4,'physical_acceptance':False,'variable_geometry_diffusion_solver_run':False,'fluid_orc_enabled':False,
      'coefficient_clipping_or_flux_fitting':False,'candidate_or_continuity_point_parameter_scan':False}
    (output/'config.json').write_text(json.dumps(cfg,indent=2)+'\n');rows=[]
    for t in cfg['times']:
        with np.load(source/f'patch-t{t:g}.npz') as z:
            positions=z['positions'];centers=z['mass_centroids'];states=z['patch_states'];original_c0=z['original_c0'];mass=z['mass']
            coefficient=z['integrated_face_coefficient'];exact=z['exact_internal_face_flux']
        np.testing.assert_array_equal(mass,history.mass);np.testing.assert_array_equal(original_c0,history.c0)
        k16=oracle_half_coefficients(history,flow,positions,t,D,T,16)
        k=oracle_half_coefficients(history,flow,positions,t,D,T,32)
        geometry=internal_face_geometry(history,positions,centers);operator=build_vertex_mpfa(history,positions,centers,k)
        h=prescribed_linear_boundary(operator,k);flux=internal_flux(operator,states,h)
        patch,_=patch_errors(operator['F'],states,exact-operator['B']@h,.1*coefficient)
        disabled,_=patch_errors(operator['F'],states,exact,.1*coefficient)
        audit,arrays=closed_operator_audit(operator['F'],geometry,mass,cfg['probe_time'])
        raw=operator['raw'];be=raw['boundary_half_edges'];boundary_cell=np.argmax(history.incidence[:,be]!=0,axis=0)
        exterior=np.zeros_like(states);np.add.at(exterior,boundary_cell,h)
        derivative=-(geometry['incidence']@flux+exterior)/mass[:,None]
        linear_boundary_balance=float(np.max(abs(mass@derivative+h.sum(axis=0))))
        row={'time':t,'linear_patch_with_matching_Neumann_data':patch,
             'linear_patch_with_boundary_data_disabled_incompatible_control':disabled,
             'half_coefficient_16_32_max_relative_difference':float(np.max(abs(k-k16)/k)),
             'sum_half_vs_previous_full_coefficient_max_relative_difference':float(np.max(abs(k[geometry['edge_indices']].sum(axis=1)-coefficient)/coefficient)),
             'prescribed_boundary_global_balance_residual':linear_boundary_balance,
             'closed_operator':audit,'local_elimination':operator['diagnostics']}
        np.savez_compressed(output/f'mpfa-t{t:g}.npz',positions=positions,mass=mass,original_c0=original_c0,mass_centroids=centers,
             patch_states=states,half_coefficients=k,half_coefficients_order16=k16,exact_internal_face_flux=exact,
             patch_boundary_outward_half_flux=h,patch_internal_flux=flux,linear_probe_derivative=derivative,
             F=operator['F'],B=operator['B'],**raw,**arrays)
        rows.append(row);print('time',t,'linear patch',patch['maximum_joint_basis_relative_flux_error'],
             'closed min Q',audit['minimum_offdiagonal_generator'],'raw minimum propagator',audit['minimum_raw_propagator_entry'],flush=True)
    result={'frozen_snapshots':rows,'physical_acceptance':False,'fluid_orc_enabled':False,
       'candidate_count':1,'variable_geometry_diffusion_solver_run':False,
       'decision':'candidate remains experimental; linear consistency with correct Neumann data and homogeneous-boundary positivity are separate gates'}
    (output/'results.json').write_text(json.dumps(result,indent=2)+'\n')
    (output/'environment.json').write_text(json.dumps({'python':platform.python_version(),'numpy':np.__version__,'scipy':scipy.__version__,'BLAS_threads_command':4},indent=2)+'\n')
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('source');p.add_argument('--output',required=True);a=p.parse_args();main(a.source,a.output)
