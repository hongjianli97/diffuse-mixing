"""One nonlinear no-flux frozen probe of the unmodified diagnostic graph."""
from pathlib import Path
import argparse,json,platform
import numpy as np
import scipy
from becker import Becker
from kernel_joint_study import verify
from noflux_probe import NoFluxProbe,mass_coordinate_average,volume_reference
from frozen_physical_faces import frozen_faces,physical_scalar_face_flux


def weighted_error(mass,approx,reference,mask):
    m=mass[mask];r=reference[mask];e=approx[mask]-r;den=np.sqrt(np.dot(m,r*r))
    return {'relative_mass_weighted_rate_error':float(np.sqrt(np.dot(m,e*e))/den) if den!=0 else None,
        'mass_weighted_RMS_rate_error_per_s':float(np.sqrt(np.dot(m,e*e)/m.sum())),
        'maximum_absolute_rate_error_per_s':float(np.max(abs(e)))}


def main(source,output):
    root=Path(__file__).resolve().parents[2];source=Path(source).resolve();output=Path(output).resolve();verify(root,source)
    old=json.loads((source/'config.json').read_text());p=old['physical_parameters'];t=old['time'];D=p['D']
    with np.load(source/'moment-budget.npz') as z:
        mass=z['mass'];original_c0=z['c0'];quads=z['quads'];centers=z['centroids'];columns=z['columns'];rows=z['rows']
        Q=z['diagnostic_Q'];g=z['raw_conductances'];pairs=z['pairs'];boundary=z['boundary_mask']
    with np.load(root/'runs/rebuild-stage3-half-transverse/half-transverse-cone.npz') as z:mu=z['mass_edges']
    flow=Becker(delta=p['delta'],mach=p['Mach'],gamma=p['gamma'],rho1=p['rho1'],p1=p['p1'],R=p['R'])
    xi=flow.x_mu(mu,t);eta=np.linspace(-.5,.5,5);probe=NoFluxProbe((float(xi[0]),float(xi[-1])),time=t,T=p['T'])
    topology=frozen_faces(quads,columns,rows)
    output.mkdir(parents=True,exist_ok=False)
    cfg={'study':'one independent nonlinear frozen manufactured state with true no-flux outer boundary',
        'source':str(source),'physical_parameters':p,'time':t,'field':'c=.5+.25*b((xi-xi_L)/(xi_R-xi_L))*b((eta+.5)/1), b(s)=16*s^2*(1-s)^2',
        'xi_bounds':probe.xi_bounds,'eta_bounds':probe.eta_bounds,'scalar_range':[.5,.75],
        'mass_average':'primary mass-coordinate product integral; independent physical density quadrature divided by unchanged fixed m',
        'reference_rate':'negative real shared-face flux divergence / m; independent volume integral of div(rho D grad c) / m',
        'controls':'32/64 point quadrature, fixed before comparison; no time step',
        'graph':'unchanged raw minimax diagnostic Q, weights and connectivity; not a selected discretization',
        'error_definition':'sqrt(sum m*(Q*cbar-r_ref)^2)/sqrt(sum m*r_ref^2), rate comparison only',
        'graph_physical_face_flux_defined':False,'no_arbitrary_graph_to_face_allocation':True,
        'original_c0_replaced':False,'physical_acceptance':False,'fluid_orc_enabled':False,'new_transient_run':False,
        'physical_targets_weights_or_thresholds_changed':False,'field_amplitude_or_shape_search':False}
    (output/'config.json').write_text(json.dumps(cfg,indent=2)+'\n')
    c32=mass_coordinate_average(probe,flow,mu,eta,columns,rows,32);c64=mass_coordinate_average(probe,flow,mu,eta,columns,rows,64)
    v32=volume_reference(probe,flow,quads,mass,D,32);v64=volume_reference(probe,flow,quads,mass,D,64)
    q32=physical_scalar_face_flux(topology,probe,flow,D,32);q64=physical_scalar_face_flux(topology,probe,flow,D,64)
    r32=-(topology['incidence']@q32)/mass;r64=-(topology['incidence']@q64)/mass;graph=Q@c64
    face_boundary=topology['boundary_mask'];sign=topology['incidence'].sum(axis=0)
    isolated=np.flatnonzero(np.all(Q==0,axis=1));error=graph-r64
    result={'all_cells':weighted_error(mass,graph,r64,np.ones(len(mass),dtype=bool)),
        'interior_cells':weighted_error(mass,graph,r64,~boundary),'boundary_cells':weighted_error(mass,graph,r64,boundary),
        'reference_controls':{'mass_coordinate_mean_32_64_max_abs_difference':float(np.max(abs(c64-c32))),
            'mass_coordinate_vs_density_mean_max_abs_difference':float(np.max(abs(c64-v64['average']))),
            'density_integral_mass_relative_error_max':float(np.max(abs(v64['actual_mass']-mass)/mass)),
            'face_rate_32_64_max_abs_difference_per_s':float(np.max(abs(r64-r32))),
            'volume_rate_32_64_max_abs_difference_per_s':float(np.max(abs(v64['rate']-v32['rate']))),
            'face_vs_volume_rate_max_abs_difference_per_s':float(np.max(abs(r64-v64['rate']))),
            'graph_rate_due_to_mean_32_64_max_abs_difference_per_s':float(np.max(abs(Q@(c64-c32))))},
        'physical_faces':len(q64),'interior_faces':int(np.count_nonzero(~face_boundary)),'outer_faces':int(face_boundary.sum()),
        'maximum_raw_outer_face_flux_kg_per_s':float(np.max(abs(q64[face_boundary]))),
        'physical_global_rate_mass_balance_kg_per_s':float(np.dot(mass,r64)),
        'physical_rate_vs_boundary_total_residual_kg_per_s':float(np.dot(mass,r64)+np.dot(sign[face_boundary],q64[face_boundary])),
        'graph_global_rate_mass_balance_kg_per_s':float(np.dot(mass,graph)),
        'scalar_mass_average_range':[float(c64.min()),float(c64.max())],
        'mass_average_vs_mass_centroid_point_value_max_abs_difference':float(np.max(abs(c64-probe.world(centers)[0]))),
        'isolated_cells':[{'child':int(i),'column':int(columns[i]),'row':int(rows[i]),'mass_average':float(c64[i]),
            'physical_rate_per_s':float(r64[i]),'graph_rate_per_s':float(graph[i])} for i in isolated],
        'isolated_cells_fraction_of_squared_mass_weighted_rate_error':float(np.dot(mass[isolated],error[isolated]**2)/np.dot(mass,error**2)),
        'maximum_error_child':int(np.argmax(abs(error))),
        'interior_face_cycle_space_dimension':topology['interior_flux_cycle_dimension'],
        'graph_physical_face_flux_defined':False,'facewise_graph_error_claimed':False,
        'physical_acceptance':False,'fluid_orc_enabled':False,'new_transient_run':False,
        'scope':'one frozen nonlinear field of the saved nonunique graph witness; not an optimized-method convergence result or the original c0 terminal error'}
    saved={'mass':mass,'original_c0':original_c0,'quads':quads,'mass_centroids':centers,'columns':columns,'rows':rows,'boundary_cell_mask':boundary,
        'unchanged_Q':Q,'unchanged_conductances':g,'unchanged_pairs':pairs,'mass_edges':mu,'eta_edges':eta,
        'probe_average_order32':c32,'probe_average_order64':c64,'density_average_order32':v32['average'],'density_average_order64':v64['average'],
        'density_integral_mass_order64':v64['actual_mass'],'face_flux_order32':q32,'face_flux_order64':q64,
        'face_rate_order32':r32,'face_rate_order64':r64,'volume_rate_order32':v32['rate'],'volume_rate_order64':v64['rate'],
        'graph_rate':graph,'rate_error':error}
    for key,value in topology.items():saved['topology_'+key]=value
    np.savez_compressed(output/'noflux-probe.npz',**saved)
    (output/'results.json').write_text(json.dumps(result,indent=2)+'\n')
    (output/'environment.json').write_text(json.dumps({'python':platform.python_version(),'numpy':np.__version__,'scipy':scipy.__version__,'BLAS_threads_command':4},indent=2)+'\n')
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('source');p.add_argument('--output',required=True);a=p.parse_args();main(a.source,a.output)
