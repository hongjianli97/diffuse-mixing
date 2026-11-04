"""One predeclared actual shear scalar case, 4/8 temporal intervals."""
from pathlib import Path
import argparse,hashlib,json,platform,resource
import numpy as np
import scipy
from chain_study import verify
from positive_kernel_2d import build_periodic_exchange
from shear_probe import make_shear_samples
from ordered_compression import interval_grid,frozen_exponential_step,dissipation
from compression_reference_study import metrics
from sparse_exchange_audit import audit_laplacian,sparse_laplacian,sparse_max_abs
from trajectory_chain import ALGEBRA_TOL


def main(output):
    root=Path(__file__).resolve().parents[2];source=root/'runs/reconnect-shear-reference';verify(root,source)
    protocol=json.loads((source/'stage3-status.json').read_text())['next_candidate_shear_transient_predeclared']
    if protocol['ordered_time_interval_counts']!=[4,8] or (protocol['N_per_direction'],protocol['h_m'],protocol['D_m2_per_s'],protocol['shear_end'])!=(128,.04,.02,.25) or protocol['time_interval_s']!=[0.,.05] or protocol['fluid_orc_enabled']:
        raise RuntimeError('predeclared single-shear temporal control changed')
    if protocol['expected_new_midpoint_graphs']!=12 or protocol['observation_times_s']!=[0.,.025,.05] or protocol['epsilon_initial_density']!=.25 or protocol['full_P_constructed']:
        raise RuntimeError('predeclared graph count/reference observations changed')
    out=Path(output).resolve();out.mkdir(parents=True,exist_ok=False)
    cfg={'study':'one actual shear scalar flow, chronological frozen-midpoint exponential actions with 4/8 intervals',
        'source':str(source),'protocol':protocol,'protocol_source_sha256':hashlib.sha256((source/'stage3-status.json').read_bytes()).hexdigest(),
        'candidate_backend_unchanged':True,'full_P_or_generator_matrix_products':False,
        'density_source':'saved given rho0 along fixed identities; area-preserving prescribed shear producer',
        'candidate_geometry_inputs':'positions, fixed mass, oracle point density only; no J/F/C/material faces',
        'reference':'saved raw M127-k256 full 2D material PDE projected to original N128 mass averages; not rerun',
        'error_normalization':'sqrt(sum m*(candidate-reference)^2)/sqrt(sum m*(original_c0-mean_m)^2)',
        'graph_storage':'raw CSR per sampled midpoint plus complete point inputs and edge diagnostics; no graph repair',
        'exponential_controls':'sampled endpoint vs direct action vs two half actions on the same frozen graph',
        'dissipation_control':'raw c^T L c at interval start/exponential midpoint/end; Simpson integral is an empirical quadrature check',
        'parameter_or_mesh_scan':False,'clipping_fitting_or_normalization':False,'fluid_orc_enabled':False,'stage3_physical_acceptance':'incomplete'}
    (out/'config.json').write_text(json.dumps(cfg,indent=2)+'\n')
    state_path=root/protocol['saved_input_source']
    with np.load(state_path) as z:
        m=z['mass'];c0=z['c0'];ids=z['ids'];labels=z['material_label_centers'];rho0=z['rho0_point']
    with np.load(source/'shear-reference-observations.npz') as z:
        for name,value in [('mass',m),('c0',c0),('ids',ids),('material_label_centers',labels)]:
            np.testing.assert_array_equal(z['N128_'+name],value)
        ref=z['N128_best_reference_states'];reference_times=z['snapshot_time']
        ref_space=z['N128_space_difference'];ref_time=z['N128_time_difference']
    mu0=float(m@c0/m.sum());variance0=float(m@((c0-mu0)**2));sigma0=np.sqrt(variance0)
    relative=lambda v:float(np.sqrt(m@(v*v))/sigma0)
    saved={'mass':m,'c0':c0,'ids':ids,'material_label_centers':labels,'rho0_point':rho0,
        'reference_times':reference_times,'reference_states':ref,'reference_space_difference':ref_space,'reference_time_difference':ref_time}
    reference_status=json.loads((source/'stage3-status.json').read_text())
    cases={};snapshots={};graph_count=0
    for K in protocol['ordered_time_interval_counts']:
        boundaries,midpoints,times=interval_grid(K,.05)
        data,map_arrays,map_audit=make_shear_samples(labels,ids,m,c0,rho0,times)
        state=c0.copy();states=[state.copy()];half_states=[];direct_states=[];two_half_states=[];steps=[]
        for step,t in enumerate(midpoints):
            frame=2*step+1;dt=.05/K
            L,a,diag=build_periodic_exchange(data,frame,.02,.04,[1.,1.]);graph_audit=audit_laplacian(m,L)
            if not graph_audit['graph_algebra_eligible']:raise RuntimeError('raw graph algebra failed; no repair')
            independent=sparse_laplacian(len(m),a['pairs'],a['conductances']);shared_defect=sparse_max_abs(independent-L);del independent
            following,half,check=frozen_exponential_step(m,L,state,dt)
            next_metrics=metrics(m,following,c0);before_metrics=metrics(m,state,c0)
            chi=[dissipation(L,c) for c in [state,half,following]]
            energy_loss=.5*(before_metrics['variance']-next_metrics['variance']);simpson=dt*(chi[0]+4*chi[1]+chi[2])/6
            i,j=a['pairs'].T;pair_chi=float(a['conductances']@((state[j]-state[i])**2))
            row={'step':step,'left_time_s':float(boundaries[step]),'right_time_s':float(boundaries[step+1]),'frozen_midpoint_time_s':float(t),
                'operator_diagnostics':diag,'sparse_graph_audit':graph_audit,'shared_edge_assembly_max_difference':shared_defect,
                'start_metrics':before_metrics,'end_metrics':next_metrics,
                'minimum_state':float(following.min()),'maximum_state':float(following.max()),
                'raw_component_mass_change_kg':float(m@(following-state)),
                'chi_start_mid_end_kg_per_s':chi,'pair_chi_start_vs_CSR_absolute_difference':abs(pair_chi-chi[0]),
                'half_variance_energy_loss_kg':energy_loss,'Simpson_chi_integral_kg':simpson,'Simpson_energy_balance_difference_kg':energy_loss-simpson,
                'direct_vs_sampled_relative_difference':relative(check['direct_minus_path']),
                'two_half_vs_sampled_relative_difference':relative(check['two_half_minus_path'])}
            if abs(float(m@(following-c0)))/float(m@c0)>ALGEBRA_TOL or following.min()<c0.min()-ALGEBRA_TOL or following.max()>c0.max()+ALGEBRA_TOL or energy_loss < -ALGEBRA_TOL:
                raise RuntimeError('raw scalar fails original algebra/bounds check; no correction')
            graph_path=out/f'graph-k{K}-step{step:02d}.npz'
            np.savez_compressed(graph_path,L_data=L.data,L_indices=L.indices,L_indptr=L.indptr,L_shape=np.array(L.shape),
                frame=np.array(frame),midpoint_time=np.array(t))
            row['saved_graph']=graph_path.name;row['saved_graph_sha256']=hashlib.sha256(graph_path.read_bytes()).hexdigest()
            states.append(following.copy());half_states.append(half);direct_states.append(check['direct_endpoint']);two_half_states.append(check['two_half_endpoint'])
            steps.append(row);state=following;graph_count+=1
            print(json.dumps({'intervals':K,'step_completed':step+1,'of':K,'midpoint_s':float(t),'H':next_metrics['H']},separators=(',',':')),flush=True)
            del L,a
        states=np.array(states);observed=states[[0,K//2,K]];snapshots[K]=observed
        rows=[]
        for index,t in enumerate(reference_times):
            actual=observed[index];candidate_metrics=metrics(m,actual,c0);ref_metrics=metrics(m,ref[index],c0)
            rows.append({'time_s':float(t),'relative_concentration_error_initial_fluctuation_norm':relative(actual-ref[index]),
                'candidate_metrics':candidate_metrics,'reference_metrics':ref_metrics,'H_candidate_minus_reference':candidate_metrics['H']-ref_metrics['H'],
                'candidate_min':float(actual.min()),'candidate_max':float(actual.max()),'raw_component_mass_change_from_c0_kg':float(m@(actual-c0))})
        cases[str(K)]={'map_audit':map_audit,'steps':steps,'observations':rows,
            'maximum_direct_exponential_control_relative_difference':max(v['direct_vs_sampled_relative_difference'] for v in steps),
            'maximum_two_half_exponential_control_relative_difference':max(v['two_half_vs_sampled_relative_difference'] for v in steps),
            'maximum_absolute_Simpson_energy_balance_difference_kg':max(abs(v['Simpson_energy_balance_difference_kg']) for v in steps),
            'total_half_variance_energy_loss_kg':sum(v['half_variance_energy_loss_kg'] for v in steps)}
        for name,value in [('time',times),('positions',data.positions),('rho',data.rho),('boundary_time',boundaries),('midpoint_time',midpoints),
            ('states',states),('exponential_half_states',np.array(half_states)),('direct_endpoints',np.array(direct_states)),('two_half_endpoints',np.array(two_half_states))]:
            saved[f'k{K}_{name}']=value
    difference=snapshots[4]-snapshots[8]
    temporal={'relative_state_difference_initial_fluctuation_norm':[relative(v) for v in difference],
        'H_4_minus_8':[cases['4']['observations'][i]['candidate_metrics']['H']-cases['8']['observations'][i]['candidate_metrics']['H'] for i in range(3)],
        'reference_empirical_state_difference_indicator_final':reference_status['reference_uncertainty_controls']['128']['final_raw_space_plus_time_difference_indicator'],
        'reference_H_space_difference':reference_status['reference_uncertainty_controls']['128']['H_final_space_difference'],'reference_H_fine_time_difference':reference_status['reference_uncertainty_controls']['128']['H_final_time_difference'],
        'reference_space_differences_relative_to_initial_fluctuation':[relative(v) for v in ref_space],
        'reference_time_differences_relative_to_initial_fluctuation':[relative(v) for v in ref_time],
        'time_difference_is_a_rigorous_error_bound':False,'two_time_levels_prove_convergence_order':False,'Richardson_extrapolated_state_used':False}
    result={'cases':cases,'time_and_reference_controls':temporal,'physical_cases':1,'new_midpoint_graphs':graph_count,
        'new_time_interval_counts':[4,8],'parameter_or_mesh_scan':False,'same_saved_mass_c0_ids':True,
        'reference_rerun':False,'full_P_constructed':False,'fluid_orc_enabled':False,'stage3_physical_acceptance':'incomplete',
        'peak_process_RSS_MiB_linux':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
        'limits':['one smooth full 2D shear trajectory, one N/h, oracle density','4/8 temporal difference is empirical, not a validated asymptotic order',
                  'finite-width/quadrature/mass-average spatial closure remains combined','Simpson energy integral is approximate, not an exact balance proof',
                  'no averaged or reversed flow propagation, shock/wall/ledger/nonplanar/DNS or ORC acceptance']}
    saved['time_control_difference']=difference
    np.savez_compressed(out/'ordered-shear.npz',**saved)
    for name,value in [('results.json',result),('environment.json',{'python':platform.python_version(),'numpy':np.__version__,'scipy':scipy.__version__,'threads':4})]:
        (out/name).write_text(json.dumps(value,indent=2)+'\n')
    print(json.dumps({'physical_cases':1,'midpoint_graphs':graph_count,'observations':{k:v['observations'] for k,v in cases.items()},
        'time_and_reference_controls':temporal,'peak_process_RSS_MiB_linux':result['peak_process_RSS_MiB_linux'],'fluid_orc_enabled':False},indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args();main(a.output)
