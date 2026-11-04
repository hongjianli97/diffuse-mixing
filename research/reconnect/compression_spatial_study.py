"""One predeclared N256/h.03 actual-flow refinement, common material means."""
from pathlib import Path
import argparse,hashlib,json,platform,resource
import numpy as np
import scipy
from chain_study import verify
from positive_kernel_2d import build_periodic_exchange
from compression_probe import make_compression_samples
from compression_spatial_observation import fine_inputs,reuse_reference,restrict_path
from compression_reference_study import metrics
from ordered_compression import interval_grid,frozen_exponential_step,dissipation
from sparse_exchange_audit import audit_laplacian,sparse_laplacian,sparse_max_abs
from trajectory_chain import ALGEBRA_TOL


def main(output):
    root=Path(__file__).resolve().parents[2];source=root/'runs/reconnect-ordered-compression';verify(root,source)
    protocol=json.loads((source/'stage3-status.json').read_text())['next_joint_spatial_control_predeclared']
    if protocol['one_new_level']!={'N_per_direction':256,'h_m':.03} or protocol['ordered_time_interval_counts']!=[4,8] or protocol['ORC']:
        raise RuntimeError('single predeclared spatial level changed')
    out=Path(output).resolve();out.mkdir(parents=True,exist_ok=False)
    reference_source=root/'runs/reconnect-compression-reference/compression-reference.npz'
    cfg={'study':'one N256/h.03 joint spatial level on same actual compression flow, common N128 material observations',
        'source':str(source),'protocol':protocol,'protocol_source_sha256':hashlib.sha256((source/'stage3-status.json').read_bytes()).hexdigest(),
        'reference_source_sha256':hashlib.sha256(reference_source.read_bytes()).hexdigest(),
        'identity_namespace':'new fixed material-square-N256 identities, explicit saved N128 parent map',
        'candidate_backend_and_exponential_step_unchanged':True,'candidate_inputs':'points, initial fixed mass, oracle point density; no J/F/C/true faces',
        'observation':'raw 2x2 child component sums divided by unchanged original saved parent masses',
        'reference':'saved 512/1024-column B snapshots only; no reference solves',
        'one_new_spatial_level':True,'ordered_time_interval_counts':[4,8],'new_midpoint_graphs_expected':12,
        'parameter_or_mesh_matrix_scan':False,'clipping_fitting_or_mass_state_normalization':False,
        'full_P_constructed':False,'fluid_orc_enabled':False,'stage3_physical_acceptance':'incomplete'}
    (out/'config.json').write_text(json.dumps(cfg,indent=2)+'\n')
    with np.load(source/'ordered-compression.npz') as z:
        parent={key:z[key] for key in ['mass','c0','ids','material_label_centers','reference_times','reference_states']}
        coarse={K:z[f'k{K}_states'][[0,K//2,K]] for K in [4,8]}
    f,mass_audit=fine_inputs(256,parent['ids'],parent['mass']);m=f['mass'];c0=f['c0'];labels=f['material_label_centers'];mp=parent['mass'];cp0=parent['c0']
    with np.load(reference_source) as z:references=reuse_reference(z,256,m)
    ref=references['n1024-k1024'];times_ref=parent['reference_times']
    mu0=m@c0/m.sum();mup0=mp@cp0/mp.sum();sigma=np.sqrt(m@((c0-mu0)**2));sigmap=np.sqrt(mp@((cp0-mup0)**2))
    relative=lambda v:float(np.sqrt(m@(v*v))/sigma)
    common_relative=lambda v:float(np.sqrt(mp@(v*v))/sigmap)
    saved={**f,**{'parent_'+key:value for key,value in parent.items()}}
    for key,value in references.items():saved['reference_'+key+'_states']=value
    projected_ref=restrict_path(ref,m,mp);reference_projection_difference=projected_ref-parent['reference_states']
    saved['reference_parent_projection_difference']=reference_projection_difference
    cases={};actual={};common={};graph_count=0
    for K in [4,8]:
        boundaries,midpoints,sample_times=interval_grid(K,.05)
        data,arrays,map_audit=make_compression_samples(labels,f['ids'],m,c0,f['rho0_point'],sample_times)
        state=c0.copy();states=[state.copy()];halves=[];direct=[];two_half=[];rows=[]
        for step,t in enumerate(midpoints):
            print(json.dumps({'intervals':K,'step':step,'phase':'building_fine_graph','midpoint_s':float(t)}),flush=True)
            L,a,diag=build_periodic_exchange(data,2*step+1,.02,.03,[1.,1.])
            audit=audit_laplacian(m,L)
            if not audit['graph_algebra_eligible']:raise RuntimeError('raw fine graph failed, no repair')
            independent=sparse_laplacian(len(m),a['pairs'],a['conductances']);shared=sparse_max_abs(independent-L);del independent
            following,half,checks=frozen_exponential_step(m,L,state,.05/K)
            before_metrics=metrics(m,state,c0);after_metrics=metrics(m,following,c0)
            chi=[dissipation(L,c) for c in [state,half,following]]
            loss=.5*(before_metrics['variance']-after_metrics['variance']);integral=(.05/K)*(chi[0]+4*chi[1]+chi[2])/6
            i,j=a['pairs'].T;pair_chi=float(a['conductances']@((state[j]-state[i])**2))
            if abs(float(m@(following-c0)))/float(m@c0)>ALGEBRA_TOL or following.min()<c0.min()-ALGEBRA_TOL or following.max()>c0.max()+ALGEBRA_TOL or loss< -ALGEBRA_TOL:
                raise RuntimeError('raw fine state violates original algebra/bounds criterion')
            path=out/f'graph-k{K}-step{step:02d}.npz'
            np.savez_compressed(path,L_data=L.data,L_indices=L.indices,L_indptr=L.indptr,L_shape=np.array(L.shape),frame=np.array(2*step+1),midpoint_time=np.array(t))
            row={'step':step,'left_time_s':float(boundaries[step]),'right_time_s':float(boundaries[step+1]),'frozen_midpoint_time_s':float(t),
                'operator_diagnostics':diag,'sparse_graph_audit':audit,'shared_edge_assembly_max_difference':shared,
                'start_metrics':before_metrics,'end_metrics':after_metrics,'minimum_state':float(following.min()),'maximum_state':float(following.max()),
                'raw_component_mass_change_kg':float(m@(following-state)),'chi_start_mid_end_kg_per_s':chi,
                'pair_chi_start_vs_CSR_absolute_difference':abs(pair_chi-chi[0]),'half_variance_energy_loss_kg':loss,
                'Simpson_energy_balance_difference_kg':loss-integral,
                'direct_vs_sampled_relative_difference':relative(checks['direct_minus_path']),
                'two_half_vs_sampled_relative_difference':relative(checks['two_half_minus_path']),
                'saved_graph':path.name,'saved_graph_sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
            rows.append(row);states.append(following.copy());halves.append(half);direct.append(checks['direct_endpoint']);two_half.append(checks['two_half_endpoint'])
            state=following;graph_count+=1
            print(json.dumps({'intervals':K,'completed':step+1,'of':K,'fine_H':after_metrics['H'],'graph_edges':diag['geometric_support_pairs']},separators=(',',':')),flush=True)
            del L,a,i,j
        states=np.array(states);parent_states=restrict_path(states,m,mp);actual[K]=states[[0,K//2,K]];common[K]=parent_states[[0,K//2,K]]
        observations=[]
        for index,t in enumerate(times_ref):
            own=actual[K][index];observed=common[K][index]
            own_metrics=metrics(m,own,c0);common_metrics=metrics(mp,observed,cp0);reference_metrics=metrics(mp,parent['reference_states'][index],cp0)
            observations.append({'time_s':float(t),'fine_own_cell_concentration_error':relative(own-ref[index]),
                'common_parent_concentration_error':common_relative(observed-parent['reference_states'][index]),
                'fine_own_metrics':own_metrics,'common_parent_metrics':common_metrics,'reference_common_metrics':reference_metrics,
                'common_H_minus_reference':common_metrics['H']-reference_metrics['H']})
        cases[str(K)]={'map_audit':map_audit,'steps':rows,'observations':observations}
        for key,value in [('time',sample_times),('positions',data.positions),('rho',data.rho),('boundary_time',boundaries),('midpoint_time',midpoints),
            ('states',states),('parent_states',parent_states),('exponential_half_states',np.array(halves)),('direct_endpoints',np.array(direct)),('two_half_endpoints',np.array(two_half))]:
            saved[f'k{K}_{key}']=value
    fine_time=common[4]-common[8];coarse_time=coarse[4]-coarse[8]
    space_reference=restrict_path(references['n512-k256']-references['n1024-k256'],m,mp)
    time_reference=restrict_path(references['n1024-k512']-references['n1024-k1024'],m,mp)
    comparisons=[]
    for index,t in enumerate(times_ref):
        old_error=common_relative(coarse[8][index]-parent['reference_states'][index]);new_error=common_relative(common[8][index]-parent['reference_states'][index])
        components={'coarse_4_vs_8':common_relative(coarse_time[index]),'fine_4_vs_8':common_relative(fine_time[index]),
            'reference_512_vs_1024':common_relative(space_reference[index]),'reference_fine_time':common_relative(time_reference[index]),
            'reference_projection_roundoff':common_relative(reference_projection_difference[index])}
        indicator=sum(components.values());improvement=old_error-new_error
        old_H=metrics(mp,coarse[8][index],cp0)['H'];new_H=metrics(mp,common[8][index],cp0)['H'];ref_H=metrics(mp,parent['reference_states'][index],cp0)['H']
        comparisons.append({'time_s':float(t),'coarse_error_common_cells':old_error,'fine_error_common_cells':new_error,
            'absolute_error_reduction':improvement,'relative_error_reduction_fraction':improvement/old_error if old_error else None,
            'empirical_difference_components':components,'combined_empirical_indicator':indicator,
            'improvement_over_indicator':improvement/indicator if indicator else None,'improvement_exceeds_empirical_indicator':bool(improvement>indicator),
            'coarse_H':old_H,'fine_H_common_cells':new_H,'reference_H':ref_H,
            'coarse_H_minus_reference':old_H-ref_H,'fine_H_minus_reference':new_H-ref_H})
    result={'cases':cases,'comparison_common_original_cells':comparisons,'parent_mass_roundoff_audit':mass_audit,
        'initial_parent_projection_vs_saved_c0_max_difference':float(np.max(abs(common[8][0]-cp0))),
        'fine_initial_reference_vs_exact_c0_max_difference':float(np.max(abs(ref[0]-c0))),
        'reference_projected_twice_vs_saved_parent_max_difference':float(np.max(abs(reference_projection_difference))),
        'reference_uncertainty_is_empirical_not_bound':True,'reference_rerun':False,'coarse_candidate_rerun':False,
        'fine_time_difference_own_cells':[relative(v) for v in actual[4]-actual[8]],
        'physical_cases':1,'one_new_joint_spatial_level':True,'new_midpoint_graphs':graph_count,'new_time_interval_counts':[4,8],
        'physical_acceptance_threshold_changed':False,'full_P_constructed':False,'fluid_orc_enabled':False,'stage3_physical_acceptance':'incomplete',
        'peak_process_RSS_MiB_linux':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
        'limits':['only two joint N/h levels on one smooth compression flow','empirical temporal/reference differences, not rigorous bounds',
                  'finite-width, quadrature and average-state errors remain combined','no shock/wall/ledger/nonplanar/DNS or ORC acceptance']}
    saved.update({'fine_time_difference_common':fine_time,'coarse_time_difference_common':coarse_time,'reference_space_difference_common':space_reference,'reference_time_difference_common':time_reference})
    for K in [4,8]:saved[f'coarse_k{K}_observations']=coarse[K]
    np.savez_compressed(out/'compression-spatial.npz',**saved)
    for name,value in [('results.json',result),('comparison.json',comparisons),('environment.json',{'python':platform.python_version(),'numpy':np.__version__,'scipy':scipy.__version__,'threads':4})]:
        (out/name).write_text(json.dumps(value,indent=2)+'\n')
    print(json.dumps({'comparison':comparisons,'mass_roundoff':mass_audit,'new_graphs':graph_count,'RSS_MiB':result['peak_process_RSS_MiB_linux'],'fluid_orc_enabled':False},indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args();main(a.output)
