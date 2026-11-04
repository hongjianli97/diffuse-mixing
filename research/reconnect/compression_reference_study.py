"""One prescribed flow reference, fixed spatial and necessary temporal controls."""
from pathlib import Path
import argparse,hashlib,json,platform,resource
import numpy as np
import scipy
from chain_study import verify
from compression_reference import reference_path


def metrics(m,c,c0):
    mu=float(m@c/m.sum());mu0=float(m@c0/m.sum());E=float(m@((c-mu)**2));E0=float(m@((c0-mu0)**2))
    return {'mean':mu,'variance':E,'I':E/E0,'H':1-E/E0}


def main(output):
    root=Path(__file__).resolve().parents[2];source=root/'runs/reconnect-kernel2d-compression-midpoint';verify(root,source)
    protocol=json.loads((source/'stage3-status.json').read_text())['next_compression_reference_predeclared']
    if protocol['reference_spatial_levels']!=[512,1024] or protocol['time_interval_s']!=[0.,.05] or protocol['physical_D_m2_per_s']!=.02 or protocol['new_point_graphs']!=0:
        raise RuntimeError('predeclared single-flow reference changed')
    out=Path(output).resolve();out.mkdir(parents=True,exist_ok=False)
    # Fixed before the four solves; these are uncertainty controls, not a search.
    cases=[(512,256),(1024,256),(1024,512),(1024,1024)]
    cfg={'study':'independent time-dependent compression PDE reference only','source':str(source),'protocol':protocol,
        'protocol_source_sha256':hashlib.sha256((source/'stage3-status.json').read_bytes()).hexdigest(),
        'reference_method':'periodic conservative material FV + nonautonomous midpoint Crank-Nicolson for B=exp(D*(2*pi)^2*t)*A; transverse decay exact',
        'reference_cases':[{'columns':M,'time_steps':steps} for M,steps in cases],
        'reference_snapshot_times_s':[0.,.025,.05],'time_cases_fixed_before_solves':True,
        'reference_inputs':'analytic material rho0/J^2 face coefficient; no candidate kernel/weights/results',
        'observation':'nested fine mass/component integrals divided by original saved masses; no state or mass normalization',
        'new_candidate_graphs':0,'new_candidate_transient_runs':0,'full_P_constructed':False,
        'candidate_backend_modified':False,'fluid_orc_enabled':False,'stage3_physical_acceptance':'incomplete'}
    (out/'config.json').write_text(json.dumps(cfg,indent=2)+'\n')
    with np.load(source/'midpoint-observations.npz') as z:
        m=z['mass'];c0=z['c0'];ids=z['ids'];labels=z['material_label_centers']
    saved={'mass':m,'c0':c0,'ids':ids,'material_label_centers':labels};answers={};audits={};N=128
    expected=(np.arange(N)+.5)/N
    np.testing.assert_array_equal(labels,np.stack(np.meshgrid(expected,expected,indexing='xy'),axis=-1).reshape(-1,2))
    mean=float(m@c0/m.sum());fluct_norm=np.sqrt(m@((c0-mean)**2));relative=lambda v:float(np.sqrt(m@(v*v))/fluct_norm)
    for M,steps in cases:
        name=f'n{M}-k{steps}';a,audit=reference_path(M,steps,N,m)
        audit['initial_projected_state_vs_original_c0_max_difference']=float(np.max(abs(a['projected_states'][0]-c0)))
        audit['snapshot_metrics']=[metrics(m,c,c0) for c in a['projected_states']]
        answers[name]=a;audits[name]=audit
        for key,value in a.items():saved[name+'_'+key]=value
    space=answers['n512-k256']['projected_states']-answers['n1024-k256']['projected_states']
    time_coarse=answers['n1024-k256']['projected_states']-answers['n1024-k512']['projected_states']
    time_fine=answers['n1024-k512']['projected_states']-answers['n1024-k1024']['projected_states']
    controls={'state_differences_relative_to_original_mass_weighted_fluctuation':{
        'space_512_vs_1024_at_256_steps':[relative(v) for v in space],
        'time_256_vs_512_at_1024_columns':[relative(v) for v in time_coarse],
        'time_512_vs_1024_at_1024_columns':[relative(v) for v in time_fine]},
        'final_time_difference_ratio':relative(time_coarse[-1])/relative(time_fine[-1]),
        'final_time_empirical_order':float(np.log2(relative(time_coarse[-1])/relative(time_fine[-1]))),
        'final_raw_space_plus_fine_time_difference_indicator':relative(space[-1])+relative(time_fine[-1]),
        'indicator_is_a_rigorous_error_bound':False,'two_spatial_levels_prove_convergence_order':False,
        'Richardson_extrapolated_state_used':False,'best_raw_reference':'n1024-k1024'}
    best=answers['n1024-k1024']['projected_states'];best_metrics=[metrics(m,c,c0) for c in best]
    Hspace=abs(audits['n512-k256']['snapshot_metrics'][-1]['H']-audits['n1024-k256']['snapshot_metrics'][-1]['H'])
    Htime=abs(audits['n1024-k512']['snapshot_metrics'][-1]['H']-best_metrics[-1]['H'])
    result={'reference_runs':audits,'uncertainty_controls':controls,'best_raw_reference_metrics':best_metrics,
        'H_space_difference':Hspace,'H_fine_time_difference':Htime,
        'maximum_initial_projection_vs_original_c0_difference':max(v['initial_projected_state_vs_original_c0_max_difference'] for v in audits.values()),
        'same_saved_mass_c0_ids':True,'reference_flow_time_interval_s':[0.,.05],
        'new_candidate_graphs':0,'new_candidate_transient_runs':0,'new_reference_runs':4,'fluid_orc_enabled':False,'stage3_physical_acceptance':'incomplete',
        'peak_process_RSS_MiB_linux':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
        'limits':['empirical reference differences, not rigorous continuum error bounds','reference finite-volume means retain their own second-order spatial closure',
                  'no actual point-kernel compression propagation or order-error measurement','no general flow/shock/wall/ledger/DNS acceptance']}
    saved.update({'snapshot_time':np.array([0.,.025,.05]),'best_projected_states':best,'space_difference':space,'time_coarse_difference':time_coarse,'time_fine_difference':time_fine})
    np.savez_compressed(out/'compression-reference.npz',**saved)
    for name,value in [('results.json',result),('environment.json',{'python':platform.python_version(),'numpy':np.__version__,'scipy':scipy.__version__,'threads':4})]:
        (out/name).write_text(json.dumps(value,indent=2)+'\n')
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args();main(a.output)
