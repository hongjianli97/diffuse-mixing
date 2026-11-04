"""One independent time-dependent 2D shear reference, three fixed controls."""
from pathlib import Path
import argparse,hashlib,json,platform,resource
import numpy as np
import scipy
from chain_study import verify
from shear_reference import ShearReferenceOperator,reference_path,project_mass_average
from shear_probe import shear_point_fields
from compression_reference_study import metrics
from compression_midpoint import weighted_norm
from variable_parent_observation import observe_parent


def main(output):
    root=Path(__file__).resolve().parents[2];source=root/'runs/reconnect-shear-spatial';verify(root,source)
    protocol=json.loads((source/'stage3-status.json').read_text())['next_shear_reference_predeclared']
    expected=[{'M_per_direction':63,'max_step_s':.05/128},{'M_per_direction':127,'max_step_s':.05/128},{'M_per_direction':127,'max_step_s':.05/256}]
    if protocol['reference_cases']!=expected or protocol['rtol']!=1e-10 or protocol['atol']!=1e-12 or protocol['new_candidate_graphs'] or protocol['new_candidate_transient_runs'] or protocol['fluid_orc_enabled']:
        raise RuntimeError('predeclared three-reference/no-candidate protocol changed')
    out=Path(output).resolve();out.mkdir(parents=True,exist_ok=False)
    input_paths=[root/rel for rel in protocol['saved_mass_c0_ids_sources']]
    cfg={'study':'independent full two-dimensional time-dependent shear reference only','source':str(source),'protocol':protocol,
        'protocol_source_sha256':hashlib.sha256((source/'stage3-status.json').read_bytes()).hexdigest(),
        'input_sources_sha256':{str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in input_paths},
        'method':'unfiltered odd-grid Fourier conservative divergence, real DOP853 states; no full spatial matrix',
        'projection':'Fourier cell integral of rho0*c divided by original saved m; real and imaginary parts retained',
        'new_reference_cases':3,'new_candidate_graphs':0,'new_candidate_transient_runs':0,
        'clipping_filtering_mass_correction_or_extrapolation':False,'fluid_orc_enabled':False,'stage3_physical_acceptance':'incomplete'}
    (out/'config.json').write_text(json.dumps(cfg,indent=2)+'\n')
    saved={};inputs={}
    for N,path in zip([128,256],input_paths):
        with np.load(path) as z:inputs[N]={key:z[key] for key in ['mass','c0','ids','material_label_centers']}
        axis=(np.arange(N)+.5)/N;labels=np.stack(np.meshgrid(axis,axis,indexing='xy'),axis=-1).reshape(-1,2)
        np.testing.assert_array_equal(inputs[N]['material_label_centers'],labels)
        for key,value in inputs[N].items():saved[f'N{N}_{key}']=value
    times=np.array(protocol['observation_times_s']);saved['snapshot_time']=times
    audits={};projected={};point_audits={}
    for index,case in enumerate(protocol['reference_cases']):
        M=case['M_per_direction'];max_step=case['max_step_s'];name=['M63-k128','M127-k128','M127-k256'][index]
        print(json.dumps({'reference_case':name,'phase':'integrating_full_material_PDE'}),flush=True)
        path,audit=reference_path(M,max_step,times,rtol=protocol['rtol'],atol=protocol['atol'])
        for key,value in path.items():saved[name+'_'+key]=value
        op=ShearReferenceOperator(M);cinit=op.initial_scalar();initial_rate_controls=[]
        for t in times:
            actual=op.rhs(float(t),cinit.ravel()).reshape(M,M)
            exact=shear_point_fields(op.a,op.b,.25,.25*t/.05,.02)[2]['rate']
            initial_rate_controls.append(float(np.max(abs(actual-exact))))
        point_audits[name]={'analytic_initial_profile_full_PDE_max_difference_per_s_at_times':initial_rate_controls}
        projected[name]={};audit['observations']={}
        for N in [128,256]:
            original=inputs[N];states=[];imaginary=[]
            for c in path['point_snapshots']:
                state,imag=project_mass_average(c,path['rho0_grid'],N,original['mass']);states.append(state);imaginary.append(imag)
            states=np.array(states);imaginary=np.array(imaginary);projected[name][N]=states
            saved[f'{name}_N{N}_states']=states;saved[f'{name}_N{N}_projection_imaginary']=imaginary
            audit['observations'][str(N)]={'initial_vs_original_c0_max_difference':float(np.max(abs(states[0]-original['c0']))),
                'maximum_imaginary_mass_average':float(np.max(abs(imaginary))),
                'metrics':[metrics(original['mass'],c,original['c0']) for c in states]}
        audits[name]=audit
        print(json.dumps({'reference_case':name,'phase':'finished','accepted_steps':audit['accepted_steps'],'RHS_evaluations':audit['RHS_evaluations']}),flush=True)
    controls={}
    for N in [128,256]:
        m=inputs[N]['mass'];c0=inputs[N]['c0'];mu=m@c0/m.sum();sigma=weighted_norm(m,c0-mu)
        relative=lambda v:weighted_norm(m,v)/sigma
        space=projected['M63-k128'][N]-projected['M127-k128'][N]
        temporal=projected['M127-k128'][N]-projected['M127-k256'][N]
        controls[str(N)]={'space_63_vs127_relative_to_original_fluctuation':[relative(v) for v in space],
            'max_step_halving_127_relative_to_original_fluctuation':[relative(v) for v in temporal],
            'final_raw_space_plus_time_difference_indicator':relative(space[-1])+relative(temporal[-1]),
            'H_final_space_difference':abs(audits['M63-k128']['observations'][str(N)]['metrics'][-1]['H']-audits['M127-k128']['observations'][str(N)]['metrics'][-1]['H']),
            'H_final_time_difference':abs(audits['M127-k128']['observations'][str(N)]['metrics'][-1]['H']-audits['M127-k256']['observations'][str(N)]['metrics'][-1]['H']),
            'indicator_is_a_rigorous_error_bound':False,'differences_prove_convergence_order':False,'extrapolated_state_used':False}
        saved[f'N{N}_space_difference']=space;saved[f'N{N}_time_difference']=temporal
        saved[f'N{N}_best_reference_states']=projected['M127-k256'][N]
    restricted=np.array([observe_parent(c,inputs[256]['mass'],inputs[128]['mass']) for c in projected['M127-k256'][256]])
    nesting=restricted-projected['M127-k256'][128];saved['reference_parent_nesting_difference']=nesting
    result={'reference_cases':audits,'analytic_initial_profile_PDE_controls':point_audits,'reference_uncertainty_controls':controls,
        'best_raw_reference':'M127-k256','best_reference_metrics_N128':audits['M127-k256']['observations']['128']['metrics'],
        'best_reference_metrics_N256':audits['M127-k256']['observations']['256']['metrics'],
        'reference_mass_projection_nesting_max_difference':float(np.max(abs(nesting))),
        'same_saved_mass_c0_ids':True,'new_reference_cases':3,'new_candidate_graphs':0,'new_candidate_transient_runs':0,
        'matrix_free_reference_only':True,'full_P_constructed':False,'fluid_orc_enabled':False,'stage3_physical_acceptance':'incomplete',
        'peak_process_RSS_MiB_linux':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
        'limits':['one prescribed smooth periodic shear, empirical reference differences not rigorous bounds',
                  'reference grid states are point values; comparisons use original-cell mass averages',
                  'no candidate shear actual scalar propagation or general shock/wall/ledger/DNS acceptance']}
    np.savez_compressed(out/'shear-reference.npz',**saved)
    for name,value in [('results.json',result),('environment.json',{'python':platform.python_version(),'numpy':np.__version__,'scipy':scipy.__version__,'threads':4})]:
        (out/name).write_text(json.dumps(value,indent=2)+'\n')
    print(json.dumps({'uncertainty_controls':controls,'best_reference_metrics_N128':result['best_reference_metrics_N128'],
        'projection_nesting_max_difference':result['reference_mass_projection_nesting_max_difference'],'peak_RSS_MiB':result['peak_process_RSS_MiB_linux'],'fluid_orc_enabled':False},indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args();main(a.output)
