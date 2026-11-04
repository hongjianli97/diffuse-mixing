"""One finite-time actual scalar/mixing comparison on the saved smooth case."""
from pathlib import Path
import argparse,hashlib,json,platform,resource
import numpy as np
import scipy
from scipy.sparse import csr_matrix
from trajectory_chain import TrajectorySamples,SI_UNITS,mixing_metrics,ALGEBRA_TOL
from chain_study import verify
from spectral_density_reference import reference_path
from sparse_density_evolution import evolve_saved


def main(output):
    root=Path(__file__).resolve().parents[2];source=root/'runs/reconnect-kernel2d-variable-refinement';verify(root,source)
    protocol=json.loads((source/'stage3-status.json').read_text())['next_transient_research_predeclared']
    if protocol['times_s']!=[0.,.05,.1] or protocol['reference_resolutions']!=[256,512] or not protocol['same_saved_N128_h004_D002_epsilon025_scalar_and_graph']:raise RuntimeError('predeclared transient protocol changed')
    out=Path(output).resolve();out.mkdir(parents=True,exist_ok=False)
    cfg={'study':'one actual finite-time scalar/mixing research comparison on saved positive graph','source':str(source),'protocol':protocol,
        'protocol_source_sha256':hashlib.sha256((source/'stage3-status.json').read_bytes()).hexdigest(),
        'N_per_direction':128,'h_m':.04,'D_m2_per_s':.02,'density_amplitude':.25,
        'candidate':'unchanged saved L, Q=-M^-1 L, sparse exponential action on actual saved c0',
        'trajectory_source':'given analytic rigid translation explicitly sampled at all requested times',
        'reference':'independent 1D periodic self-adjoint transformed local PDE, raw spectral eigendecomposition and exact Fourier cell integration',
        'error_normalization':'sqrt(sum m*(candidate-reference)^2)/sqrt(sum m*(saved_c0-mean_m)^2)',
        'mixing_scope':'mass-average cell states on the same saved material partition, not unresolved continuum variance',
        'algorithm_controls':'full vs two half exponentials; sampled path; raw mass-symmetric representation',
        'new_graphs':0,'spatial_parameter_scan':False,'clip_or_renormalize':False,'fluid_orc_enabled':False,'stage3_physical_acceptance':'incomplete'}
    (out/'config.json').write_text(json.dumps(cfg,indent=2)+'\n')
    with np.load(source/'variable-density-refined.npz') as z:
        m=z['mass'];c0=z['c0'];labels=z['material_label_centers'];rho=z['rho'][0];ids=z['ids']
        L=csr_matrix((z['L_data'],z['L_indices'],z['L_indptr']),shape=tuple(z['L_shape']))
        saved_positions=z['positions'];saved_times=z['time']
    times=np.array(protocol['times_s']);velocity=np.array([.2,-.1]);positions=np.mod(labels[None,...]+times[:,None,None]*velocity,1.)
    data=TrajectorySamples(times,positions,ids,m,c0,SI_UNITS,rho=np.broadcast_to(rho,(len(times),len(m))))
    for index,t in enumerate(saved_times):
        matches=np.flatnonzero(times==t)
        if len(matches)!=1 or not np.array_equal(positions[matches[0]],saved_positions[index]):raise RuntimeError('explicit analytic trajectory differs from saved frame')
    # Geometry changes only by the prescribed translation. Periodic pair
    # distances and along-identity rho are invariant analytically, so saved L
    # remains the intended material-label operator throughout this interval.
    states,algorithm_arrays,algorithm=evolve_saved(m,L,c0,times)
    ref_states={};ref_amplitudes={};ref_audits={}
    for M in protocol['reference_resolutions']:
        ref_states[M],ref_amplitudes[M],ref_audits[M]=reference_path(M,128,.25,.02,times,m)
    E0=mixing_metrics(m,c0,1.)['energy_kg'];norm=np.sqrt(2*E0);error=lambda v:float(np.sqrt(m@(v*v))/norm)
    rows=[]
    for index,t in enumerate(times):
        actual=states[index];reference=ref_states[512][index];candidate_metrics=mixing_metrics(m,actual,E0);reference_metrics=mixing_metrics(m,reference,E0)
        row={'time_s':float(t),'relative_concentration_error_initial_fluctuation_norm':error(actual-reference),
            'reference_256_vs_512_relative_difference':error(ref_states[256][index]-reference),
            'candidate_metrics':candidate_metrics,'reference_metrics_same_saved_initial_E':reference_metrics,
            'H_candidate_minus_reference':candidate_metrics['H']-reference_metrics['H'],
            'candidate_min':float(actual.min()),'candidate_max':float(actual.max()),'reference_min':float(reference.min()),'reference_max':float(reference.max()),
            'relative_component_mass_drift':abs(float(m@(actual-c0)))/float(m@c0),
            'energy_dissipation_rate_kg_per_s':float(actual@(L@actual))}
        if row['relative_component_mass_drift']>ALGEBRA_TOL or actual.min()<c0.min()-ALGEBRA_TOL or actual.max()>c0.max()+ALGEBRA_TOL:raise RuntimeError('raw candidate violates unchanged algebra/bounds check')
        rows.append(row)
    algorithm['full_vs_two_half_initial_fluctuation_norm']=error(algorithm_arrays['full_endpoint']-algorithm_arrays['two_half_endpoint'])
    algorithm['Q_vs_mass_symmetric_initial_fluctuation_norm']=error(algorithm_arrays['full_endpoint']-algorithm_arrays['mass_symmetric_endpoint'])
    reference_control={'spectral_audits':{str(M):a for M,a in ref_audits.items()},
        'initial_reference512_vs_saved_c0_max_difference':float(np.max(abs(ref_states[512][0]-c0))),
        'initial_reference512_vs_saved_c0_relative_difference':error(ref_states[512][0]-c0),
        'spectral_refinement_is_empirical_not_rigorous_bound':True}
    result={'runs':rows,'algorithm_controls':algorithm,'reference_controls':reference_control,
        'candidate_state_history_matches_actual_saved_c0_at_zero':bool(np.array_equal(states[0],c0)),
        'mixing_energy_monotone_at_saved_times':bool(all(rows[i+1]['candidate_metrics']['energy_kg']<=rows[i]['candidate_metrics']['energy_kg'] for i in range(len(rows)-1))),
        'same_saved_mass_c0_and_L':True,'new_graphs':0,'new_finite_time_case':1,'full_P_constructed':False,
        'peak_process_RSS_MiB_linux':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
        'fluid_orc_enabled':False,'stage3_physical_acceptance':'incomplete',
        'limits':['one smooth periodic translation, fixed h and fixed graph','no production backward-Euler time-step acceptance','no general time-dependent/noncommuting flow geometry acceptance','no shock/contact, nonuniform cloud, wall, ledger or DNS acceptance']}
    saved={'time':times,'positions':data.positions,'ids':ids,'mass':m,'c0':c0,'rho':data.rho,'material_label_centers':labels,
        'candidate_states':states,'reference256_states':ref_states[256],'reference512_states':ref_states[512],
        'reference256_amplitudes':ref_amplitudes[256],'reference512_amplitudes':ref_amplitudes[512],**algorithm_arrays}
    np.savez_compressed(out/'density-transient.npz',**saved)
    for name,value in [('results.json',result),('environment.json',{'python':platform.python_version(),'numpy':np.__version__,'scipy':scipy.__version__,'threads':4})]:
        (out/name).write_text(json.dumps(value,indent=2)+'\n')
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args();main(a.output)
