"""One fixed-time analytic contact reference with predeclared controls."""
from decimal import Decimal as Dec, localcontext
from fractions import Fraction as F
from pathlib import Path
import argparse
import hashlib
import json
import platform

from audit_route_decision import ROOT, digest, read, save
from contact_spectral_reference import reference_case,compare_cases,field


SOURCE=ROOT/'runs/reconnect-FV-global-frozen'
THEORY=ROOT/'research/reconnect/CONTACT_PERIODIC_SPECTRAL_BASIS.md'
CASES=[(64,60),(128,60),(64,90),(128,90)]


def array_record(array):
    return {'shape':list(array.shape),'dtype':array.dtype.str,
        'C_order_sha256':hashlib.sha256(array.tobytes(order='C')).hexdigest()}


def compute(packet):
    cases=[reference_case(packet['model'],K,p) for K,p in CASES]
    result=compare_cases(cases,packet['source_arrays'])
    return cases,result


def replay(output):
    out=Path(output).resolve();packet=read(out/'reference-inputs.json')
    cases,result=compute(packet)
    for (K,p),case in zip(CASES,cases):assert case==read(out/f'reference-K{K}-p{p}.json')
    assert result==read(out/'results.json')
    for name,expected in read(out/'input-provenance.json')['receipt_SHA256'].items():
        assert digest(out/name)==expected,name
    print(json.dumps({'standard_library_reference_replay_identical':True,'physical_time':'1/20',
        'truncations':[64,128],'decimal_digits':[60,90],'source_state_unchanged':True,
        'candidate_and_FV_time_propagations':0,'recorded_checks_total':396,'fluid_orc_enabled':False}),flush=True)


def main(output,tests,test_log):
    import numpy as np
    manifest=read(SOURCE/'sha256.json')
    for rel,expected in manifest.items():assert digest(ROOT/rel)==expected,rel
    protocol_path=SOURCE/'stage3-status.json'
    protocol=read(protocol_path)['next_same_contact_finite_time_reference_predeclared']
    assert not protocol['already_executed'] and protocol['T_exact']=='1/20'
    assert protocol['reference_truncation_cutoffs']==[64,128] and protocol['reference_arithmetic_digits']==[60,90]
    assert digest(ROOT/protocol['source_snapshot'])==protocol['source_snapshot_sha256']
    checks=read(tests);assert checks['successful'] and checks['checks_run']==4
    assert checks['errors']==checks['failures']==checks['skipped']==0
    out=Path(output).resolve();out.mkdir(parents=True,exist_ok=False)
    source_observations=SOURCE/'selected-source-observations.npz'
    provenance=read(SOURCE/'input-provenance.json')
    assert digest(source_observations)==provenance['selected_NPZ_sha256']
    with np.load(source_observations,allow_pickle=False) as z:
        raw={key:z[key].copy() for key in ['ids','mass','c0','positions','rho']}
    N=128;assert raw['ids'].tolist()==list(range(N*N))
    assert raw['positions'].tolist()==[[(i+.5)/N,(j+.5)/N] for j in range(N) for i in range(N)]
    assert raw['rho'].tolist()==[1. if i<N//2 else 4. for j in range(N) for i in range(N)]
    assert all(F(float(m))==F(float(rho))/(N*N) for m,rho in zip(raw['mass'],raw['rho']))
    packet={'model':{'N':N,'rho_left':'1','rho_right':'4','D_exact':protocol['D_binary_input_exact'],
                     'T_exact':protocol['T_exact'],'amplitude_exact':'1/5','phase_pi_ratio':'1/5'},
            'source_arrays':{key:array.tolist() for key,array in raw.items()},
            'source_arrays_are_used_only_for_geometry_validation_and_rounding_bridge_not_projections':True}
    save(out/'reference-inputs.json',packet)
    receipts={'basis-theory-at-evaluation.md':THEORY,'source-stage3-status.json':protocol_path,
              'source-global-frozen-results.json':SOURCE/'results.json','tests.json':Path(tests),'tests.log':Path(test_log)}
    for name,path in receipts.items():
        with (out/name).open('xb') as f:f.write(path.read_bytes())
    save(out/'config.json',{'study':'independent same-contact finite-time material-mean PDE reference',
        'protocol':protocol,'protocol_source_sha256':digest(protocol_path),
        'basis_theory_sha256_before_reference_evaluation':digest(THEORY),
        'source_manifest_entries_verified':len(manifest),'controls':CASES,'software_checks':checks,
        'reference_initial_formula_is_not_fitted_to_saved_means':True,
        'bridge_is_only_a_bound_under_explicit_constant_perturbation_lift':True,
        'no_FV_or_kernel_time_propagation':True,'no_physical_acceptance_threshold_modified':True})
    cases,result=compute(packet)
    for (K,p),case in zip(CASES,cases):save(out/f'reference-K{K}-p{p}.json',case)
    save(out/'results.json',result)
    chosen=cases[-1]
    with localcontext() as ctx:
        ctx.prec=120
        reference=np.asarray([float(v) for v in field(chosen)],dtype=np.float64)
        initial=np.asarray([float(Dec('.5')+Dec(x)*Dec(y)) for y in chosen['transverse_cell_means']
                            for x in chosen['initial_normal_cell_means']],dtype=np.float64)
    np.savez_compressed(out/'reference-T0.05.npz',**raw,reference_cbar_T=reference,
        math_initial_cbar_reference_only=initial,T_exact_as_float_label_only=np.array(.05))
    save(out/'input-provenance.json',{'original_snapshot':protocol['source_snapshot'],
        'original_snapshot_sha256':protocol['source_snapshot_sha256'],
        'source_selected_observations':str(source_observations.relative_to(ROOT)),
        'source_selected_observations_sha256':digest(source_observations),
        'original_member_records':{key:array_record(array) for key,array in raw.items()},
        'output_reference_NPZ_sha256':digest(out/'reference-T0.05.npz'),
        'reference_mean_array_record':array_record(reference),
        'math_initial_mean_array_record':array_record(initial),
        'receipt_SHA256':{name:digest(path) for name,path in receipts.items()},
        'NPZ_T_float_is_label_only_exact_time_is_1_over_20':True})
    save(out/'environment.json',{'python':platform.python_version(),'numpy_serialization_only':np.__version__,
        'analytic_reference_and_replay':'standard library Fraction and Decimal; no eigenvalue solver or numerical quadrature'})
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    group=parser.add_mutually_exclusive_group(required=True);group.add_argument('--output');group.add_argument('--replay')
    parser.add_argument('--tests');parser.add_argument('--test-log');args=parser.parse_args()
    if args.replay:replay(args.replay)
    else:
        if not args.tests or not args.test_log:parser.error('--tests and --test-log required for source extraction')
        main(args.output,args.tests,args.test_log)
