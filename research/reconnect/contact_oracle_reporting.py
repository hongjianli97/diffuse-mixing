"""Restore original NumPy scalar types for unchanged budget summation and replay."""
from pathlib import Path
import argparse,json,copy
import numpy as np
from audit_route_decision import read,save,digest
from contact_oracle_core import evaluate,derive_material,cartesian_control
from contact_oracle_study import load_all
from affine_shear_completion import plain_json
from shear_union64_study import arrays,native_run
from t2_d0_core import bitwise_same


def evaluate_saved(data,meta,run,path,control,cm):
    # cm.total was np.float64 in the original Cartesian builder. Python3.12
    # sum(float...) uses compensated summation; sum(np.float64...) uses the
    # original sequential NumPy operation. Restore TYPE ONLY, not any bits.
    original_types=copy.deepcopy(cm)
    for row in original_types['rows']:
        value=row['total_absolute'];restored=np.float64(value)
        assert float(restored)==value
        row['total_absolute']=restored
    return evaluate(data,meta,run,path,control,original_types)


def record_recovery(directory,first_log):
    import shutil
    out=Path(directory).resolve();before={str(p.relative_to(out)):digest(p) for p in out.rglob('*') if p.is_file()}
    save(out/'pre-reporting-type-recovery-sha256.json',before)
    shutil.copyfile(first_log,out/'audit-first-attempt.log')
    source=Path(__file__).with_name('audit_contact_oracle.py');shutil.copyfile(source,out/'audit-first-attempt.py')
    r=plain_json(evaluate_saved(*load_all(out)));assert r==read(out/'results.json')
    save(out/'reporting-type-recovery.json',{'original_scientific_results_and_all_saved_files_unchanged':True,
      'reason':'original Cartesian totals np.float64; JSON reload Pythonfloat makes Python3.12 compensatedsum change lowbits',
      'change':'restore np.float64 type of saved Cartesian total scalars before same original evaluate; valuebits unchanged',
      'saved_values_thresholds_formulas_or_states_changed':False,'scientific_paths_rerun':0,'pre_recovery_files':len(before),
      'reporting_code_sha256':digest(Path(__file__)),'first_audit_source_sha256':digest(source)})
    for rel,h in before.items():assert digest(out/rel)==h,rel
    print('Original NumPy total-scalar types restored; unchanged evaluate reproduces original JSON exactly; no value/state/formula/threshold changed.',flush=True)


def replay(directory):
    out=Path(directory).resolve();data,meta,run,path,control,cm=load_all(out)
    d2,m2=derive_material(read(out/'source-reference-K128-p90.json'),arrays(out/'builder-inputs.npz'),32)
    for key,v in data.items():assert bitwise_same(v,d2[key]),key
    assert meta==m2
    r2,p2=native_run(out,16,data,read(out/'graph-index.json'),replay=True);assert r2==run
    for key,v in path.items():assert bitwise_same(v,p2[key]),key
    c2,cm2=cartesian_control(data,32);assert plain_json(cm2)==cm
    for key,v in control.items():assert bitwise_same(v,c2[key]),key
    expected=read(out/'results.json');assert plain_json(evaluate(data,meta,r2,p2,c2,cm2))==expected
    assert plain_json(evaluate_saved(data,meta,run,path,control,cm))==expected
    for rel,h in read(out/'pre-reporting-type-recovery-sha256.json').items():assert digest(out/rel)==h,rel
    print('One saved point-FV graph: K16 actual/full independent path,halfstates,rawCSR/weights,Cartesian full modes,finite-cellmeans,stored-coefficient references replayed bitwise. Original typed budget summation andallresults reproduced exactly; original auditfailure/allpre-recovery files unchanged. No geometry/oldcandidate/PDE/ORC rerun.',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--directory',required=True);p.add_argument('--first-log');p.add_argument('--replay',action='store_true');a=p.parse_args()
    if a.replay:replay(a.directory)
    elif a.first_log:record_recovery(a.directory,a.first_log)
    else:p.error('--first-log or--replay required')
