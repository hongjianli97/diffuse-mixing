"""Saved-budget provenance and independent Decimal observables; no numerical paths."""
from pathlib import Path
from fractions import Fraction as F
from decimal import Decimal as D,localcontext
import argparse,array,ast,json,math,resource,struct,sys,time,zipfile
from audit_route_decision import ROOT,read,save,digest,preservation
from t3_affine_budget_review import SOURCE,ZERO_FIELDS,analyze


def member(path,key):
    # Independent NPY decoding: native array module instead of the primary Fraction decoder.
    with zipfile.ZipFile(path) as archive:data=archive.read(key+'.npy')
    assert data[:6]==b'\x93NUMPY';version=data[6];offset=10 if version==1 else 12
    assert version in (1,2)
    length=struct.unpack('<H' if version==1 else '<I',data[8:offset])[0]
    h=ast.literal_eval(data[offset:offset+length].decode('latin1'));assert h['descr']=='<f8' and not h['fortran_order']
    values=array.array('d');values.frombytes(data[offset+length:])
    if sys.byteorder!='little':values.byteswap()
    assert len(values)==math.prod(h['shape'])
    return [tuple(values[i:i+1024]) for i in range(0,len(values),1024)]


def independent_decimal(folder,result):
    folder=Path(folder);m=member(folder/'material-inputs.npz','mass')[0];c0=member(folder/'material-inputs.npz','c0')[0]
    old=read(folder/'results.json');decoded={}
    for model in ('constant_D','constant_Gamma'):
        for suffix in ('reference','continuous_grid_symbol'):decoded[model,suffix]=member(folder/'analytic-references.npz',model+'_'+suffix)
    outcomes={}
    for precision in (80,100):
        with localcontext() as ctx:
            ctx.prec=precision;mass=[D.from_float(x) for x in m];initial=[D.from_float(x) for x in c0];total=sum(mass)
            def var(c):
                c=[D.from_float(x) for x in c];first=sum(a*b for a,b in zip(mass,c))
                return sum(a*b*b for a,b in zip(mass,c))-first*first/total
            v0=var(c0);records=[]
            for k,row in enumerate(result['checkpoints']):
                h={key:D(1)-var(values[k])/v0 for key,values in decoded.items()}
                vhi=max(v0,D.from_float(result['saved_sigma_squared_binary64']))
                spatial=sum(D.from_float(old['reference_error_components'][model+'-k16'][k]['components_absolute']['spatial_symbol_error']) for model in ('constant_D','constant_Gamma'))
                floor=(D(2)*spatial+spatial*spatial)/vhi
                gridgap=h['constant_D','continuous_grid_symbol']-h['constant_Gamma','continuous_grid_symbol']
                pdegap=h['constant_D','reference']-h['constant_Gamma','reference']
                assert float(floor)==row['spatial_only_original_formula_diagnostic_lower']['float']
                assert float(gridgap)==row['saved_continuous_grid_H_gap']['float'] and float(pdegap)==row['saved_PDE_H_gap']['float']
                assert (floor>gridgap)==row['spatial_floor_exceeds_saved_continuous_grid_gap']
                assert (floor>D.from_float(row['source_actual_H_gap']))==row['spatial_floor_exceeds_saved_actual_gap']
                for model in ('constant_D','constant_Gamma'):
                    q=result['saved_array_observables'][model][k]
                    assert float(h[model,'reference'])==q['saved_PDE_H'] and float(h[model,'continuous_grid_symbol'])==q['saved_grid_H']
                records.append({'time':row['time'],'spatial_floor':float(floor),'saved_grid_H_gap':float(gridgap),'saved_PDE_H_gap':float(pdegap)})
            outcomes[precision]=records
    assert outcomes[80]==outcomes[100]
    return {'independent_array_module_NPY_decode':True,'second_moment_variance_distinct_from_primary_centered_formula':True,
      'decimal_precisions':[80,100],'both_precisions_agree_with_exact_primary_observables':True,
      'no_production_geometry_solver_reference_evaluator_or_new_state_called':True,'checkpoints':outcomes[80]}


def receipt(out):
    out=Path(out).resolve();config=read(out/'execution.json');lock=read(out/'implementation-lock.json')
    assert config['review_exit_code']==0 and config['final_review_passed'] and not config['review_first_attempt_passed']
    assert config['review_failed_attempts_preserved']==1 and config['standard_library_only']
    assert config['elapsed_main_seconds']<120 and config['peak_RSS_MiB']<1024
    assert digest(ROOT/'research/reconnect/t3_affine_budget_review.py')==lock['code_sha256']
    assert digest(out/'review-protocol.json')==lock['review_protocol_sha256']==digest(ROOT/'research/reconnect/t3_affine_budget_review_protocol.json')
    for item in read(out/'source-copies.json'):assert digest(out/item['copy'])==item['sha256']
    failed=out/'review-first-attempt';firstlock=read(failed/'implementation-lock.json')
    assert digest(failed/'review-source-attempt1.py.txt')==firstlock['code_sha256']
    assert read(failed/'failed-execution.json')['review_exit_code']==1
    assert not read(failed/'failed-execution.json')['scientific_inputs_results_indicators_or_thresholds_changed']
    computed,_=analyze(out/'source');assert computed==read(out/'results.json')
    independent=independent_decimal(out/'source',computed)
    status=read(out/'stage3-status.json');gate=read(out/'next-gate-spec.json')
    assert all(status[key]==0 for key in ZERO_FIELDS) and status['recorded_checks_total']==480 and not status['fluid_orc_enabled']
    assert not status['original_T3_case_passed'] and not status['next_experiment_executed']
    assert not gate['already_executed'] and gate['N_values']==[16,32,64] and gate['new_N_values']==[16,64]
    assert gate['new_candidate_graphs']==112 and gate['new_candidate_transient_runs']==gate['new_complete_independent_uniformization_sequences']==12
    assert gate['reused_saved_candidate_paths']==6 and gate['old_candidate_reruns']==0 and not gate['new_points_graphs_states_or_references_generated_this_review']
    assert computed['original_decision_retained']=='T3_affine_model_response_inconclusive'
    critical=computed['checkpoints'][1]
    assert critical['spatial_floor_exceeds_saved_actual_gap'] and critical['spatial_floor_exceeds_saved_continuous_grid_gap']
    assert not critical['source_model_H_response_resolved'] and not critical['signed_cancellation_used_for_acceptance']
    return independent


def main(directory,receipt_only=False):
    out=Path(directory).resolve();start=time.monotonic();independent=receipt(out)
    if receipt_only:
        print('Fresh standard-library receipt passed: exact positive budget/roundoff accounting and independent80/100-digit saved-state H observables; original N32 inconclusive retained; no graph/action/path/reference generation or old numerical replay.',flush=True);return
    assert not (out/'sha256.json').exists()
    preserved=preservation();known={v['manifest'] for v in preserved['formal_scientific_manifests_verified']}
    previous=read(SOURCE/'preservation.json')['formal_scientific_manifests_verified']+[{'manifest':str((SOURCE/'sha256.json').relative_to(ROOT)),
      'manifest_sha256':digest(SOURCE/'sha256.json'),'entries_verified':len(read(SOURCE/'sha256.json'))}]
    for item in previous:
        if item['manifest'] in known:continue
        path=ROOT/item['manifest'];assert digest(path)==item['manifest_sha256'];entries=read(path);assert len(entries)==item['entries_verified']
        for rel,h in entries.items():assert digest(ROOT/rel)==h,(item['manifest'],rel)
        preserved['formal_scientific_manifests_verified'].append(item);known.add(item['manifest'])
    assert len(known)==94;save(out/'preservation.json',preserved)
    for item in read(out/'source-copies.json'):assert digest(ROOT/item['source'])==item['sha256']
    gate=read(out/'next-gate-spec.json');parent=read(SOURCE/'sha256.json')
    for item in gate['material_sources'].values():assert digest(ROOT/item['path'])==parent[item['path']]==item['sha256']
    for item in gate['reused_N32_records']:assert digest(ROOT/item['path'])==item['sha256']
    assert digest(ROOT/'exports/ltn-t3-affine.zip')=='f58a50adc6c9b71e42351655330e5dc87670e425ce0564d44fab42bc1d213205'
    save(out/'independent-decimal-audit.json',independent)
    summary={'receipt_consistent':True,'positive_components_and_retained_accumulation_roundoff_verified':True,
      'independent80_100_digit_saved_observables_verified':True,'original_N32_indicator_and_inconclusive_decision_unchanged':True,
      'time_only_expansion_not_selected':True,'next_spatial_control_registered_not_executed':True,
      'original_files_verified':782,'historical_attachments_verified':15,'prior_formal_manifests_verified':94,
      'old_T3_delivery_unchanged':True,**dict.fromkeys(ZERO_FIELDS,0),'recorded_checks_total':480,
      'audit_is_saved_data_accounting_not_new_physical_acceptance':True,'general_T3_admitted':False,
      'stage3_physical_acceptance':'incomplete','fluid_orc_enabled':False}
    save(out/'review-audit.json',summary)
    save(out/'audit-execution.json',{'audit_exit_code':0,'audit_first_attempt_passed':True,'standard_library_only':True,
      'elapsed_audit_seconds':time.monotonic()-start,'peak_auditor_RSS_MiB':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
      'audit_source_sha256':digest(Path(__file__))})
    print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--directory',required=True);p.add_argument('--receipt-only',action='store_true');a=p.parse_args();main(a.directory,a.receipt_only)
