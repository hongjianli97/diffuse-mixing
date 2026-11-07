"""Read-only signed-error and retained-budget diagnostics of saved moving paths."""
from pathlib import Path
import argparse,math,resource,signal,time,traceback
import numpy as np
from audit_route_decision import ROOT,read,save,digest

PARENT=ROOT/'runs/reconnect-compression-density-calibration'
ORIGINAL=ROOT/'runs/reconnect-compression-union64'
PROTOCOL=ROOT/'research/reconnect/compression_density_diagnostics_protocol.json'
REF_C=('four_reference_space','four_reference_time')
REF_H=('two_reference_space_same_basis','two_reference_time_same_basis')


def inputs():
    with np.load(ORIGINAL/'union-definition.npz',allow_pickle=False) as z:
        d={k:z[k].copy() for k in ('mass','c0','reference','space','time','reference_space_coarse','reference_space_fine','reference_time_check')}
    paths={};runs={}
    for s in ('oracle','kernel','ledger'):
        for K in (16,32):
            folder=(ORIGINAL if s=='oracle' else PARENT/s)/f'k{K}'
            with np.load(folder/'trajectory-states.npz',allow_pickle=False) as z:paths[s,K]={key:z[key].copy() for key in ('states','independent_states')}
            runs[s,K]=read(folder/'native-results.json')
            for field in paths[s,K].values():
                assert field.shape==(K+1,4096) and np.isfinite(field).all();np.testing.assert_array_equal(field[0],d['c0']);field.setflags(write=False)
    for v in d.values():assert np.isfinite(v).all();v.setflags(write=False)
    assert d['mass'].shape==d['c0'].shape==(4096,) and np.all(d['mass']>0)
    return d,paths,runs


def diagnostics(d,paths,runs,original):
    m,c0=d['mass'],d['c0'];M=float(m.sum());V0=float(m@((c0-float(m@c0/M))**2));sigma=math.sqrt(V0)
    assert sigma==original['common_sigma64']
    inner=lambda a,b:float(m@(a*b));norm=lambda a:math.sqrt(inner(a,a))
    H=lambda a:1-float(m@((a-float(m@a/M))**2))/V0
    hm=lambda e:(2*math.sqrt(M)*max(1.,float(np.max(abs(c0))))*e+e*e)/V0
    rows=[]
    for s in ('kernel','ledger'):
        for K in (16,32):
            for k,t in ((1,.025),(2,.05)):
                u=paths[s,K]['states'][k*K//2];o=paths['oracle',K]['states'][k*K//2];ref=d['reference'][k]
                delta=u-o;eo=o-ref;es=u-ref;mu_delta=float(m@delta/M)
                oc=o-float(m@o/M);dc=delta-mu_delta
                cross=2*inner(eo,delta)/V0;energy=inner(delta,delta)/V0
                Eo,Es=norm(eo)/sigma,norm(es)/sigma;lhs=(inner(es,es)-inner(eo,eo))/V0
                hc=-2*inner(oc,dc)/V0;he=-inner(dc,dc)/V0;dh=H(u)-H(o)
                assert abs(lhs-cross-energy)<=64*np.finfo(float).eps*max(1.,abs(lhs),abs(cross)+energy)
                assert abs(dh-hc-he)<=64*np.finfo(float).eps*max(1.,abs(dh),abs(hc)+abs(he))
                assert energy>=0 and he<=0 and Eo+Es>0
                rows.append({'source':s,'K':K,'time':t,'concentration':{
                    'oracle_relative_error':Eo,'source_relative_error':Es,'relative_error_change':Es-Eo,
                    'squared_error_change':lhs,'twice_error_cross_term':cross,'source_difference_squared_norm':energy,
                    'identity_residual':lhs-cross-energy,'error_change_from_identity':(cross+energy)/(Es+Eo)},
                    'H':{'oracle_H':H(o),'source_H':H(u),'reference_H':H(ref),'signed_H_change':dh,
                         'negative_twice_centered_cross':hc,'negative_centered_difference_energy':he,
                         'identity_residual':dh-hc-he,'error_change':abs(H(u)-H(ref))-abs(H(o)-H(ref))},
                    'mass':{'source_minus_oracle_mass':float(m@delta),'source_minus_oracle_mean':mu_delta,
                            'mean_square_term_retained':M*mu_delta*mu_delta},'causal_density_accuracy_claim':False})
    budgets=[]
    for old in original['source_comparisons']:
        s=old['source'];k=round(old['time']/.025)
        cp={'four_reference_space':4*norm(d['space'][k]),'four_reference_time':4*norm(d['time'][k]),
            'two_initial_bridge':2*norm(d['reference'][0]-c0),'same_N64_projection_identity':0.}
        hp={'two_reference_space_same_basis':2*abs(H(d['reference_space_coarse'][k])-H(d['reference_space_fine'][k])),
            'two_reference_time_same_basis':2*abs(H(d['reference_time_check'][k])-H(d['reference'][k])),
            'two_initial_bridge':2*abs(H(d['reference'][0])-H(c0)),'same_N64_projection_identity':0.}
        mt_total=0.
        for src in (s,'oracle'):
            a,b=paths[src,16]['states'][8*k],paths[src,32]['states'][16*k]
            cp[src+'_16_32']=norm(a-b);hp[src+'_16_32']=abs(H(a)-H(b))
            for K in (16,32):
                rr=runs[src,K]['steps'][k*K//2-1];mt=rr['cumulative_machine_indicator_absolute']+rr['cumulative_tail_indicator_absolute'];mt_total+=mt
                a,b=paths[src,K]['states'][k*K//2],paths[src,K]['independent_states'][k*K//2]
                cp[f'{src}_e{K}']=mt+norm(a-b);hp[f'{src}_complete_sequence_H{K}']=abs(H(a)-H(b))
        hp['both_machine_tail_to_H']=hm(mt_total)
        assert cp==old['concentration_parts_absolute'] and hp==old['H_parts']
        record={'source':s,'time':old['time']}
        for metric,parts,ref_names,denom,change,key in (
            ('concentration',cp,REF_C,sigma,old['concentration_error_change'],'concentration_indicator'),
            ('H',hp,REF_H,1.,old['H_error_change'],'H_indicator')):
            assert all(v>=0 for v in parts.values());full=sum(parts.values())/denom;assert full==old[key]
            kept={name:v for name,v in parts.items() if name not in ref_names}
            floor=sum(kept.values())/denom
            categories={'time_16_32':math.fsum(parts[name] for name in (s+'_16_32','oracle_16_32'))/denom,
                'reference_space':parts[ref_names[0]]/denom,'reference_time':parts[ref_names[1]]/denom,
                'initial_bridge':parts['two_initial_bridge']/denom,'identity_projection':parts['same_N64_projection_identity']/denom,
                'numerical_sequences_machine_tail':math.fsum(v for name,v in parts.items() if name not in (*ref_names,s+'_16_32','oracle_16_32','two_initial_bridge','same_N64_projection_identity'))/denom}
            assert abs(math.fsum(categories.values())-full)<64*np.finfo(float).eps*max(1.,full)
            record[metric]={'full_indicator_unchanged':full,'observed_error_change_unchanged':change,'original_decision':old[metric+'_decision'],
                'parts_in_comparison_units':{name:v/denom for name,v in parts.items()},
                'ranked_terms':sorted(parts,key=lambda name:(-parts[name],name)),
                'category_values':categories,'category_percent_of_full':{name:100*v/full for name,v in categories.items()},
                'hypothetical_zero_reference_space_time_floor':floor,'retained_terms':list(kept),
                'removed_only_in_hypothetical_floor':list(ref_names),'floor_over_observed_change':floor/abs(change),
                'signed_reference_allowance_at_fixed_observed_change':abs(change)-floor,
                'reference_indicator_reduction_alone_sufficient_at_fixed_observed_change':abs(change)>floor,
                'hypothetical_floor_used_for_acceptance':False,'replacement_reference_gain_assumed_unchanged':False}
        budgets.append(record)
    return {'date':'2026-11-07','timezone':'Asia/Shanghai','common_sigma64':sigma,'M':M,'V0':V0,'signed_identities':rows,'budget_diagnostics':budgets,
        'original_decisions_preserved':True,'new_density_fields':0,'new_coefficient_packets':0,'new_geometry_graphs':0,
        'new_actual_or_independent_paths':0,'new_reference_PDE_solves':0,'new_software_checks':0,'recorded_checks_total':506,
        'time_order_or_decline_certified':False,'indicators_are_certified_PDE_error_bounds':False,
        'all_Becker_unresolved_gates_preserved':True,'stage3':'incomplete','fluid_orc_enabled':False,'kappa_computed':False}


def preserve():
    old=read(PARENT/'preservation.json');formal=old['formal_manifests_verified']+[{'manifest':str((PARENT/'sha256.json').relative_to(ROOT)),
        'manifest_sha256':digest(PARENT/'sha256.json'),'entries_verified':len(read(PARENT/'sha256.json'))}]
    assert len(formal)==119
    for item in formal:
        p=ROOT/item['manifest'];assert digest(p)==item['manifest_sha256'];entries=read(p);assert len(entries)==item['entries_verified']
        for name,h in entries.items():assert digest(ROOT/name)==h,name
    originals=read('/workspace/original-handoff-sha256.json');attachments=read('/workspace/setup-ltn/attachment-manifest.json')
    assert len(originals)==782 and len(attachments)==15
    for name,h in originals.items():assert digest(ROOT/name)==h,name
    for item in attachments:
        p=Path('/workspace')/item['path'];assert p.stat().st_size==item['bytes'] and digest(p)==item['sha256']
    for item in old['historical_diagnostic_differences_retained']:
        p=ROOT/item['manifest'];assert digest(p)==item['manifest_sha256'];entries=read(p)
        diff=[{'path':k,'saved':v,'current':digest(ROOT/k)} for k,v in entries.items() if digest(ROOT/k)!=v]
        assert sorted(diff,key=lambda x:x['path'])==sorted(item['historical_paths_differing_from_later_formal_versions'],key=lambda x:x['path'])
    return {'formal_manifests_verified':formal,'originals_verified':782,'attachments_verified':15,
        'historical_diagnostic_differences_retained':old['historical_diagnostic_differences_retained'],'all_prior_scientific_files_unchanged':True}


def main(directory):
    out=Path(directory).resolve();out.mkdir(parents=True,exist_ok=False);start=time.monotonic()
    def timeout(*_):raise TimeoutError('read-only120second budget')
    signal.signal(signal.SIGALRM,timeout);signal.alarm(120)
    try:
        p=read(PROTOCOL);gate=read(PARENT/'next-gate-spec.json')
        assert gate['type']=='read_only_moving_density_signed_error_and_resolution_budget' and not gate['already_executed']
        assert digest(PARENT/'next-gate-spec.json')==p['source_gate_sha256']
        for name,h in p['sources'].items():assert digest(ROOT/name)==h,name
        save(out/'protocol.json',p);save(out/'source-gate-spec.json',gate)
        save(out/'implementation-lock.json',{'files':{name:digest(ROOT/name) for name in p['implementation_files']},
            'locked_before_new_diagnostics':True,'no_new_physical_arrays_or_solvers':True})
        d,paths,runs=inputs();result=diagnostics(d,paths,runs,read(PARENT/'results.json'));save(out/'results.json',result)
        save(out/'preservation.json',preserve())
        for name,h in p['sources'].items():assert digest(ROOT/name)==h,name
        elapsed=time.monotonic()-start;peak=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024
        assert elapsed<120 and peak<1024
        save(out/'execution.json',{'completed':True,'elapsed_seconds':elapsed,'peak_RSS_MiB':peak,'source_files_verified':len(p['sources']),
            'new_density_or_geometry_or_solver_calls':0,'new_reference_PDE_solves':0,'new_software_checks':0,'old_tests_rerun':0,
            'published_or_pushed':False})
        print('Eight saved signed-error/H identities and four retained-budget rows computed; no new physical calculations.',flush=True)
    except BaseException as exc:
        save(out/'failed-execution.json',{'type':type(exc).__name__,'message':str(exc),'traceback':traceback.format_exc(),'no_automatic_retry_or_repair':True});raise
    finally:signal.alarm(0)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--directory',required=True);main(p.parse_args().directory)
