"""Read only saved states/densities and retain all prior scientific judgments."""
from pathlib import Path
import argparse,math,resource,signal,time,traceback
import numpy as np
from audit_route_decision import ROOT,read,save,digest

PARENT=ROOT/'runs/reconnect-compression-joint-control'
OLD=ROOT/'runs/reconnect-compression-density-calibration'
PROTOCOL=ROOT/'research/reconnect/joint_evidence_protocol.json'


def state_identities():
    with np.load(PARENT/'comparison-inputs.npz',allow_pickle=False) as z:m,c,R=z['mass'],z['c0'],z['reference']
    M=float(m.sum());V=float(m@((c-float(m@c/M))**2));inner=lambda a,b:float(m@(a*b))
    H=lambda a:1-inner(a-float(m@a/M),a-float(m@a/M))/V
    paths={}
    for s in ('oracle','ledger'):
        with np.load(PARENT/s/'k64/trajectory-states.npz',allow_pickle=False) as z:paths[s]=z['states'][::32]
    rows=[]
    for k,t in ((1,.025),(2,.05)):
        o,u=paths['oracle'][k],paths['ledger'][k];e=o-R[k];d=u-o;mu=float(m@d/M)
        cross=2*inner(e,d)/V;positive=inner(d,d)/V;eo=math.sqrt(inner(e,e)/V);es=math.sqrt(inner(u-R[k],u-R[k])/V)
        oc=o-float(m@o/M);dc=d-mu;hc=-2*inner(oc,dc)/V;hq=-inner(dc,dc)/V
        assert abs(es*es-eo*eo-cross-positive)<64*np.finfo(float).eps
        assert abs(H(u)-H(o)-hc-hq)<64*np.finfo(float).eps
        rows.append({'time':t,'twice_error_cross':cross,'state_difference_square':positive,
            'squared_error_change':es*es-eo*eo,'relative_error_change_from_identity':(cross+positive)/(es+eo),
            'error_update_cosine':inner(e,d)/math.sqrt(inner(e,e)*inner(d,d)),
            'H_negative_twice_centered_cross':hc,'H_negative_centered_square':hq,'signed_H_change':H(u)-H(o),
            'raw_source_mass_difference':float(m@d),'mean_square_term_retained':M*mu*mu,
            'density_accuracy_implied_by_concentration_improvement':False})
    density=[]
    for step in range(64):
        with np.load(PARENT/f'oracle/k64/density-step{step:02d}.npz',allow_pickle=False) as z:o=z['rho']
        with np.load(PARENT/f'ledger/k64/density-step{step:02d}.npz',allow_pickle=False) as z:u=z['rho']
        assert np.all(o>0) and np.all(u>0)
        d=u-o
        density.append({'step':step,'midpoint_time':(2*step+1)/2560,
            'relative_point_Linf':float(np.max(abs(d/o))),
            'relative_mass_L2':math.sqrt(inner(d,d)/inner(o,o)),
            'mass_weighted_signed_bias':float(m@d/M),'minimum_signed_relative_error':float(np.min(d/o)),
            'maximum_signed_relative_error':float(np.max(d/o))})
    return rows,density


def budgets():
    new=read(PARENT/'results.json');old=read(OLD/'results.json');rows=[]
    for r in new['source_comparisons']:
        previous=next(q for q in old['source_comparisons'] if q['source']=='ledger' and q['time']==r['time'])
        for metric,key,denom,refs in (
            ('concentration','concentration_parts_absolute',new['common_sigma64'],('four_reference_space','four_reference_time')),
            ('H','H_parts',1.,('two_reference_space_same_basis','two_reference_time_same_basis'))):
            p=r[key];full=sum(p.values())/denom;gain=-r[metric+'_error_change']
            times=('ledger_32_64','oracle_32_64')
            categories={'candidate_time':sum(p[k] for k in times)/denom,'reference_space':p[refs[0]]/denom,
                'reference_time':p[refs[1]]/denom,'initial_bridge':p['two_initial_bridge']/denom,
                'identity':p['same_N64_projection_identity']/denom,
                'numerical':sum(v for k,v in p.items() if k not in (*refs,*times,'two_initial_bridge','same_N64_projection_identity'))/denom}
            assert all(v>=0 for v in p.values()) and full==r[metric+'_indicator'] and gain>0
            rows.append({'time':r['time'],'metric':metric,'full_budget':full,'observed_gain':gain,
                'margin_gain_minus_budget':gain-full,'budget_over_gain':full/gain,
                'unchanged_decision':r[metric+'_decision'],'categories':categories,
                'category_percent':{k:100*v/full for k,v in categories.items()},
                'fixed_gain_zero_reference_remaining':sum(v for k,v in p.items() if k not in refs)/denom,
                'fixed_gain_zero_candidate_time_remaining':sum(v for k,v in p.items() if k not in times)/denom,
                'old_case':{'decision':previous[metric+'_decision'],'budget':previous[metric+'_indicator'],
                    'gain':-previous[metric+'_error_change']},
                'new_over_old_budget':full/previous[metric+'_indicator'],
                'new_over_old_observed_gain':gain/(-previous[metric+'_error_change']),
                'old_case_rejudged':False,'hypothetical_remainders_used_for_acceptance':False,
                'future_gain_assumed_fixed':False})
    return rows


def evidence(protocol):
    old=read(ROOT/'runs/reconnect-stage3-evidence/evidence-ledger.json');cases=[]
    for case in old['case_rows']:
        saved=read(ROOT/case['source']);assert digest(ROOT/case['source'])==case['source_sha256']
        assert saved[case['flag']]==case['expected'];cases.append(case)
    latest={}
    for item in protocol['latest_cases']:
        r=read(ROOT/item['path']);latest[item['id']]={'path':item['path'],'sha256':digest(ROOT/item['path']),
            'saved_fields':{key:r[key] for key in item['fields']}}
    requirements=[]
    for q in protocol['requirements']:
        assert (ROOT/q['path']).read_text().splitlines()[q['line']-1]==q['text'];requirements.append(q)
    return {'original_requirements':requirements,'unchanged_prior_cases':cases,'latest_evidence':latest,
        'coverage':protocol['coverage'],'ledger_beating_oracle_is_required':False,
        'new_density_accuracy_threshold_added':False,'physical_completion_percentage':None,
        'DNS_T5_T6_v080_or_ORC_added_as_prerequisite':False}


def preserve():
    old=read(PARENT/'preservation.json');p=PARENT/'sha256.json'
    formal=old['formal_manifests_verified']+[{'manifest':str(p.relative_to(ROOT)),
        'manifest_sha256':digest(p),'entries_verified':len(read(p))}]
    assert len(formal)==122
    cache={}
    def sha(p):
        key=str(p)
        if key not in cache:cache[key]=digest(p)
        return cache[key]
    for item in formal:
        p=ROOT/item['manifest'];assert sha(p)==item['manifest_sha256'];entries=read(p);assert len(entries)==item['entries_verified']
        for name,h in entries.items():assert sha(ROOT/name)==h,name
    originals=read('/workspace/original-handoff-sha256.json');attachments=read('/workspace/setup-ltn/attachment-manifest.json')
    assert len(originals)==782 and len(attachments)==15
    for name,h in originals.items():assert sha(ROOT/name)==h,name
    for item in attachments:
        p=Path('/workspace')/item['path'];assert p.stat().st_size==item['bytes'] and sha(p)==item['sha256']
    for item in old['historical_diagnostic_differences_retained']:
        p=ROOT/item['manifest'];assert sha(p)==item['manifest_sha256']
        diff=[{'path':k,'saved':v,'current':sha(ROOT/k)} for k,v in read(p).items() if sha(ROOT/k)!=v]
        assert sorted(diff,key=lambda x:x['path'])==sorted(item['historical_paths_differing_from_later_formal_versions'],key=lambda x:x['path'])
    return {'formal_manifests_verified':formal,'originals_verified':782,'attachments_verified':15,
        'historical_diagnostic_differences_retained':old['historical_diagnostic_differences_retained'],
        'all_prior_scientific_files_unchanged':True}


def main(directory):
    out=Path(directory).resolve();out.mkdir(exist_ok=False);start=time.monotonic()
    def alarm(*_):raise TimeoutError('read-only120second cap')
    signal.signal(signal.SIGALRM,alarm);signal.alarm(120)
    try:
        p=read(PROTOCOL);gate=read(PARENT/'next-gate-spec.json')
        assert gate['type']=='read_only_joint_control_evidence_synthesis' and gate['new_physical_runs_authorized']==0
        assert digest(PARENT/'next-gate-spec.json')==p['source_gate_sha256']
        for name,h in p['sources'].items():assert digest(ROOT/name)==h,name
        save(out/'protocol.json',p);save(out/'source-gate-spec.json',gate)
        save(out/'implementation-lock.json',{'files':{name:digest(ROOT/name) for name in p['implementation_files']},'locked_before_diagnostics':True})
        identities,density=state_identities()
        save(out/'results.json',{'state_error_identities':identities,'saved_density_diagnostics':density,
            'budget_diagnostics':budgets(),'new_physical_fields_or_solves':0,'new_software_tests':0,'recorded_checks_total':508,
            'old_decisions_unchanged':True,'stage3':'incomplete','fluid_orc_enabled':False,'kappa_computed':False})
        save(out/'evidence-ledger.json',evidence(p));save(out/'preservation.json',preserve())
        for name,h in p['sources'].items():assert digest(ROOT/name)==h,name
        elapsed=time.monotonic()-start;rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024
        assert elapsed<120 and rss<1024
        save(out/'execution.json',{'completed':True,'elapsed_seconds':elapsed,'peak_RSS_MiB':rss,
            'new_physical_fields_or_solves':0,'new_software_tests':0,'old_tests_rerun':0,'published_or_pushed':False})
        print('Saved K64 identities,128 existing density arrays and old/new budgets summarized; Stage3 evidence updated without new physics.')
    except BaseException as exc:
        save(out/'failed-execution.json',{'type':type(exc).__name__,'message':str(exc),'traceback':traceback.format_exc()});raise
    finally:signal.alarm(0)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--directory',required=True);main(p.parse_args().directory)
