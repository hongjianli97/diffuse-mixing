"""Independent exact-moment and high-precision review; no physical producers."""
from pathlib import Path
from fractions import Fraction as F
import argparse,ast,math,resource,signal,time
import numpy as np
import mpmath as mp
from audit_route_decision import ROOT,read,save,digest

PARENT=ROOT/'runs/reconnect-compression-joint-control'


def audit(out):
    p=read(out/'protocol.json');r=read(out/'results.json');ledger=read(out/'evidence-ledger.json')
    for field,container in (('sources',p),('files',read(out/'implementation-lock.json'))):
        for name,h in container[field].items():assert digest(ROOT/name)==h,name
    for filename in ('joint_evidence_review.py','audit_joint_evidence_review.py'):
        tree=ast.parse((ROOT/'research/reconnect'/filename).read_text())
        for n in ast.walk(tree):
            if isinstance(n,ast.ImportFrom):assert n.module in ('pathlib','fractions','audit_route_decision')
            if isinstance(n,ast.Import):assert {x.name for x in n.names}<={'argparse','ast','math','resource','signal','time','traceback','numpy','mpmath'}
    ex=read(out/'execution.json');assert ex['completed'] and ex['elapsed_seconds']<120 and ex['peak_RSS_MiB']<1024
    with np.load(PARENT/'comparison-inputs.npz',allow_pickle=False) as z:
        m=list(map(F,z['mass'].tolist()));c=list(map(F,z['c0'].tolist()));refs=[list(map(F,a.tolist())) for a in z['reference']]
    dot=lambda a,b:sum((w*x*y for w,x,y in zip(m,a,b)),F(0));sm=lambda a:sum((w*x for w,x in zip(m,a)),F(0))
    M=sum(m,F(0));var=lambda a:dot(a,a)-sm(a)**2/M;V=var(c);sub=lambda a,b:[x-y for x,y in zip(a,b)]
    states={}
    for source in ('oracle','ledger'):
        with np.load(PARENT/source/'k64/trajectory-states.npz',allow_pickle=False) as z:states[source]=[list(map(F,a.tolist())) for a in z['states'][::32]]
    exact=[]
    for k,t in ((1,.025),(2,.05)):
        o,u=states['oracle'][k],states['ledger'][k];e=sub(o,refs[k]);d=sub(u,o);eo,es=dot(e,e),dot(sub(u,refs[k]),sub(u,refs[k]))
        D=dot(d,d);C=2*dot(e,d);md=sm(d);Q=dot(o,d)-sm(o)*md/M;Qc=D-md*md/M
        assert es-eo==C+D and var(u)-var(o)==2*Q+Qc and D>0 and Qc>=0
        exact.append({'time':t,'cross':C/V,'positive':D/V,'squared':(es-eo)/V,
            'hc':-2*Q/V,'hq':-Qc/V,'dh':-(var(u)-var(o))/V,'mass':md,'mean_square':md*md/M,
            'eo':eo,'es':es,'ed':dot(e,d),'D':D})
    density_exact=[]
    for row in r['saved_density_diagnostics']:
        step=row['step'];assert row['midpoint_time']==float(F(2*step+1,2560))
        pair=[]
        for source in ('oracle','ledger'):
            with np.load(PARENT/f'{source}/k64/density-step{step:02d}.npz',allow_pickle=False) as z:pair.append(list(map(F,z['rho'].tolist())))
        o,u=pair;assert min(o)>0 and min(u)>0;d=sub(u,o);rel=[a/b for a,b in zip(d,o)]
        density_exact.append((row,{'linf':max(map(abs,rel)),'norm2':dot(d,d)/dot(o,o),
            'bias':sm(d)/M,'lo':min(rel),'hi':max(rel)}))
    new=read(PARENT/'results.json');old=read(ROOT/'runs/reconnect-compression-density-calibration/results.json');reports=[]
    for digits in (80,100):
        with mp.workdps(digits):
            f=lambda x:mp.mpf(F(x).numerator)/F(x).denominator
            maximum=mp.mpf(0)
            def close(a,b):
                nonlocal maximum
                delta=abs(a-f(b));maximum=max(maximum,delta)
                assert delta<=64*np.finfo(float).eps*max(1,abs(a))
            for x,row in zip(exact,r['state_error_identities']):
                assert row['time']==x['time']
                for src,dst in [('cross','twice_error_cross'),('positive','state_difference_square'),('squared','squared_error_change'),
                    ('hc','H_negative_twice_centered_cross'),('hq','H_negative_centered_square'),('dh','signed_H_change'),
                    ('mass','raw_source_mass_difference'),('mean_square','mean_square_term_retained')]:close(f(x[src]),row[dst])
                change=(mp.sqrt(f(x['es']))-mp.sqrt(f(x['eo'])))/mp.sqrt(f(V))
                close(change,row['relative_error_change_from_identity'])
                close(f(x['ed'])/mp.sqrt(f(x['eo'])*f(x['D'])),row['error_update_cosine'])
                assert not row['density_accuracy_implied_by_concentration_improvement']
            for row,x in density_exact:
                for a,key in ((f(x['linf']),'relative_point_Linf'),(mp.sqrt(f(x['norm2'])),'relative_mass_L2'),
                    (f(x['bias']),'mass_weighted_signed_bias'),(f(x['lo']),'minimum_signed_relative_error'),(f(x['hi']),'maximum_signed_relative_error')):close(a,row[key])
            terminal=[]
            for row in r['budget_diagnostics']:
                t,metric=row['time'],row['metric'];n=next(x for x in new['source_comparisons'] if x['time']==t)
                prev=next(x for x in old['source_comparisons'] if x['source']=='ledger' and x['time']==t)
                parts={key:f(v) for key,v in n['concentration_parts_absolute' if metric=='concentration' else 'H_parts'].items()}
                den=f(new['common_sigma64']) if metric=='concentration' else mp.mpf(1)
                refs=('four_reference_space','four_reference_time') if metric=='concentration' else ('two_reference_space_same_basis','two_reference_time_same_basis')
                times=('ledger_32_64','oracle_32_64');budget=mp.fsum(parts.values())/den;gain=-f(n[metric+'_error_change'])
                cat={'candidate_time':mp.fsum(parts[k] for k in times)/den,'reference_space':parts[refs[0]]/den,
                    'reference_time':parts[refs[1]]/den,'initial_bridge':parts['two_initial_bridge']/den,'identity':parts['same_N64_projection_identity']/den,
                    'numerical':mp.fsum(v for k,v in parts.items() if k not in (*refs,*times,'two_initial_bridge','same_N64_projection_identity'))/den}
                zero_ref=mp.fsum(v for k,v in parts.items() if k not in refs)/den
                zero_time=mp.fsum(v for k,v in parts.items() if k not in times)/den
                for a,key in ((budget,'full_budget'),(gain,'observed_gain'),(gain-budget,'margin_gain_minus_budget'),(budget/gain,'budget_over_gain'),
                    (zero_ref,'fixed_gain_zero_reference_remaining'),(zero_time,'fixed_gain_zero_candidate_time_remaining'),
                    (budget/f(prev[metric+'_indicator']),'new_over_old_budget'),(gain/(-f(prev[metric+'_error_change'])),'new_over_old_observed_gain')):close(a,row[key])
                for key,v in cat.items():close(v,row['categories'][key]);close(100*v/budget,row['category_percent'][key])
                assert row['unchanged_decision']==n[metric+'_decision']
                assert row['old_case']=={'decision':prev[metric+'_decision'],'budget':prev[metric+'_indicator'],'gain':-prev[metric+'_error_change']}
                assert not row['old_case_rejudged'] and not row['hypothetical_remainders_used_for_acceptance'] and not row['future_gain_assumed_fixed']
                terminal.append({'time':t,'metric':metric,'full_budget':mp.nstr(budget,65),'zero_reference_over_gain':mp.nstr(zero_ref/gain,65),
                    'zero_time_over_gain':mp.nstr(zero_time/gain,65),'classification':row['unchanged_decision']})
            reports.append({'digits':digits,'maximum_float_difference':float(maximum),'rows':terminal})
    assert reports[0]['rows']==reports[1]['rows']
    for q in ledger['original_requirements']:assert (ROOT/q['path']).read_text().splitlines()[q['line']-1]==q['text']
    old_ledger=read(ROOT/'runs/reconnect-stage3-evidence/evidence-ledger.json')
    assert ledger['unchanged_prior_cases']==old_ledger['case_rows']
    for case in ledger['unchanged_prior_cases']:
        assert digest(ROOT/case['source'])==case['source_sha256'] and read(ROOT/case['source'])[case['flag']]==case['expected']
    for item in p['latest_cases']:
        row=ledger['latest_evidence'][item['id']];source=read(ROOT/item['path'])
        assert row['path']==item['path'] and row['sha256']==digest(ROOT/item['path'])
        assert row['saved_fields']=={key:source[key] for key in item['fields']}
    assert ledger['coverage']==p['coverage'] and not ledger['ledger_beating_oracle_is_required']
    assert not ledger['new_density_accuracy_threshold_added'] and not ledger['DNS_T5_T6_v080_or_ORC_added_as_prerequisite']
    assert r['recorded_checks_total']==508 and r['new_physical_fields_or_solves']==r['new_software_tests']==0
    assert r['stage3']=='incomplete' and not r['fluid_orc_enabled']
    return {'passed':True,'exact_state_identities':2,'existing_density_fields_verified':128,
        'budget_rows_verified':4,'unchanged_prior_cases_verified':len(ledger['unchanged_prior_cases']),
        'latest_case_records_verified':len(p['latest_cases']),'precision':reports,
        'no_physical_producer_imports_or_calls':True,'new_software_tests':0,'recorded_checks_total':508,
        'stage3':'incomplete','fluid_orc_enabled':False}


def main(directory,write=False):
    out=Path(directory).resolve();start=time.monotonic()
    def alarm(*_):raise TimeoutError('read-only120second audit cap')
    signal.signal(signal.SIGALRM,alarm);signal.alarm(120)
    try:
        findings=audit(out);elapsed=time.monotonic()-start;rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024
        assert elapsed<120 and rss<1024
        if write:save(out/'audit.json',{'passed':True,'findings':findings,'elapsed_seconds':elapsed,'peak_RSS_MiB':rss})
        else:assert read(out/'audit.json')['findings']==findings
        print('Exact saved-state identities,128 density fields,80/100-digit budgets and source-based evidence ledger verified.')
    finally:signal.alarm(0)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--directory',required=True);p.add_argument('--write',action='store_true');a=p.parse_args();main(a.directory,a.write)
