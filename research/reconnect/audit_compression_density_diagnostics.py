"""Exact rational identities and80/100digit saved-budget audit; no producer calls."""
from pathlib import Path
from fractions import Fraction as F
import argparse,ast,math,resource,signal,time
import numpy as np
import mpmath as mp
from audit_route_decision import ROOT,read,save,digest

PARENT=ROOT/'runs/reconnect-compression-density-calibration'
ORIGINAL=ROOT/'runs/reconnect-compression-union64'


def audit(directory):
    out=Path(directory);p=read(out/'protocol.json');lock=read(out/'implementation-lock.json');result=read(out/'results.json')
    for name,h in p['sources'].items():assert digest(ROOT/name)==h,name
    for name,h in lock['files'].items():assert digest(ROOT/name)==h,name
    assert digest(PARENT/'next-gate-spec.json')==p['source_gate_sha256']
    ex=read(out/'execution.json');assert ex['completed'] and ex['elapsed_seconds']<120 and ex['peak_RSS_MiB']<1024
    for name in ('compression_density_diagnostics.py','audit_compression_density_diagnostics.py'):
        tree=ast.parse((ROOT/'research/reconnect'/name).read_text())
        modules={n.module for n in ast.walk(tree) if isinstance(n,ast.ImportFrom)}
        assert modules<={'pathlib','fractions','audit_route_decision'}
        for n in ast.walk(tree):
            if isinstance(n,ast.Import):assert {a.name for a in n.names}<={'argparse','ast','math','resource','signal','time','traceback','numpy','mpmath'}
    # Independent reads and exact moments of the stored binary64 values.
    ff=lambda v:F(float(v))
    with np.load(ORIGINAL/'union-definition.npz',allow_pickle=False) as z:
        m=list(map(ff,z['mass']));c=list(map(ff,z['c0']));data={key:[list(map(ff,row)) for row in z[key]] for key in ('reference','space','time','reference_space_coarse','reference_space_fine','reference_time_check')}
    M=sum(m,F(0));dot=lambda a,b:sum((w*x*y for w,x,y in zip(m,a,b)),F(0));mean_sum=lambda a:sum((w*x for w,x in zip(m,a)),F(0))
    variance=lambda a:dot(a,a)-mean_sum(a)**2/M
    diff=lambda a,b:[x-y for x,y in zip(a,b)]
    V0=variance(c);assert V0>0;v={};variance_cache={};runs={}
    for s in ('oracle','kernel','ledger'):
        for K in (16,32):
            folder=(ORIGINAL if s=='oracle' else PARENT/s)/f'k{K}';runs[s,K]=read(folder/'native-results.json')
            with np.load(folder/'trajectory-states.npz',allow_pickle=False) as z:
                for field in ('states','independent_states'):
                    for k in (1,2):
                        key=(s,K,k,field);v[key]=list(map(ff,z[field][k*K//2]));variance_cache[key]=variance(v[key])
    exact_rows=[]
    for row in result['signed_identities']:
        s,K=row['source'],row['K'];k=round(row['time']/.025);o=v['oracle',K,k,'states'];u=v[s,K,k,'states'];ref=data['reference'][k]
        delta=diff(u,o);eo=diff(o,ref);es=diff(u,ref);D2=dot(delta,delta);C=2*dot(eo,delta);Eo2=dot(eo,eo);Es2=dot(es,es)
        assert Es2-Eo2==C+D2
        So,Sd=mean_sum(o),mean_sum(delta);Q=dot(o,delta)-So*Sd/M;Qc=D2-Sd*Sd/M
        vo,vs=variance_cache['oracle',K,k,'states'],variance_cache[s,K,k,'states'];assert vs-vo==2*Q+Qc
        assert D2>=0 and Qc>=0 and Sd==mean_sum(u)-mean_sum(o)
        Href=1-variance(ref)/V0;Ho=1-vo/V0;Hs=1-vs/V0
        exact_rows.append({'source':s,'K':K,'time':row['time'],'squared_error_change':str((Es2-Eo2)/V0),
            'twice_error_cross_term':str(C/V0),'source_difference_squared_norm':str(D2/V0),
            'negative_twice_centered_cross':str(-2*Q/V0),'negative_centered_difference_energy':str(-Qc/V0),
            'signed_H_change':str(Hs-Ho),'source_minus_oracle_mass':str(Sd),'mean_square_term_retained':str(Sd*Sd/M),
            'Eo2':Eo2,'Es2':Es2,'Herrorchange':abs(Hs-Href)-abs(Ho-Href),'Ho':Ho,'Hs':Hs,'Href':Href})
    norm2=lambda a:dot(a,a);state_difference_norm2={};sequence_difference_norm2={}
    for s in ('oracle','kernel','ledger'):
        for k in (1,2):
            state_difference_norm2[s,k]=norm2(diff(v[s,16,k,'states'],v[s,32,k,'states']))
            for K in (16,32):sequence_difference_norm2[s,K,k]=norm2(diff(v[s,K,k,'states'],v[s,K,k,'independent_states']))
    reference_variances={name:[variance(a) for a in data[name]] for name in ('reference','reference_space_coarse','reference_space_fine','reference_time_check')}
    b0=norm2(diff(data['reference'][0],c));space=[norm2(a) for a in data['space']];time2=[norm2(a) for a in data['time']]
    original=read(PARENT/'results.json');oldrows={(r['source'],r['time']):r for r in original['source_comparisons']};records=[]
    for digits in (80,100):
        with mp.workdps(digits):
            f=lambda x:mp.mpf(F(x).numerator)/F(x).denominator
            fp=lambda x:f(ff(x));v0=f(V0);sigma=mp.sqrt(v0);mass=f(M);H=lambda vv:1-f(vv)/v0
            hm=lambda e:(2*mp.sqrt(mass)*max(1,max(map(lambda x:abs(f(x)),c)))*e+e*e)/v0
            maximum=mp.mpf(0)
            def close(x,stored):
                nonlocal maximum
                error=abs(x-fp(stored));maximum=max(maximum,error)
                assert error<=64*np.finfo(float).eps*max(1,abs(x)),(mp.nstr(x,18),stored,mp.nstr(error,10))
            signed=[]
            for exact,row in zip(exact_rows,result['signed_identities']):
                ec=row['concentration'];eh=row['H'];eo=mp.sqrt(f(exact['Eo2']))/sigma;es=mp.sqrt(f(exact['Es2']))/sigma
                for name in ('squared_error_change','twice_error_cross_term','source_difference_squared_norm'):close(f(exact[name]),ec[name])
                close(es-eo,ec['relative_error_change']);close(es-eo,ec['error_change_from_identity'])
                close(eo,ec['oracle_relative_error']);close(es,ec['source_relative_error'])
                for name in ('negative_twice_centered_cross','negative_centered_difference_energy','signed_H_change'):close(f(exact[name]),eh[name])
                close(f(exact['Herrorchange']),eh['error_change'])
                for a,b in (('Ho','oracle_H'),('Hs','source_H'),('Href','reference_H')):close(f(exact[a]),eh[b])
                close(f(exact['source_minus_oracle_mass']),row['mass']['source_minus_oracle_mass']);close(f(exact['mean_square_term_retained']),row['mass']['mean_square_term_retained'])
                signed.append({'source':row['source'],'K':row['K'],'time':row['time'],
                    'cross_term_sign':int(mp.sign(f(exact['twice_error_cross_term']))),'squared_change_sign':int(mp.sign(f(exact['squared_error_change']))),
                    'exact_concentration_identity_verified':True,'exact_centered_H_identity_verified':True})
            checks=[]
            for row in result['budget_diagnostics']:
                s=row['source'];k=round(row['time']/.025);old=oldrows[s,row['time']]
                cp={'four_reference_space':4*mp.sqrt(f(space[k])),'four_reference_time':4*mp.sqrt(f(time2[k])),
                    'two_initial_bridge':2*mp.sqrt(f(b0)),'same_N64_projection_identity':mp.mpf(0)}
                hp={'two_reference_space_same_basis':2*abs(H(reference_variances['reference_space_coarse'][k])-H(reference_variances['reference_space_fine'][k])),
                    'two_reference_time_same_basis':2*abs(H(reference_variances['reference_time_check'][k])-H(reference_variances['reference'][k])),
                    'two_initial_bridge':2*abs(H(reference_variances['reference'][0])-H(V0)),'same_N64_projection_identity':mp.mpf(0)}
                total=mp.mpf(0)
                for src in (s,'oracle'):
                    cp[src+'_16_32']=mp.sqrt(f(state_difference_norm2[src,k]));hp[src+'_16_32']=abs(H(variance_cache[src,16,k,'states'])-H(variance_cache[src,32,k,'states']))
                    for K in (16,32):
                        rr=runs[src,K]['steps'][k*K//2-1];mt=fp(rr['cumulative_machine_indicator_absolute'])+fp(rr['cumulative_tail_indicator_absolute']);total+=mt
                        cp[f'{src}_e{K}']=mt+mp.sqrt(f(sequence_difference_norm2[src,K,k]))
                        hp[f'{src}_complete_sequence_H{K}']=abs(H(variance_cache[src,K,k,'states'])-H(variance_cache[src,K,k,'independent_states']))
                hp['both_machine_tail_to_H']=hm(total)
                exact=next(r for r in exact_rows if r['source']==s and r['K']==32 and r['time']==row['time'])
                gc=(mp.sqrt(f(exact['Es2']))-mp.sqrt(f(exact['Eo2'])))/sigma;gh=f(exact['Herrorchange'])
                for metric,terms,refnames,denom,gain,oldparts in (
                    ('concentration',cp,('four_reference_space','four_reference_time'),sigma,gc,old['concentration_parts_absolute']),
                    ('H',hp,('two_reference_space_same_basis','two_reference_time_same_basis'),mp.mpf(1),gh,old['H_parts'])):
                    target=row[metric];full=mp.fsum(terms.values())/denom;floor=mp.fsum(v for name,v in terms.items() if name not in refnames)/denom
                    for name,val in terms.items():assert val>=0;close(val,oldparts[name]);close(val/denom,target['parts_in_comparison_units'][name])
                    close(full,target['full_indicator_unchanged']);close(gain,target['observed_error_change_unchanged']);close(floor,target['hypothetical_zero_reference_space_time_floor'])
                    decide=lambda x,b:'error_larger_resolvably' if x>b else ('error_smaller_resolvably' if x < -b else 'unresolved')
                    assert decide(gain,full)==target['original_decision']==old[metric+'_decision']
                    assert bool(abs(gain)>floor)==target['reference_indicator_reduction_alone_sufficient_at_fixed_observed_change']
                    assert not target['hypothetical_floor_used_for_acceptance'] and not target['replacement_reference_gain_assumed_unchanged']
                    assert set(target['retained_terms'])==set(terms)-set(refnames) and target['removed_only_in_hypothetical_floor']==list(refnames)
                    # Percentages/ratios are views of the ORIGINAL saved rounded scalars.
                    rounded_denom=fp(original['common_sigma64']) if metric=='concentration' else mp.mpf(1)
                    rounded={name:fp(val)/rounded_denom for name,val in oldparts.items()};rounded_full=fp(target['full_indicator_unchanged'])
                    rounded_floor=mp.fsum(val for name,val in rounded.items() if name not in refnames);saved_gain=abs(fp(target['observed_error_change_unchanged']))
                    close(rounded_floor/saved_gain,target['floor_over_observed_change']);close(saved_gain-rounded_floor,target['signed_reference_allowance_at_fixed_observed_change'])
                    categories={'time_16_32':rounded[s+'_16_32']+rounded['oracle_16_32'],'reference_space':rounded[refnames[0]],
                        'reference_time':rounded[refnames[1]],'initial_bridge':rounded['two_initial_bridge'],'identity_projection':rounded['same_N64_projection_identity'],
                        'numerical_sequences_machine_tail':mp.fsum(v for name,v in rounded.items() if name not in (*refnames,s+'_16_32','oracle_16_32','two_initial_bridge','same_N64_projection_identity'))}
                    for name,val in categories.items():close(val,target['category_values'][name]);close(100*val/rounded_full,target['category_percent_of_full'][name])
                    assert target['ranked_terms']==sorted(oldparts,key=lambda name:(-oldparts[name],name))
                    checks.append({'source':s,'time':row['time'],'metric':metric,'original_decision':decide(gain,full),
                        'zero_reference_floor':mp.nstr(floor,65),'fixed_gain_reference_only_feasible':bool(abs(gain)>floor)})
            records.append({'digits':digits,'exact_identities':signed,'budget_rows':checks,'maximum_float_difference':float(maximum)})
    assert records[0]['exact_identities']==records[1]['exact_identities'] and records[0]['budget_rows']==records[1]['budget_rows']
    assert len(exact_rows)==8 and len(result['budget_diagnostics'])==4 and result['recorded_checks_total']==506
    for name in ('new_density_fields','new_coefficient_packets','new_geometry_graphs','new_actual_or_independent_paths','new_reference_PDE_solves','new_software_checks'):assert result[name]==0
    exact_serializable=[{k:str(val) if isinstance(val,F) else val for k,val in row.items()} for row in exact_rows]
    return {'passed':True,'exact_saved_binary64_moment_identities':exact_serializable,'high_precision':records,
        'original_decisions_preserved':True,'no_new_density_geometry_propagation_or_reference':True,
        'recorded_checks_total':506,'stage3':'incomplete','fluid_orc_enabled':False,'kappa_computed':False}


def main(directory,write=False):
    start=time.monotonic();out=Path(directory).resolve()
    def timeout(*_):raise TimeoutError('read-only independent audit120second cap')
    signal.signal(signal.SIGALRM,timeout);signal.alarm(120)
    try:
        findings=audit(out);elapsed=time.monotonic()-start;peak=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024
        assert elapsed<120 and peak<1024
        if write:save(out/'audit.json',{'passed':True,'findings':findings,'elapsed_seconds':elapsed,'peak_RSS_MiB':peak})
        else:assert findings==read(out/'audit.json')['findings']
        print('Eight exact saved-state identities;80/100-digit full budgets,reference-zero floors and original classifications verified.',flush=True)
    finally:signal.alarm(0)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--directory',required=True);p.add_argument('--write',action='store_true');a=p.parse_args();main(a.directory,a.write)
