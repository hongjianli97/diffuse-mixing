"""Saved endpoints and exact projection/budget audit; never rerun a PDE."""
from pathlib import Path
from fractions import Fraction as F
from decimal import Decimal,localcontext
import argparse,math,resource,time
import numpy as np
from audit_route_decision import ROOT,read,save,digest


def arrays(path):
    with np.load(path,allow_pickle=False) as z:return {k:z[k] for k in z.files}


def exact_project(edges,values,target):
    # Sweep once across stored fine intervals, independently of searchsorted projection.
    ee=list(map(lambda x:F(float(x)),edges));tt=list(map(lambda x:F(float(x)),target));vv=list(map(lambda x:F(float(x)),values))
    j=0;result=[]
    for lo,hi in zip(tt[:-1],tt[1:]):
        while ee[j+1]<=lo:j+=1
        k=j;total=F(0)
        while k<len(vv) and ee[k]<hi:
            left=max(lo,ee[k]);right=min(hi,ee[k+1])
            if right>left:total+=(right-left)*vv[k]
            k+=1
        result.append(float(total/(hi-lo)))
    return np.array(result)


def audit(out):
    started=time.monotonic();out=Path(out).resolve();r=read(out/'results.json');protocol=read(out/'protocol.json');lock=read(out/'implementation-lock.json')
    ex=read(out/'execution.json');assert not ex['first_attempt_passed'] and ex['original_main_exit_code']==1 and ex['completion_exit_code']==0
    assert digest(ROOT/'research/reconnect/becker_reference_completion.py')==read(out/'completion-lock.json')['code_sha256']
    for rel,h in read(out/'pre-completion-sha256.json').items():assert digest(out/rel)==h
    for path,key in [('research/reconnect/becker_reference_check.py','code_sha256'),('research/reconnect/becker_reference_protocol.json','protocol_sha256'),
      ('research/rebuild/reference.py','reference_source_sha256'),('research/rebuild/becker.py','Becker_source_sha256')]:assert digest(ROOT/path)==lock[key]
    for item in protocol['sources']:assert digest(ROOT/item['path'])==item['sha256']
    for item in read(out/'source-copies.json'):assert digest(ROOT/item['source'])==digest(out/item['copy'])==item['sha256']
    col=arrays(out/'source/column.npz');data=arrays(out/'source/material.npz');old=read(out/'source/results.json');mc=col['mass'];m=data['mass'];c0=data['c0']
    norm=lambda mass,v:float(np.sqrt(mass@(v*v)));refs=[];bridges=[];initial=[];maxfinalmass=0.
    for n,k in ((16384,8192),(32768,8192),(32768,16384)):
        z=arrays(out/f'reference-n{n}-k{k}.npz');raw=arrays(out/f'raw-reference-n{n}-k{k}.npz');guard=read(out/f'guard-n{n}-k{k}.json')
        for name in ('mass_edges','c0','c'):np.testing.assert_array_equal(z[name],raw[name])
        assert z['c'].shape==(n,) and np.isfinite(z['c']).all() and guard['cells']==n and guard['steps']==k
        e=z['mass_edges'];np.testing.assert_array_equal(e,np.linspace(0,col['mass_edges'][-1],n+1))
        assert guard['max_relative_mass_drift']<=1e-9 and guard['min_all_times']>=guard['initial_min']-1e-10 and guard['max_all_times']<=guard['initial_max']+1e-10
        assert z['c'].min()>=guard['min_all_times'] and z['c'].max()<=guard['max_all_times']
        mass0=(e[-1]/n)*z['c0'].sum();drift=abs((e[-1]/n)*z['c'].sum()-mass0)/abs(mass0)
        assert drift<=guard['max_relative_mass_drift'];maxfinalmass=max(maxfinalmass,drift)
        p=exact_project(e,z['c'],col['mass_edges']);ip=exact_project(e,z['c0'],col['mass_edges'])
        np.testing.assert_array_equal(p,z['projected']);np.testing.assert_array_equal(ip,z['initial_projected'])
        primitive=np.r_[0.,np.cumsum(np.diff(e)*z['c'])];q=np.diff(np.interp(col['mass_edges'],e,primitive))/mc
        np.testing.assert_array_equal(q,z['primitive_projected']);refs.append(p);bridges.append(norm(mc,p-q));initial.append(norm(mc,ip-col['c0']))
    space=norm(mc,refs[0]-refs[1]);temporal=norm(mc,refs[1]-refs[2])
    assert space==r['reference_space_indicator_absolute'] and temporal==r['reference_time_indicator_absolute']
    parts={}
    for name,value in old['time_control']['components_absolute'].items():
        if name=='twice_saved_reference_space_time':parts['twice_refined_reference_space_time']=2*(space+temporal)
        else:parts[name]=value
    parts['twice_all_new_projection_bridges']=2*sum(bridges);parts['twice_all_new_initial_mean_bridges']=2*sum(initial)
    assert parts==r['positive_budget_components_absolute'] and all(v>=0 for v in parts.values())
    b=sum(parts.values());mu=float(m@c0/m.sum());E0=float(m@((c0-mu)**2));A=2*math.sqrt(float(m.sum()))*max(1.,float(np.max(abs(c0))))
    hb=(A*b+b*b)/E0;assert hb==r['H_indicator_unchanged_formula']
    decimal_checks=[]
    for precision in (80,100):
        with localcontext() as ctx:
            ctx.prec=precision;D=lambda x:Decimal.from_float(float(x));value=(D(A)*D(b)+D(b)**2)/D(E0)
            lo=hi=hb
            for _ in range(8):lo=math.nextafter(lo,-math.inf);hi=math.nextafter(hi,math.inf)
            assert D(lo)<=value<=D(hi);decimal_checks.append({'precision':precision,'H_polynomial_of_saved_binary64_inputs':str(value)})
    states=[arrays(out/f'source/k{K}-states.npz')['states'][-1] for K in (4,8,16)]
    H=lambda u:1-float(m@((u-float(m@u/m.sum()))**2))/E0
    hs=[v['H'] for v in old['observations']]
    gap=abs(hs[0]-hs[1])-abs(hs[1]-hs[2])
    exact_initial=sum((F(float(a)) for a in m),F(0))
    def exact_H(u):
        mf=list(map(lambda a:F(float(a)),m));cf=list(map(lambda a:F(float(a)),u));zf=list(map(lambda a:F(float(a)),c0))
        def variance(v):
            mean=sum((a*c for a,c in zip(mf,v)),F(0))/exact_initial
            return sum((a*(c-mean)**2 for a,c in zip(mf,v)),F(0))
        return 1-variance(cf)/variance(zf)
    h_rounding=[]
    for i,u in enumerate(states):
        value=exact_H(u);allowance=old['observations'][i]['numeric_indicator_relative']*old['original_column_initial_fluctuation_norm']
        old_h_allowance=(A*allowance+allowance**2)/E0
        assert abs(float(value)-hs[i])<=old_h_allowance and abs(float(value)-H(u))<=old_h_allowance
        h_rounding.append({'K':(4,8,16)[i],'exact_H_of_saved_arrays':str(value),'saved_H':hs[i],'current_H':H(u),
          'current_minus_saved':H(u)-hs[i],'original_numeric_H_allowance':old_h_allowance})
    sg=norm(m,states[0]-states[1])-norm(m,states[1]-states[2])
    assert gap==r['H_decrease_unchanged'] and sg==r['state_decrease_absolute']
    assert r['new_temporal_comparison_resolved']==(gap>hb and sg>b and old['all_complete_sequence_controls_passed'])
    assert r['original_case_decision_retained']==old['comparison_decision']=='moving_Becker_control_failed_or_unresolved'
    pres=read(out/'preservation.json');assert len(pres['formal_manifests_verified'])==103
    # The cumulative latest seal covers all immutable scientific sources of this check.
    parent=ROOT/'runs/reconnect-becker-budget-review/sha256.json'
    for rel,h in read(parent).items():assert digest(ROOT/rel)==h,rel
    save(out/'audit.json',{'passed':True,'first_attempt_passed':True,'elapsed_seconds':time.monotonic()-started,
      'peak_RSS_MiB':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,'auditor_sha256':digest(Path(__file__)),
      'independent_swept_exact_projection_arrays_bitwise_equal':True,'positive_budget_and_original_H_verified':True,
      'Decimal_checks':decimal_checks,'maximum_final_reference_relative_mass_drift':maxfinalmass,
      'saved_H_cross_environment_rounding_diagnostics':h_rounding,'original_main_failure_and_snapshot_preserved':True,
      'new_PDE_solves_in_audit':0,'candidate_actions_in_audit':0,'new_software_tests':0,'recorded_checks_total':500,
      'audit_scope':'saved endpoint arrays, guard records, projection and budget arithmetic; not a second independent PDE trajectory or a certified continuum error bound',
      'source_case_reclassified':False,'stage3_physical_acceptance':'incomplete','fluid_orc_enabled':False})
    print('Three saved reference endpoints, independent exact projections, positive bridges and original H verified; no PDE or candidate rerun.',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--directory',required=True);a=p.parse_args();audit(a.directory)
