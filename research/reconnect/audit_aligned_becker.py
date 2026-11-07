"""Independent saved complete-child sums and 80/100-digit budget, no PDE rerun."""
from pathlib import Path
from fractions import Fraction as F
from decimal import Decimal,localcontext
import argparse,json,math,resource,time
import numpy as np
from audit_route_decision import ROOT,read,save,digest


def arrays(path):
    with np.load(path,allow_pickle=False) as z:return {k:z[k] for k in z.files}


def Q(v):return F(float(v))


def near(value,expected,ulps=16):
    if expected==0:assert value==0;return
    lo=hi=float(expected)
    for _ in range(ulps):lo=math.nextafter(lo,-math.inf);hi=math.nextafter(hi,math.inf)
    assert Decimal.from_float(lo)<=value<=Decimal.from_float(hi),(str(value),expected)


def compute(directory):
    out=Path(directory);r=read(out/'results.json');protocol=read(out/'protocol.json');lock=read(out/'implementation-lock.json')
    assert read(out/'execution.json')['first_attempt_passed']
    for rel,h in lock['files'].items():assert digest(ROOT/rel)==h,rel
    for item in protocol['sources']:assert digest(ROOT/item['path'])==item['sha256'],item['path']
    for item in read(out/'source-copies.json'):assert digest(out/item['copy'])==item['sha256']
    col=arrays(out/'source/column.npz');material=arrays(out/'source/material.npz')
    old=read(out/'source/results.json');prior=read(out/'source/uniform-reference-results.json');contract=read(out/'mesh-contract.json')
    mc=list(map(Q,col['mass']));parent=list(map(Q,col['mass_edges']));total=parent[-1]-parent[0]
    counts=[math.ceil(v/(total/16384)) for v in mc];assert counts==contract['counts']
    assert all(b-a==m for a,b,m in zip(parent[:-1],parent[1:],mc))
    all_refs=[];exact_gaps=[];initial_round=[];terminal_round=[];geometric_initial=[];geometric_final=[];checks=[];edges_by_level={}
    for level,k in ((0,8192),(1,8192),(1,16384)):
        name=f'l{level}-k{k}';z=arrays(out/f'reference-{name}.npz');raw=arrays(out/f'raw-{name}.npz');inp=arrays(out/f'input-{name}.npz');g=read(out/f'guard-{name}.json')
        for key in ('mass_edges','mass','c0','c'):np.testing.assert_array_equal(raw[key],z[key])
        for key in ('mass_edges','mass','c0'):np.testing.assert_array_equal(inp[key],z[key])
        assert np.array_equal(np.diff(z['mass_edges']),z['mass']) and np.isfinite(z['c']).all()
        if level in edges_by_level:np.testing.assert_array_equal(edges_by_level[level],z['mass_edges'])
        else:edges_by_level[level]=z['mass_edges']
        assert g['steps_completed']==g['linear_solves']==g['coefficient_evaluations']==k
        assert 0<=g['steps_without_sufficient_positivity']<=k and g['positivity_policy']=='monitor' and not g['states_clipped_or_normalized']
        assert g['max_relative_mass_drift']<=1e-9 and g['min_all_times']>=g['initial_min']-1e-10 and g['max_all_times']<=g['initial_max']+1e-10
        assert z['c'].min()>=g['min_all_times'] and z['c'].max()<=g['max_all_times']
        current=0;initial_sums=[];final_sums=[];ig=[];fg=[];closure=[]
        for i,(left,right) in enumerate(zip(parent[:-1],parent[1:])):
            count=counts[i]*2**level;si=sf=di=df=sm=F(0)
            for j in range(count):
                idx=current+j;elo=Q(z['mass_edges'][idx]);ehi=Q(z['mass_edges'][idx+1])
                assert float(left+(right-left)*j/count)==float(elo)
                assert float(left+(right-left)*(j+1)/count)==float(ehi)
                mass=Q(z['mass'][idx]);v0=Q(z['c0'][idx]);v=Q(z['c'][idx]);defect=abs(mass-(ehi-elo))
                si+=mass*v0;sf+=mass*v;sm+=mass;di+=defect*abs(v0);df+=defect*abs(v)
            current+=count;initial_sums.append(si/mc[i]);final_sums.append(sf/mc[i]);ig.append(di/mc[i]);fg.append(df/mc[i]);closure.append(sm-mc[i])
        assert current==len(z['mass'])
        qi=[v-Q(c) for v,c in zip(initial_sums,col['c0'])];ri=[Q(c)-v for c,v in zip(z['initial_projected'],initial_sums)];rf=[Q(c)-v for c,v in zip(z['projected'],final_sums)]
        for key,values in [('exact_projected_rounded',final_sums),('exact_initial_projected_rounded',initial_sums),
          ('exact_initial_mean_gap',qi),('initial_aggregation_rounding',ri),('terminal_aggregation_rounding',rf),('initial_geometry_bound',ig),('terminal_geometry_bound',fg)]:
            np.testing.assert_array_equal(z[key],np.array(list(map(float,values))))
        # Independent endpoint mass accounting; diagnostics alone are not all-time audit.
        fm=list(map(Q,z['mass']));f0=list(map(Q,z['c0']));fc=list(map(Q,z['c']))
        mass0=sum((a*b for a,b in zip(fm,f0)),F(0));mass1=sum((a*b for a,b in zip(fm,fc)),F(0))
        exact_drift=abs(mass1-mass0)/abs(mass0);assert exact_drift<=F(1,10**9)
        checks.append({'level':level,'cells':len(fm),'steps':k,'minimum_RHS_diagonal':g['minimum_RHS_diagonal'],
          'steps_without_sufficient_positivity':g['steps_without_sufficient_positivity'],
          'maximum_exact_parent_mass_closure':float(max(map(abs,closure))),'exact_terminal_mass_drift':float(exact_drift),
          'all_complete_child_sums_verified':True})
        all_refs.append(list(map(Q,z['projected'])));exact_gaps.append(qi);initial_round.append(ri);terminal_round.append(rf);geometric_initial.append(ig);geometric_final.append(fg)
    assert np.array_equal(edges_by_level[0],edges_by_level[1][::2])
    m=material['mass'];c0=material['c0'];mu=float(m@c0/m.sum());E0=float(m@((c0-mu)**2));A=2*math.sqrt(float(m.sum()))*max(1.,float(np.max(abs(c0))))
    precision_rows=[]
    for digits in (80,100):
        with localcontext() as ctx:
            ctx.prec=digits
            D=lambda x:Decimal(x.numerator)/Decimal(x.denominator)
            dn=lambda v:sum((D(mi)*D(vi)**2 for mi,vi in zip(mc,v)),Decimal(0)).sqrt()
            space=dn([a-b for a,b in zip(all_refs[0],all_refs[1])]);temporal=dn([a-b for a,b in zip(all_refs[1],all_refs[2])])
            near(space,r['reference_space_indicator_absolute']);near(temporal,r['reference_time_indicator_absolute'])
            parts={}
            for key,value in old['time_control']['components_absolute'].items():
                if key=='twice_saved_reference_space_time':parts['twice_aligned_reference_space_time']=2*(space+temporal)
                else:parts[key]=Decimal.from_float(value)
            parts['twice_all_initial_mean_bridges']=2*sum((dn(q)+dn(rr) for q,rr in zip(exact_gaps,initial_round)),Decimal(0))
            parts['twice_all_terminal_aggregation_bridges']=2*sum((dn(q) for q in terminal_round),Decimal(0))
            parts['twice_all_initial_and_terminal_geometry_bridges']=2*sum((dn(a)+dn(b) for a,b in zip(geometric_initial,geometric_final)),Decimal(0))
            for key,value in parts.items():near(value,r['positive_budget_components_absolute'][key])
            b=sum(parts.values());hb=(Decimal.from_float(A)*b+b*b)/Decimal.from_float(E0)
            near(b,r['budget_absolute']);near(hb,r['H_indicator_unchanged_formula'])
            statepass=Decimal.from_float(prior['state_decrease_absolute'])>b;hpass=Decimal.from_float(prior['H_decrease_unchanged'])>hb
            assert statepass==r['state_decrease_resolved'] and hpass==r['H_decrease_resolved']
            precision_rows.append({'digits':digits,'space':str(space),'time':str(temporal),'initial_bridge_term':str(parts['twice_all_initial_mean_bridges']),
              'total_budget':str(b),'H_indicator':str(hb),'state_decrease_resolved':statepass,'H_decrease_resolved':hpass})
    # Independent rational H of saved candidate endpoints; retain original budget.
    mass=list(map(Q,m));initial_mean=list(map(Q,c0));M=sum(mass,F(0))
    def variance(values):
        mean=sum((a*b for a,b in zip(mass,values)),F(0))/M
        return sum((a*(b-mean)**2 for a,b in zip(mass,values)),F(0))
    V0=variance(initial_mean);hc=[]
    for i,K in enumerate((4,8,16)):
        u=arrays(out/f'source/k{K}-states.npz')['states'][-1];h=1-variance(list(map(Q,u)))/V0
        saved=old['observations'][i]['H'];num=old['observations'][i]['numeric_indicator_relative']*old['original_column_initial_fluctuation_norm'];allowance=(A*num+num*num)/E0
        assert abs(float(h)-saved)<=allowance
        assert r['observations'][i]['saved_H']==saved
        hc.append({'K':K,'exact_H_of_saved_arrays_rounded':float(h),'saved_H':saved,'original_numeric_H_allowance':allowance})
    assert r['new_temporal_comparison_resolved']==(r['state_decrease_resolved'] and r['H_decrease_resolved'] and old['all_complete_sequence_controls_passed'])
    assert r['old_original_decision_preserved']==old['comparison_decision'] and r['old_uniform_reference_decision_preserved']==prior['decision']
    assert r['new_reference_PDE_solves']==3 and r['new_candidate_transient_runs']==0 and not r['old_results_reclassified']
    assert r['stage3_physical_acceptance']=='incomplete' and not r['fluid_orc_enabled']
    return {'passed':True,'exact_complete_child_observation_and_geometry_verified':True,'nested_edges_verified':True,
      'reference_endpoint_checks':checks,'precision_checks':precision_rows,'saved_H_checks':hc,
      'new_PDE_solves_in_audit':0,'candidate_actions_in_audit':0,'new_software_checks':0,'recorded_checks_total':502,
      'scope':'saved endpoints and guard receipts; no independent all-time trajectory or certified PDE error bound',
      'stage3_physical_acceptance':'incomplete','fluid_orc_enabled':False,'results_sha256':digest(out/'results.json')}


def main(directory,write=False):
    start=time.monotonic();findings=compute(directory);out=Path(directory)
    if write:
        for rel,h in read(ROOT/'runs/reconnect-aligned-reference/sha256.json').items():assert digest(ROOT/rel)==h,rel
        save(out/'audit.json',{'passed':True,'first_attempt_passed':True,'findings':findings,'elapsed_seconds':time.monotonic()-start,
          'peak_RSS_MiB':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,'auditor_sha256':digest(Path(__file__))})
    else:assert findings==read(out/'audit.json')['findings']
    print(json.dumps(findings,indent=2),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--directory',required=True);parser.add_argument('--write',action='store_true')
    args=parser.parse_args();main(args.directory,args.write)
