"""Independent rational moment identities and Decimal budget-floor audit; no solver."""
from pathlib import Path
from fractions import Fraction as F
from decimal import Decimal,localcontext
import argparse,ast,json,math,resource,time
import numpy as np
from audit_route_decision import ROOT,read,save,digest


def arr(path):
    with np.load(path,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}


def Q(x):return F(float(x))
def moment_variance(m,u):
    return sum((a*b*b for a,b in zip(m,u)),F(0))-sum((a*b for a,b in zip(m,u)),F(0))**2/sum(m,F(0))
def exact_saved(obj):
    q=F(obj['exact']);assert float(q)==obj['float'];return q
def near(value,saved):
    expected=Decimal.from_float(float(saved));assert abs(value-expected)<=Decimal.from_float(16*math.ulp(float(saved))),(str(value),saved)


def main(directory,write=False):
    started=time.monotonic();out=Path(directory);src=out/'source';r=read(out/'results.json');old=read(src/'results.json');p=read(out/'protocol.json')
    assert digest(out/'source-gate-spec.json')==p['source_gate_sha256']
    for name,h in read(out/'implementation-lock.json')['files'].items():assert digest(ROOT/name)==h,name
    for item in read(out/'source-copies.json'):assert digest(out/item['copy'])==item['sha256']==p['sources'][item['source']]
    for filename in ('becker_delta004_budget_review.py','audit_becker_delta004_budget.py'):
        tree=ast.parse((ROOT/'research/reconnect'/filename).read_text())
        modules=[]
        for node in ast.walk(tree):
            if isinstance(node,ast.Import):modules.extend(n.name for n in node.names)
            elif isinstance(node,ast.ImportFrom):modules.append(node.module)
        assert not any(any(x in mod for x in ('becker_point','aligned_mass','voronoi','scipy','becker_delta004_core')) for mod in modules)
    data=arr(src/'n32/material.npz');m=list(map(Q,data['mass']));c0=list(map(Q,data['c0']));M=sum(m,F(0));V0=moment_variance(m,c0)
    states={k:list(map(Q,arr(src/f'n32/k{k}/trajectory-states.npz')['states'][-1])) for k in (8,16,32)}
    h={k:1-moment_variance(m,u)/V0 for k,u in states.items()}
    for k in h:
        assert h[k]==exact_saved(r['exact_saved_H'][str(k)])
        assert abs(float(h[k])-old['H_by_K'][str(k)])<=64*np.finfo(float).eps
    groups={k:exact_saved(v) for k,v in r['positive_B_groups'].items()};expected={k:F(0) for k in ('machine_allowance','raw_tail','measured_sequence_difference')}
    squared=[]
    for k in (8,16,32):
        last=read(src/f'n32/k{k}/native-results.json')['steps'][-1];v=list(map(Q,arr(src/f'n32/k{k}/trajectory-states.npz')['independent_states'][-1]))
        expected['machine_allowance']+=2*Q(last['cumulative_machine_indicator_absolute'])
        expected['raw_tail']+=2*Q(last['cumulative_tail_indicator_absolute'])
        expected['measured_sequence_difference']+=2*Q(old['numeric_by_K'][str(k)]['complete_sequence_difference'])
        squared.append(sum((a*(b-c)**2 for a,b,c in zip(m,states[k],v)),F(0)))
    ref=old['reference_indicators']['components_absolute']
    expected.update(reference_space=2*Q(ref['space']),reference_time=2*Q(ref['time']),
        reference_initial_rounding_geometry_bridges=2*sum((Q(v) for k,v in ref.items() if k not in ('space','time')),F(0)),
        transverse_bridges=sum((Q(v) for k,v in old['B_parts'].items() if 'bridge' in k),F(0)))
    assert expected==groups and all(v>=0 for v in groups.values())
    B=exact_saved(r['saved_B']);residual=exact_saved(r['saved_accumulation_residual_signed'])
    assert B==Q(old['B'])==sum(groups.values(),F(0))+residual
    assert r['saved_time_control']==old['time_control'] and not r['parent_time_qualified'] and old['comparison_decision']==r['parent_decision_unchanged']
    assert r['old_error_changes_and_classifications_retained']=={'changes':old['error_changes_new_minus_old'],'classifications':old['error_change_decisions']}
    identity_records=[]
    for pair in r['H_variance_change_identity']:
        a,b=pair['from_K'],pair['to_K'];u,v=states[a],states[b]
        umean=sum((x*y for x,y in zip(m,u)),F(0))/M;vmean=sum((x*y for x,y in zip(m,v)),F(0))/M
        delta=[y-x for x,y in zip(u,v)];mean_delta=vmean-umean
        terms=[2*w*(x-umean)*(d-mean_delta)/V0 for w,x,d in zip(m,u,delta)]
        pos=sum((x for x in terms if x>0),F(0));neg=sum((x for x in terms if x<0),F(0))
        quadratic=moment_variance(m,delta)/V0
        assert pos==exact_saved(pair['linear_positive_sum_over_V0']) and neg==exact_saved(pair['linear_negative_sum_over_V0'])
        assert pos+neg==exact_saved(pair['linear_net_over_V0']) and quadratic==exact_saved(pair['quadratic_over_V0'])
        # Alternative second-moment formula, not the primary centered-square implementation.
        dh=(moment_variance(m,u)-moment_variance(m,v))/V0
        assert dh==-(pos+neg)-quadratic==exact_saved(pair['signed_exact_H_change'])
        assert (pos-neg)/abs(pos+neg)==exact_saved(pair['spatial_linear_cancellation_ratio'])
        assert Q(old['H_by_K'][str(b)])-Q(old['H_by_K'][str(a)])-dh==exact_saved(pair['saved_minus_exact_H_change'])
        identity_records.append({'from_K':a,'to_K':b,'exact_moment_identity_verified':True,'cancellation_ratio':float((pos-neg)/abs(pos+neg))})
    mm=data['mass'];cc=data['c0'];V=float(mm@((cc-float(mm@cc/mm.sum()))**2));A=2*math.sqrt(float(mm.sum()))*max(1.,float(max(abs(cc))))
    assert r['original_polynomial_coefficients']['A']==A and r['original_polynomial_coefficients']['V0_float']==V
    assert exact_saved(r['original_polynomial_coefficients']['V0_exact_saved_state'])==V0
    precision=[]
    for record in r['precision_checks']:
        with localcontext() as ctx:
            ctx.prec=record['digits'];D=lambda x:Decimal.from_float(float(x));R=lambda x:Decimal(x.numerator)/Decimal(x.denominator)
            poly=lambda E:(D(A)*E+E**2)/D(V)
            full=poly(R(B));machine=poly(R(groups['machine_allowance']))
            numeric=poly(sum((R(groups[k]) for k in ('machine_allowance','raw_tail','measured_sequence_difference')),Decimal(0)))
            reference=poly(sum((R(groups[k]) for k in ('reference_space','reference_time','reference_initial_rounding_geometry_bridges')),Decimal(0)))
            gap=D(old['time_control']['H_decrease']);exactgap=R(abs(h[8]-h[16])-abs(h[16]-h[32]))
            for key,value in [('full_H_indicator',full),('machine_only_lower_bound',machine),('all_numeric_only_lower_bound',numeric),
                ('reference_only_lower_bound',reference),('saved_H_decrease',gap),('exact_state_H_decrease',exactgap)]:assert Decimal(record[key])==value
            near(full,old['time_control']['H_indicator'])
            critical=Decimal(record['critical_B_for_saved_gap_diagnostic_only'])
            assert abs(poly(critical)-gap)<Decimal(10)**(-ctx.prec+10)
            assert (machine>gap)==record['machine_floor_exceeds_current_gap']
            assert (machine>D(old['time_control']['H8_16']))==record['machine_floor_exceeds_current_first_increment']
            assert (reference>gap)==record['reference_floor_exceeds_current_gap']
            assert (full>D(old['time_control']['H16_32']))==record['full_floor_exceeds_shifted_first_increment']
            assert (exactgap<=full)==record['exact_state_H_decrease_still_unresolved']
            linear=sum((D(A)*R(v)/D(V) for v in groups.values()),Decimal(0))
            for k,v in groups.items():
                assert Decimal(record['linear_H_budget_attribution'][k])==D(A)*R(v)/D(V)
                near(R(v/B),r['budget_shares'][k])
            assert abs(linear+D(A)*R(residual)/D(V)+R(B)**2/D(V)-full)<Decimal(10)**(-ctx.prec+10)
            for k,sq in zip((8,16,32),squared):near(R(sq).sqrt(),old['numeric_by_K'][str(k)]['complete_sequence_difference'])
            precision.append({'digits':ctx.prec,'full_H_indicator':str(full),'machine_floor':str(machine),'reference_floor':str(reference),
                'current_gap':str(gap),'shifted_ceiling':str(D(old['time_control']['H16_32'])),
                'all_three_conditional_obstructions_verified':machine>gap and reference>gap and full>D(old['time_control']['H16_32'])})
    assert [v['digits'] for v in precision]==[80,100] and all(r['conditional_obstruction_proofs'].values())
    assert not r['terms_removed_from_acceptance'] and not r['original_acceptance_recomputed_or_changed']
    assert all(r[k]==0 for k in ('new_graphs','new_actual_paths','new_independent_paths','new_reference_PDE_solves','new_density_sources','new_software_checks'))
    assert r['stage3_physical_acceptance']=='incomplete' and not r['fluid_orc_enabled'] and not r['kappa_computed']
    findings={'passed':True,'exact_rational_budget_reassembly_verified':True,'independent_second_moment_identities':identity_records,
        'precision_checks':precision,'original_decision_and_all_positive_terms_retained':True,'new_physical_data_or_solves':0,
        'new_software_checks':0,'recorded_checks_total':502,'scope':'saved-state conditioning and conditional budget lower bounds,not physical error causation or new acceptance'}
    if write:save(out/'audit.json',{'passed':True,'first_attempt_passed':True,'findings':findings,'elapsed_seconds':time.monotonic()-started,
        'peak_RSS_MiB':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,'auditor_sha256':digest(Path(__file__))})
    else:assert read(out/'audit.json')['findings']==findings
    print(json.dumps(findings,ensure_ascii=False,indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--directory',required=True);p.add_argument('--write',action='store_true');a=p.parse_args();main(a.directory,a.write)
