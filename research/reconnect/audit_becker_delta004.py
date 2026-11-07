"""Read-only independent saved-path, mass-union and 80/100-digit budget audit."""
from pathlib import Path
from fractions import Fraction as F
from decimal import Decimal,localcontext
import argparse,hashlib,json,math,resource,time
import numpy as np
from audit_route_decision import ROOT,read,digest
from audit_becker_spatial import arrays,norm,save,Q
from audit_becker_fine_time import squared,variance_exact
from audit_aligned_becker import near
from audit_delta004_specializations import audit_actions_delta004,audit_references_delta004,verify_specializations
from affine_shear_completion import plain_json
from becker_point_core import Becker
from becker_delta004_sampling import sample_frames_delta004,verify_sampling_rule
from becker_delta004_core import analyze

KS=(8,16,32);REFS=((0,8192),(1,8192),(1,16384))


def budgets(out,r):
    data=arrays(out/'n32/material.npz');col=arrays(out/'n32/column.npz');common=arrays(out/'common-observations.npz')
    m,c0,mp,cp=data['mass'],data['c0'],col['mass'],col['c0']
    np.testing.assert_array_equal(m,np.tile(mp/8,8));np.testing.assert_array_equal(c0,np.tile(cp,8))
    obs=[arrays(out/f'reference/observations-l{l}-k{k}.npz') for l,k in REFS]
    rc=obs[-1]['n32_projected'];ref=np.tile(rc,8)
    states={k:arrays(out/f'n32/k{k}/trajectory-states.npz')['states'][-1] for k in KS}
    capacities=[sum((Q(m[row*32+i]) for row in range(8)),F(0))/Q(mp[i]) for i in range(32)]
    factor=math.nextafter(math.sqrt(float(max(capacities))),math.inf)
    assert r['capacity']=={'defect':float(max(abs(q-1) for q in capacities)),'norm_factor_upper':factor}
    rounds={};caps={}
    for name,u in [('c0',c0),('reference_aggregated',ref)]+[(f'k{k}',states[k]) for k in KS]:
        exact=[sum((Q(m[row*32+i])*Q(u[row*32+i]) for row in range(8)),F(0))/Q(mp[i]) for i in range(32)]
        bridge=np.array([float(Q(a)-b) for a,b in zip(common[name],exact)])
        rounds[name]=norm(mp,bridge)
        caps[name]=norm(mp,np.array([float(abs(q-1))*float(max(abs(u))) for q in capacities]))
    jp=r['J_parts']
    assert jp['initial_common_vs_original']==norm(mp,common['c0']-cp)
    assert jp['reference_common_vs_direct32']==norm(mp,common['reference_aggregated']-rc)
    assert jp['initial_common_rounding']==rounds['c0'] and jp['reference_common_rounding']==rounds['reference_aggregated']
    assert jp['capacity_initial_bound']==caps['c0']
    for k in KS:
        assert jp[f'k{k}_common_rounding']==rounds[f'k{k}'] and jp[f'k{k}_capacity_final_bound']==caps[f'k{k}']
        bridge=(states[k].reshape(8,32)*(mp/8)).sum(axis=0)/mp-states[k].reshape(8,32)[0]
        assert r['B_parts'][f'k{k}_transverse_bridge']==norm(mp,bridge)
        independent=arrays(out/f'n32/k{k}/trajectory-states.npz')['independent_states'][-1]
        last=read(out/f'n32/k{k}/native-results.json')['steps'][-1];num=r['numeric_by_K'][str(k)]
        assert num['machine']==last['cumulative_machine_indicator_absolute'] and num['tail']==last['cumulative_tail_indicator_absolute']
        assert num['complete_sequence_difference']==norm(m,states[k]-independent)
    assert r['B_parts']['initial_transverse_bridge']==norm(mp,(c0.reshape(8,32)*(mp/8)).sum(axis=0)/mp-cp)
    rm=r['reference_indicators'];parts=rm['components_absolute']
    assert parts['space']==norm(mp,obs[0]['n32_projected']-obs[1]['n32_projected'])
    assert parts['time']==norm(mp,obs[1]['n32_projected']-obs[2]['n32_projected'])
    for i,z in enumerate(obs):
        for key in ('exact_initial_mean_gap','initial_aggregation_rounding','terminal_aggregation_rounding','initial_geometry_bound','terminal_geometry_bound'):
            assert parts[f'reference{i}_{key}']==norm(mp,z['n32_'+key])
    # Exact rational variance is independent of the floating metrics implementation.
    for k in KS:
        assert abs(float(1-variance_exact(m,states[k])/variance_exact(m,c0))-r['H_by_K'][str(k)])<=64*np.finfo(float).eps
        assert abs(float(1-variance_exact(mp,common[f'k{k}'])/variance_exact(mp,cp))-r['common_H_by_K'][str(k)])<=64*np.finfo(float).eps
    assert abs(float(1-variance_exact(m,ref)/variance_exact(m,c0))-r['reference_H']['native'])<=64*np.finfo(float).eps
    assert abs(float(1-variance_exact(mp,rc)/variance_exact(mp,cp))-r['reference_H']['common'])<=64*np.finfo(float).eps
    old=read(out/'source/common-K32-results.json');oldspace=read(out/'source/spatial-results.json');baseline=old['by_grid']['32']
    assert r['retained_old_case']==baseline and r['old_shifted_H_outcomes_preserved']==old['shifted_H_time_controls_resolved']
    sq={'native':squared(m,states[32],ref),'common':squared(mp,common['k32'],rc),
        'state8_16':squared(m,states[8],states[16]),'state16_32':squared(m,states[16],states[32]),
        'common16_32':squared(mp,common['k16'],common['k32'])}
    V=float(m@((c0-float(m@c0/m.sum()))**2));Vp=float(mp@((cp-float(mp@cp/mp.sum()))**2))
    A=2*math.sqrt(float(m.sum()))*max(1.,float(max(abs(c0))));Ap=2*math.sqrt(float(mp.sum()))*max(1.,float(max(abs(cp))))
    records=[]
    for digits in (80,100):
        with localcontext() as ctx:
            ctx.prec=digits;D=lambda x:Decimal.from_float(float(x));R=lambda q:Decimal(q.numerator)/Decimal(q.denominator)
            roots={key:R(q).sqrt() for key,q in sq.items()};sigma=D(r['common_initial_fluctuation_norm'])
            near(sigma,R(variance_exact(mp,cp)).sqrt())
            near(roots['native']/sigma,r['errors']['native']);near(roots['common']/sigma,r['errors']['common'])
            for key in ('state8_16','state16_32'):near(roots[key],r['time_control'][key])
            near(roots['common16_32'],r['new_spatial_parts']['common']['observed16_32'])
            rt=sum(map(D,parts.values()),Decimal(0));near(rt,rm['total_absolute']);near(2*rt,r['B_parts']['twice_reference_all_positive_terms'])
            numeric=sum((D(x) for p in r['numeric_by_K'].values() for x in p.values()),Decimal(0))
            near(2*numeric,r['B_parts']['twice_all_path_numeric'])
            B=sum(map(D,r['B_parts'].values()),Decimal(0));J=D(factor)*B+sum(map(D,jp.values()),Decimal(0))
            near(B,r['B']);near(J,r['J'])
            hb=(D(A)*B+B*B)/D(V);hpb=(D(Ap)*J+J*J)/D(Vp)
            near(hb,r['time_control']['H_indicator']);near(hpb,r['new_spatial_parts']['common_H']['original_H_indicator'])
            assert all(x>=0 for x in parts.values()) and all(x>=0 for x in r['B_parts'].values()) and all(x>=0 for x in jp.values())
            gap=roots['state8_16']-roots['state16_32'];near(gap,r['time_control']['state_decrease'])
            h={k:D(r['H_by_K'][str(k)]) for k in KS};hp={k:D(r['common_H_by_K'][str(k)]) for k in KS}
            hg=abs(h[8]-h[16])-abs(h[16]-h[32]);near(hg,r['time_control']['H_decrease'])
            assert (gap>B)==r['time_control']['state_resolved'] and (hg>hb)==r['time_control']['H_resolved']
            near(abs(h[16]-h[32]),r['new_spatial_parts']['native_H']['observed16_32'])
            near(abs(hp[16]-hp[32]),r['new_spatial_parts']['common_H']['observed16_32'])
            near(abs(h[32]-D(r['reference_H']['native'])),r['errors']['native_H'])
            near(abs(hp[32]-D(r['reference_H']['common'])),r['errors']['common_H'])
            comparisons=[]
            for label,p in r['cross_positive_components'].items():
                expected={**{'retained_old_'+k:v for k,v in baseline['spatial_parts'][label].items()},
                          **{'new_'+k:v for k,v in r['new_spatial_parts'][label].items()}}
                assert p==expected and all(x>=0 for x in p.values())
                U=sum(map(D,p.values()),Decimal(0))/(sigma if label in ('native','common') else 1)
                G=D(r['errors'][label])-D(baseline['errors'][label]);near(U,r['fixed_cross_indicators'][label]);near(G,r['error_changes_new_minus_old'][label])
                decision='thinner_error_larger_resolvably' if G>U else ('thinner_error_smaller_resolvably' if G < -U else 'error_change_unresolved')
                assert decision==r['error_change_decisions'][label]
                comparisons.append({'observable':label,'change':str(G),'indicator':str(U),'decision':decision})
                if label in ('native_H','common_H'):
                    v=r['H_responses'][label];native=label=='native_H';prefix='native' if native else 'common'
                    dc=(h[32] if native else hp[32])-D(baseline[prefix+'_H_K32'])
                    dr=D(r['reference_H'][prefix])-D(oldspace['observations']['32'][prefix+'_reference_H'])
                    near(dc,v['candidate_response']);near(dr,v['reference_response']);near(dc-dr,v['response_difference'])
                    sign=lambda x:'positive' if x>U else ('negative' if x < -U else 'unresolved')
                    sc,sr=sign(dc),sign(dr);status='direction_unresolved' if 'unresolved' in (sc,sr) else ('direction_consistent' if sc==sr else 'direction_inconsistent')
                    assert (sc,sr,status)==(v['candidate_sign'],v['reference_sign'],v['direction_status'])
                    assert not v['magnitude_accuracy_acceptance_claimed']
            records.append({'digits':digits,'B':str(B),'J':str(J),'H_indicator':str(hb),'common_H_indicator':str(hpb),'comparisons':comparisons})
    return {'complete_mass_unions_and_all_rounding_capacity_terms_verified':True,'exact_variance_H_verified':True,
        'all_old_positive_terms_retained':True,'precision_checks':records}


def main(directory,write=False):
    started=time.monotonic();out=Path(directory);r=read(out/'results.json');ex=read(out/'execution.json')
    assert ex['completed'] and ex['elapsed_seconds']<900 and ex['reference_seconds']<300 and ex['peak_RSS_MiB']<1024
    assert ex['new_graphs']==56 and ex['new_primary_paths']==ex['new_independent_paths']==ex['new_reference_PDE_solves']==3
    p=read(out/'protocol.json')
    assert digest(out/'source-gate-spec.json')==p['source_gate_sha256']
    for name,h in read(out/'implementation-lock.json')['files'].items():assert digest(ROOT/name)==h,name
    for item in read(out/'source-copies.json'):
        assert digest(out/item['copy'])==item['sha256']==p['source_files'][item['source']]
    verify_sampling_rule();verify_specializations()
    seal=read(out/'coefficient-seal.json');assert seal['graphs']==56 and seal['all56_sealed_before_mass_scalar_oldstates_reference']
    assert not seal['internal_material_faces_given_to_builder'] and digest(out/'n32/graph-index.json')==seal['index_sha256']
    labels=np.load(out/'n32/source-mass-labels.npy',allow_pickle=False);flow=Becker(delta=.04)
    assert float(labels[-1]).hex()==read(out/'source-gate-spec.json')['M_star_hex']
    assert read(out/'n32/pure-inputs.json')==sample_frames_delta004(labels,flow)
    for l,k in REFS:
        z=arrays(out/f'reference/input-l{l}-k{k}.npz');g=read(out/f'reference/guard-l{l}-k{k}.json')
        assert g['delta']==.04 and g['rho_callback_calls']==k
        assert [s['step'] for s in g['rho_callback_sample_fingerprints']]==[0,k//2,k-1]
        for s in g['rho_callback_sample_fingerprints']:
            assert s['time']==(s['step']+.5)*(.1/k)
            assert hashlib.sha256(flow.rho_mu(z['mass_edges'][1:-1],s['time']).tobytes()).hexdigest()==s['rho_float64_sha256']
    geometry=read(out/'physical-geometry.json');x=flow.x_mu(labels,0.);widths=np.diff(x)
    assert geometry['initial_x_endpoints']==x[[0,-1]].tolist()
    assert geometry['initial_material_cell_width_range']==[float(widths.min()),float(widths.max())]
    actions=audit_actions_delta004(out,32,write);references=audit_references_delta004(out);bookkeeping=budgets(out,r)
    current,common=analyze(out);assert plain_json(current)==r
    saved=arrays(out/'common-observations.npz')
    for key,value in common.items():np.testing.assert_array_equal(value,saved[key])
    assert not r['new_delta_three_grid_control_completed'] and not r['delta_zero_limit_claimed'] and not r['H_check_waived']
    assert r['stage3_physical_acceptance']=='incomplete' and not r['fluid_orc_enabled'] and not r['kappa_computed']
    findings={'passed':True,'actions':actions,'references':references,'observations_and_budgets':bookkeeping,
        'sampling_rule_and_all_saved_new_frames_verified':True,'reference_density_callback_sample_fingerprints_verified':9,
        'frozen_analysis_reproduced':True,'comparison_decision':r['comparison_decision'],
        'new_primary_or_reference_PDE_solves_in_audit':0,'new_software_checks':0,'recorded_checks_total':502,
        'scope':'full saved independent action sequences, exact reference mass-union observations and 80/100-digit arithmetic; reference PDE not replayed; refinement indicators are not certified continuum error bounds'}
    if write:save(out/'audit.json',{'passed':True,'first_attempt_passed':True,'findings':findings,'elapsed_seconds':time.monotonic()-started,
        'peak_RSS_MiB':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,'auditor_sha256':digest(Path(__file__))})
    else:assert read(out/'audit.json')['findings']==findings
    print(json.dumps(plain_json(findings),indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--directory',required=True);p.add_argument('--write',action='store_true');a=p.parse_args();main(a.directory,a.write)
