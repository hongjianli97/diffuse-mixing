"""Saved delta=.04 budget attribution and H conditioning; no flow/solver imports."""
from pathlib import Path
from fractions import Fraction as F
from decimal import Decimal,localcontext
import argparse,math,resource,shutil,signal,time,traceback
import numpy as np
from audit_route_decision import ROOT,read,save,digest

PARENT=ROOT/'runs/reconnect-becker-delta004'
PROTOCOL=ROOT/'research/reconnect/becker_delta004_budget_protocol.json'
KS=(8,16,32)


def arrays(path):
    with np.load(path,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}


def q(x):return F(float(x))
def record(x):return {'exact':str(x),'float':float(x)}
def variance(m,u):
    mean=sum((a*b for a,b in zip(m,u)),F(0))/sum(m,F(0))
    return sum((a*(b-mean)**2 for a,b in zip(m,u)),F(0))


def analyze(src):
    src=Path(src);r=read(src/'results.json');data=arrays(src/'n32/material.npz')
    m,c0=data['mass'],data['c0'];mf,cf=list(map(q,m)),list(map(q,c0));M=sum(mf,F(0));V0=variance(mf,cf)
    states={K:arrays(src/f'n32/k{K}/trajectory-states.npz')['states'][-1] for K in KS}
    groups={key:F(0) for key in ('machine_allowance','raw_tail','measured_sequence_difference')}
    for K in KS:
        step=read(src/f'n32/k{K}/native-results.json')['steps'][-1]
        u=states[K];v=arrays(src/f'n32/k{K}/trajectory-states.npz')['independent_states'][-1]
        difference=float(np.sqrt(m@((u-v)**2)))
        assert difference==r['numeric_by_K'][str(K)]['complete_sequence_difference']
        groups['machine_allowance']+=2*q(step['cumulative_machine_indicator_absolute'])
        groups['raw_tail']+=2*q(step['cumulative_tail_indicator_absolute'])
        groups['measured_sequence_difference']+=2*q(difference)
    ref=r['reference_indicators']['components_absolute']
    groups.update(reference_space=2*q(ref['space']),reference_time=2*q(ref['time']),
        reference_initial_rounding_geometry_bridges=2*sum((q(v) for k,v in ref.items() if k not in ('space','time')),F(0)),
        transverse_bridges=sum((q(v) for k,v in r['B_parts'].items() if 'bridge' in k),F(0)))
    B=q(r['B']);expanded=sum(groups.values(),F(0));residual=B-expanded
    assert all(v>=0 for v in groups.values()) and abs(float(residual))<16*math.ulp(r['B'])
    V=float(m@((c0-float(m@c0/m.sum()))**2));A=2*math.sqrt(float(m.sum()))*max(1.,float(max(abs(c0))))
    raw_H={k:1-variance(mf,list(map(q,states[k])))/V0 for k in KS};pairs=[]
    for a,b in ((8,16),(16,32)):
        u=list(map(q,states[a]));v=list(map(q,states[b]));w=[y-x for x,y in zip(u,v)]
        um=sum((x*y for x,y in zip(mf,u)),F(0))/M;wm=sum((x*y for x,y in zip(mf,w)),F(0))/M
        linear=[2*mass*(x-um)*(d-wm)/V0 for mass,x,d in zip(mf,u,w)]
        quadratic=sum((mass*(d-wm)**2 for mass,d in zip(mf,w)),F(0))/V0
        signed=-sum(linear,F(0))-quadratic;assert signed==raw_H[b]-raw_H[a]
        plus=sum((x for x in linear if x>0),F(0));minus=sum((x for x in linear if x<0),F(0))
        assert plus+minus==sum(linear,F(0))
        pairs.append({'from_K':a,'to_K':b,'signed_exact_H_change':record(signed),
            'linear_positive_sum_over_V0':record(plus),'linear_negative_sum_over_V0':record(minus),
            'linear_net_over_V0':record(plus+minus),'quadratic_over_V0':record(quadratic),
            'spatial_linear_cancellation_ratio':record((plus-minus)/abs(plus+minus)),
            'saved_signed_H_change':r['H_by_K'][str(b)]-r['H_by_K'][str(a)],
            'saved_minus_exact_H_change':record(q(r['H_by_K'][str(b)])-q(r['H_by_K'][str(a)])-signed)})
    gap_exact=abs(raw_H[8]-raw_H[16])-abs(raw_H[16]-raw_H[32])
    values=[]
    for digits in (80,100):
        with localcontext() as ctx:
            ctx.prec=digits;D=lambda x:Decimal.from_float(float(x));R=lambda x:Decimal(x.numerator)/Decimal(x.denominator)
            poly=lambda e:(D(A)*e+e*e)/D(V)
            full=poly(R(B));machine=poly(R(groups['machine_allowance']))
            numeric=poly(sum((R(groups[k]) for k in ('machine_allowance','raw_tail','measured_sequence_difference')),Decimal(0)))
            reference=poly(sum((R(groups[k]) for k in ('reference_space','reference_time','reference_initial_rounding_geometry_bridges')),Decimal(0)))
            gap=D(r['time_control']['H_decrease']);ceiling=D(r['time_control']['H8_16']);shifted=D(r['time_control']['H16_32'])
            # Algebraic inversion for diagnosis only. Never used as a replacement threshold.
            critical=2*D(V)*gap/(D(A)+(D(A)*D(A)+4*D(V)*gap).sqrt())
            values.append({'digits':digits,'full_H_indicator':str(full),'machine_only_lower_bound':str(machine),
                'all_numeric_only_lower_bound':str(numeric),'reference_only_lower_bound':str(reference),
                'saved_H_decrease':str(gap),'exact_state_H_decrease':str(R(gap_exact)),
                'critical_B_for_saved_gap_diagnostic_only':str(critical),
                'machine_floor_exceeds_current_gap':machine>gap,
                'machine_floor_exceeds_current_first_increment':machine>ceiling,
                'reference_floor_exceeds_current_gap':reference>gap,
                'full_floor_exceeds_shifted_first_increment':full>shifted,
                'exact_state_H_decrease_still_unresolved':R(gap_exact)<=full,
                'linear_H_budget_attribution':{k:str(D(A)*R(v)/D(V)) for k,v in groups.items()},
                'full_H_quadratic_term':str(R(B)*R(B)/D(V)),
                'accumulation_residual_linear_term':str(D(A)*R(residual)/D(V))})
    proofs={'reference_only_refinement_with_saved_paths_and_machine_allowance_cannot_resolve_current_H':all(v['machine_floor_exceeds_current_gap'] for v in values),
        'future_K16_32_64_decrease_cannot_clear_retained_full_B_even_if_new_H_increment_zero':all(v['full_floor_exceeds_shifted_first_increment'] for v in values),
        'improving_only_machine_allowance_with_saved_reference_terms_cannot_resolve_current_H':all(v['reference_floor_exceeds_current_gap'] for v in values)}
    return {'date':'2026-11-07','timezone':'Etc/UTC','decision':'saved_delta004_H_budget_floor_and_spatial_cancellation_identified',
        'parent_decision_unchanged':r['comparison_decision'],'parent_time_qualified':r['new_delta_temporal_qualified'],
        'parent_H_response_directions_consistent':r['H_response_directions_consistent'],
        'positive_B_groups':{k:record(v) for k,v in groups.items()},'saved_B':record(B),
        'saved_accumulation_residual_signed':record(residual),'budget_shares':{k:float(v/B) for k,v in groups.items()},
        'original_polynomial_coefficients':{'A':A,'V0_float':V,'V0_exact_saved_state':record(V0)},
        'saved_time_control':r['time_control'],'exact_saved_H':{str(k):record(v) for k,v in raw_H.items()},
        'H_variance_change_identity':pairs,'precision_checks':values,'conditional_obstruction_proofs':proofs,
        'old_error_changes_and_classifications_retained':{'changes':r['error_changes_new_minus_old'],'classifications':r['error_change_decisions']},
        'limits':['budget attribution is not a causal allocation of PDE error','machine allowance is not measured floating error',
            'spatial cancellation is an exact identity of saved states,not evidence that arithmetic caused the physical error',
            'no time order fit,no extrapolated state,no new tolerance','no causal separation of geometry,flux and layer resolution from two thicknesses',
            'no new-delta three-grid convergence or delta-zero limit','reference refinement indicators are not certified continuum error bounds'],
        'original_acceptance_recomputed_or_changed':False,'terms_removed_from_acceptance':False,
        'new_graphs':0,'new_actual_paths':0,'new_independent_paths':0,'new_reference_PDE_solves':0,'new_density_sources':0,
        'new_software_checks':0,'recorded_checks_total':502,'stage3_physical_acceptance':'incomplete','fluid_orc_enabled':False,'kappa_computed':False}


def preserve():
    previous=read(PARENT/'preservation.json');formal=previous['formal_manifests_verified']+[{
        'manifest':str((PARENT/'sha256.json').relative_to(ROOT)),'manifest_sha256':digest(PARENT/'sha256.json'),
        'entries_verified':len(read(PARENT/'sha256.json'))}]
    assert len(formal)==114
    for item in formal:
        assert digest(ROOT/item['manifest'])==item['manifest_sha256']
        entries=read(ROOT/item['manifest']);assert len(entries)==item['entries_verified']
        for name,h in entries.items():assert digest(ROOT/name)==h,name
    originals=read('/workspace/original-handoff-sha256.json');attachments=read('/workspace/setup-ltn/attachment-manifest.json')
    assert len(originals)==782 and len(attachments)==15
    for name,h in originals.items():assert digest(ROOT/name)==h,name
    for item in attachments:
        p=Path('/workspace')/item['path'];assert p.stat().st_size==item['bytes'] and digest(p)==item['sha256']
    for item in previous['historical_diagnostic_differences_retained']:
        p=ROOT/item['manifest'];assert digest(p)==item['manifest_sha256']
        entries=read(p);assert len(entries)==item['entries']
        diffs=[{'path':k,'saved':v,'current':digest(ROOT/k)} for k,v in entries.items() if digest(ROOT/k)!=v]
        assert sorted(diffs,key=lambda x:x['path'])==sorted(item['historical_paths_differing_from_later_formal_versions'],key=lambda x:x['path'])
    return {'formal_manifests_verified':formal,'originals_verified':782,'attachments_verified':15,
        'historical_diagnostic_differences_retained':previous['historical_diagnostic_differences_retained'],'all_prior_scientific_files_unchanged':True}


def main(output):
    started=time.monotonic();out=Path(output);out.mkdir(parents=True,exist_ok=False)
    def alarm(*_):raise TimeoutError('registered120s read-only review cap')
    signal.signal(signal.SIGALRM,alarm);signal.alarm(120)
    try:
        p=read(PROTOCOL);gate=read(PARENT/'next-gate-spec.json')
        assert gate['type']=='read_only_delta004_error_and_time_budget_review' and not gate['already_executed']
        assert digest(PARENT/'next-gate-spec.json')==p['source_gate_sha256']
        for name,h in p['sources'].items():assert digest(ROOT/name)==h,name
        shutil.copyfile(PROTOCOL,out/'protocol.json');shutil.copyfile(PARENT/'next-gate-spec.json',out/'source-gate-spec.json')
        save(out/'implementation-lock.json',{'files':{name:digest(ROOT/name) for name in p['implementation_files']},'no_new_flow_geometry_states_or_PDE':True})
        copies=[]
        for name in p['saved_data_files']:
            target=out/'source'/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(PARENT/name,target)
            copies.append({'source':str((PARENT/name).relative_to(ROOT)),'copy':str(target.relative_to(out)),'sha256':digest(target)})
        save(out/'source-copies.json',copies);r=analyze(out/'source');save(out/'results.json',r);save(out/'preservation.json',preserve())
        elapsed=time.monotonic()-started;peak=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024
        assert elapsed<120 and peak<1024
        save(out/'execution.json',{'completed':True,'elapsed_seconds':elapsed,'peak_RSS_MiB':peak,
            'new_graphs_or_states_or_PDE_solves':0,'published_or_pushed':False})
        print(r['decision'],flush=True)
    except BaseException as exc:
        save(out/'failed-execution.json',{'type':type(exc).__name__,'message':str(exc),'traceback':traceback.format_exc()});raise
    finally:signal.alarm(0)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);main(p.parse_args().output)
