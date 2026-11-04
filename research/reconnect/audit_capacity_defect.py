"""Independently audit midpoint faces/capacities and the unchanged Stage3 criteria."""
from fractions import Fraction as F
from pathlib import Path
import argparse
import json
import sys
import numpy as np

from audit_route_decision import ROOT,read,save,digest,preservation
from capacity_defect_theory import SOURCE,THEORY,INPUT,ZERO_FIELDS,BLOCKED,array_records,replay


def independent_axis(values):
    q=list(map(F,values));left=[];right=[];width=[];offset=[];moment=[]
    for i,x in enumerate(q):
        a=q[i-1];b=q[(i+1)%len(q)]
        if a>=x:a-=1
        if b<=x:b+=1
        l=(a+x)/2;r=(x+b)/2
        assert l<x<r
        left.append(l);right.append(r);width.append(r-l);offset.append((l+r)/2-x)
        moment.append(((r-x)**3-(l-x)**3)/(3*(r-l)))
    return left,right,width,offset,moment


def main(directory,standalone):
    out=Path(directory).resolve();assert not (out/'sha256.json').exists();replay(out)
    parent=read(SOURCE/'sha256.json');cfg=read(out/'config.json');receipt=read(out/'definition-receipt.json')
    assert len(parent)==cfg['parent_manifest_entries_verified']==1742
    assert cfg['parent_manifest_sha256']==digest(SOURCE/'sha256.json')
    assert cfg['source_next_scope_sha256']==digest(SOURCE/'next-theory-scope.json')
    assert receipt['all_written_before_certificate_or_capacity_evaluation']
    assert (out/'theory-at-definition.md').read_bytes()==THEORY.read_bytes()
    assert digest(THEORY)==receipt['theory_sha256_before_evaluation']
    assert digest(out/'capacity-contract.json')==receipt['contract_sha256_before_evaluation']
    assert digest(out/'input-screen-spec.json')==receipt['input_screen_spec_sha256_before_evaluation']
    certs=read(out/'algebra-certificates.json');assert certs==read(standalone)
    assert certs['verified_exact_algebra_groups']==8 and len({v['name'] for v in certs['groups']})==8
    p=read(out/'input-provenance.json');packet=read(out/'saved-input.json');spec=read(out/'input-screen-spec.json')
    assert INPUT==p['source_npz']==spec['source_npz']
    assert digest(ROOT/INPUT)==p['source_npz_sha256']==spec['source_npz_sha256']==parent[INPUT]
    with np.load(ROOT/INPUT,allow_pickle=False) as z:
        arrays={'ids':z['ids'],'mass':z['mass'],'positions':z['positions'][1],'rho':z['rho'][1]}
        assert packet['time']==float(z['time'][1])==spec['source_time_display_expected']
        for k,v in arrays.items():assert v.tolist()==packet[k],k
    assert array_records(packet)==p['copied_array_records']
    exact=read(out/'input-screen-exact.json');result=read(out/'input-screen-results.json');N=128;n=N*N
    ax=independent_axis([packet['positions'][i][0] for i in range(N)])
    ay=independent_axis([packet['positions'][i*N][1] for i in range(N)])
    for a,stored in zip([ax,ay],exact['axes']):
        assert list(map(F,stored['widths']))==a[2]
        assert list(map(F,stored['centroid_offsets']))==a[3]
        assert list(map(F,stored['second_moments_about_original_points']))==a[4]
    mass=list(map(F,packet['mass']));rho=list(map(F,packet['rho']))
    auxiliary=[rho[i]*(ax[1][i%N]-ax[0][i%N])*(ay[1][i//N]-ay[0][i//N]) for i in range(n)]
    ratio=[a/m for a,m in zip(auxiliary,mass)]
    assert list(map(str,auxiliary))==exact['auxiliary_masses']
    assert list(map(str,ratio))==exact['capacity_ratios']
    lo,hi=min(ratio),max(ratio);eta=max(abs(v-1) for v in ratio);R=hi/lo
    expected={'original_total_mass':sum(mass,F(0)),'auxiliary_total_mass':sum(auxiliary,F(0)),
        'auxiliary_minus_original_total_mass':sum(auxiliary,F(0))-sum(mass,F(0)),
        'minimum_capacity_ratio':lo,'maximum_capacity_ratio':hi,'eta_maximum_local_capacity_defect':eta,
        'ratio_spread_R':R,'uniform_same_L_capacity_comparison_bound_over_initial_fluctuation':eta*(2+R)}
    offsets=[abs(ax[3][i%N])+abs(ay[3][i//N]) for i in range(n)]
    moments=[ax[4][i%N]+ay[4][i//N] for i in range(n)]
    eps_c=F(0);eps_2=F(0)
    for r in range(N):
        for c in range(N):
            i=r*N+c
            for j,d in [(r*N+(c+1)%N,(F(packet['positions'][r*N+(c+1)%N][0])-F(packet['positions'][i][0]))%1),
                (((r+1)%N)*N+c,(F(packet['positions'][((r+1)%N)*N+c][1])-F(packet['positions'][i][1]))%1)]:
                eps_c=max(eps_c,(offsets[i]+offsets[j])/d);eps_2=max(eps_2,(moments[i]+moments[j])/d)
    expected['epsilon_c_L1_upper']=eps_c;expected['epsilon_2']=eps_2
    for k,v in expected.items():
        assert F(result[k]['exact_of_saved_binary_data'])==v,k
        assert result[k]['display']==float(v),k
    assert not result['PDE_accuracy_evaluated'] and not result['new_physical_threshold_defined']
    assert result['auxiliary_mass_not_applied_to_a_solver'] and result['true_material_observation_error_not_measured']
    ledger=read(out/'acceptance-ledger.json');gate=read(out/'next-gate-spec.json')
    for rel,item in ledger['source_criteria'].items():assert digest(ROOT/rel)==item['sha256']==parent[rel]
    method=(ROOT/'docs/METHOD.md').read_text();task=(ROOT/'docs/CODEX_TASK.md').read_text()
    assert '每个空间实验至少三档 h；时间实验至少三档' in method
    assert '生产后端尚未通过 T1–T4 前' in method
    assert 'oracle、kernel、ledger 三种密度来源使用同一交换后端与物理 D' in task
    assert ledger['minimum_space_levels_from_METHOD']==ledger['minimum_time_levels_from_METHOD']==3
    assert not ledger['new_numeric_physical_threshold_defined'] and not ledger['full_stage3_pass_asserted']
    assert not ledger['reliable_calendar_finish_date_available']
    assert not gate['already_executed'] and gate['source_npz_sha256']==digest(ROOT/INPUT)
    assert gate['source_input_screen_sha256']==digest(out/'input-screen-results.json')
    assert gate['candidate_state_actions_max']==1 and gate['no_new_flow_PDE_time_grid_or_full_matrix']
    s=read(out/'stage3-status.json')
    for k in ZERO_FIELDS:assert s[k]==0,k
    assert s['new_saved_input_diagnostics']==1 and s['recorded_checks_total']==428
    assert s['new_exact_algebra_certificate_groups']==8 and not s['candidate_PDE_accuracy_evaluated']
    for obj in [s,gate]:
        assert not obj['finite_node_candidate_admitted'] and not obj['contact_transient_physically_admitted']
        assert obj['stage3_physical_acceptance']=='incomplete' and not obj['fluid_orc_enabled']
    for name in BLOCKED:assert name not in sys.modules,name
    preserved=preservation();known={v['manifest'] for v in preserved['formal_scientific_manifests_verified']}
    prior=read(SOURCE/'preservation.json')['formal_scientific_manifests_verified']+[
        {'manifest':str((SOURCE/'sha256.json').relative_to(ROOT)),'manifest_sha256':digest(SOURCE/'sha256.json'),
            'entries_verified':len(parent)}]
    for item in prior:
        if item['manifest'] in known:continue
        path=ROOT/item['manifest'];assert digest(path)==item['manifest_sha256']
        entries=read(path);assert len(entries)==item['entries_verified']
        for rel,h in entries.items():assert digest(ROOT/rel)==h,(item['manifest'],rel)
        preserved['formal_scientific_manifests_verified'].append(item);known.add(item['manifest'])
    assert len(known)==77
    save(out/'preservation.json',preserved)
    audit={'theory_and_single_input_spec_saved_before_evaluation_verified':True,
        'eight_algebra_groups_equal_standalone_output':True,'original_input_members_match_by_bytes':True,
        'independent_midplanes_integrals_capacity_ratios_and_geometric_moments_exact':True,
        'uniform_capacity_bound_not_claimed_as_full_PDE_error':True,
        'original_stage3_criteria_and_later_stage_distinction_indexed_without_new_thresholds':True,
        'one_future_fixed_state_gate_predeclared_not_executed':True,
        **{k:0 for k in ZERO_FIELDS},'new_exact_algebra_certificate_groups':8,'new_saved_input_diagnostics':1,
        'recorded_checks_total':428,'original_files_verified':782,'historical_attachments_verified':15,
        'prior_formal_manifests_verified':77,'stage3_physical_acceptance':'incomplete','fluid_orc_enabled':False}
    save(out/'capacity-audit.json',audit);print(json.dumps(audit,indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--directory',required=True);p.add_argument('--standalone',required=True);args=p.parse_args()
    main(args.directory,args.standalone)
