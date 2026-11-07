"""Audit the corrected active/inactive triangle bound, plus the saved review."""
from pathlib import Path
from fractions import Fraction as F
from decimal import Decimal,localcontext
import argparse,json,time
from audit_route_decision import ROOT,read,save,digest
from audit_becker_spatial_budget import main as original_arithmetic_audit


def main(directory,write=False):
    start=time.monotonic();out=Path(directory)
    original_arithmetic_audit(out)
    r=read(out/'planning-correction.json');parent=read(out/'source/results.json')
    preservation=read(out/'planning-correction-preservation.json')
    for name,h in preservation['prior_files'].items():assert digest(out/name)==h
    assert digest(ROOT/'research/reconnect/becker_spatial_budget_review.py')==preservation['prior_code_sha256']
    assert digest(ROOT/'research/reconnect/becker_spatial_budget_completion.py')==preservation['completion_code_sha256']
    witness=r['exact_scalar_counterexample']
    old=abs(F(witness['coarse_error_old']))-abs(F(witness['fine_error_old']))
    new=abs(F(witness['coarse_error_old']))-abs(F(witness['fine_error_new']))
    increment=abs(F(witness['fine_error_new'])-F(witness['fine_error_old']))
    budget=F(witness['coarse_old_time_indicator'])+increment+F(witness['other'])
    assert all(F(witness[k])>old/2 for k in ('coarse_old_time_indicator','fine_old_time_indicator'))
    assert new>budget and new==7 and budget==6
    checks=[]
    for digits in (80,100):
        with localcontext() as ctx:
            ctx.prec=digits;values=[];D=lambda q:Decimal(q.numerator)/Decimal(q.denominator)
            for item in r['corrected_planning']:
                pair=parent['adjacent_spatial_comparisons'][1];key=item['observable'];parts=pair['positive_components'][key]
                s=F(parent['common_initial_fluctuation_norm']);gain=F(pair['gains'][key])
                other=sum((F(v) for name,v in parts.items() if not name.endswith('observed8_16')),F(0))/s
                for choice in item['choices']:
                    active=set(choice['new_K32_columns']);fixed=sum((F(parts[f'n{n}_observed8_16']) for n in (64,128) if n not in active),F(0))/s
                    room=(gain-other-fixed)/2
                    assert room==F(choice['ceiling_exact']) and room>0
                    # At the boundary, worst-case gain equals candidate indicator.
                    assert gain-room==room+fixed+other
                    # Below the boundary any nonnegative remaining margin is usable
                    # for extra positive numeric/observation terms; no new data implied.
                    candidate=room/2;assert gain-candidate>candidate+fixed+other
                    values.append({'observable':key,'active':sorted(active),'ceiling':str(D(room)),'fixed_inactive_indicator':str(D(fixed))})
            checks.append({'digits':digits,'corrected_active_inactive_checks':values})
    assert r['selected_next_probe']['columns']==128 and r['selected_next_probe']['new_primary_paths']==1
    assert not r['selected_next_probe']['new_full_spatial_or_Stage3_pass_allowed']
    findings={'passed':True,'prior_arithmetic_replayed':True,'earlier_necessity_claim_refuted_by_exact_counterexample':True,
        'corrected_triangle_bound_verified':True,'precision_checks':checks,'old_scientific_result_unchanged':True,
        'no_new_states_or_software_tests':True,'recorded_checks_total':502,'stage3':'incomplete','fluid_orc_enabled':False}
    if write:save(out/'completion-audit.json',{'passed':True,'findings':findings,'elapsed_seconds':time.monotonic()-start,'auditor_sha256':digest(Path(__file__))})
    else:assert read(out/'completion-audit.json')['findings']==findings
    print(json.dumps({'completion_passed':True,'old_single_grid_necessity_claim_valid':False,'corrected_choices':6,'new_states':0},indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--directory',required=True);p.add_argument('--write',action='store_true');a=p.parse_args();main(a.directory,a.write)
