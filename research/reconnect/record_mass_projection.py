"""Record the mass-average projection theory and one predeclared row control."""
from pathlib import Path
import argparse,json


def main(output):
    root=Path(__file__).resolve().parents[2];out=Path(output).resolve()
    assert not (out/'sha256.json').exists(),'never rewrite sealed evidence'
    r=json.loads((out/'proof-certificates.json').read_text());audit=json.loads((out/'proof-audit.json').read_text())
    assert audit['nine_exact_rational_certificate_groups_recomputed'] and audit['standalone_standard_library_result_identical']
    next_plan={'scope':'one original-ratio ideal aggregate quadratic row-response integral; not a full graph',
        'source_report':'research/reconnect/MASS_AVERAGE_PROJECTION_WEAK_BUDGET.md',
        'N_per_direction':128,'h_m':.04,'D_m2_per_s':.02,'cell_side_m':'1/128','lambda':'25/128',
        'geometry':'same original regular squares in an interior pure phase; no contact inside local support',
        'analytic_test':'c=a^2/2, exact material mean a_center^2/2+ell^2/24; does not replace original c0',
        'quantity':'eta(lambda)=lambda^2*integral beta(q)*frac(q_a/lambda)*(1-frac(q_a/lambda)) d^2q; aggregate response D*(1+eta)',
        'fixed_quadrature_levels':[{'normal_order':32,'transverse_order':32},{'normal_order':64,'transverse_order':64}],
        'normal_panels':'split at original support endpoints, zero and all s=k*ell inside support; no fractional-part smoothing',
        'best_raw_rule_preselected':'fine 64/64; no extrapolation or choosing by agreement with PDE',
        'controls':'retain eta and aggregate rate from both rules and their differences; compare strict rational bounds, not an acceptance threshold',
        'quadrature_difference_is_rigorous_bound':False,
        'original_mass_ids_c0_and_saved_candidate_and_PDE_unchanged':True,
        'no_new_edge_weights_or_full_graph':True,'no_division_by_one_plus_eta_or_moment_renormalization':True,
        'do_not_attribute_result_to_saved_cosine_errors_or_promote_backend':True,
        'ordinary_central_resistance_FV_baseline_retained':True,'new_candidate_graphs':0,'new_candidate_transient_runs':0,
        'new_reference_transient_runs':0,'parameter_or_mesh_matrix_scan':False,'fluid_orc_enabled':False}
    status={'updated':'2026-11-01','timezone':'Etc/UTC','stage3_physical_acceptance':'incomplete','fluid_orc_enabled':False,
        'scope':'completed predeclared mass-average projection weak budget theory review',
        'reference_only_extra_material_geometry_and_full_density':True,'candidate_and_saved_total_and_mass_states_unchanged':True,
        'ideal_closure_identity':'Q_star P_rho = P_rho L_h P_rho; not generally P_rho L_h',
        'projection_weak_bound':'Gamma_max*V*C1*P1*((32/3)*(Delta/h)+18*(Delta/h)^2)',
        'sufficient_joint_limit':'h -> 0 and maximum cell diameter Delta/h -> 0',
        'sufficient_condition_claimed_necessary_for_all_methods':False,
        'fixed_width_ratio_ideal_closure_counterexample':'pure-phase uniform square cells, quadratic mass averages: rate D*(1+eta(ell/h)), eta>0 independent of h',
        'original_width_ratio':'25/128','original_maximum_diameter_ratio':'25*sqrt(2)/128',
        'eta_at_original_ratio_numerically_evaluated':False,
        'strict_eta_bounds':r['certificate_groups']['fixed_original_ratio_strict_excess_bounds'],
        'saved_point_candidate_given_new_numerical_verdict':False,
        'new_numerical_or_flow_tests':0,'exact_algebra_certificate_groups':9,'recorded_checks_total':352,
        'old_v080_or_246_tests_recovered':False,'new_candidate_graphs':0,'new_candidate_transient_runs':0,
        'new_reference_transient_runs':0,'new_operator_quadrature_evaluations':0,'finite_node_candidate_admitted':False,'contact_transient_admitted':False,
        'next_single_row_response_predeclared':next_plan,'next_task':'one original-ratio ideal aggregate quadratic row-response integral with two fixed rules',
        'remaining':['quantify original-ratio ideal mean closure excess','finite-node pair quadrature and subcell density identifiability',
            'transient convergence and general geometry, shocks, walls, ledger, nonplanar/DNS acceptance'],
        'report':'research/reconnect/MASS_AVERAGE_PROJECTION_WEAK_BUDGET.md','pdf_generated':False,'software_version_created':False,'published':False}
    (out/'stage3-status.json').write_text(json.dumps(status,indent=2,ensure_ascii=False)+'\n')
    p=root/'docs/RESEARCH_STATUS.md';text=p.read_text().replace(
        '48. [同一oracle核的弱界面一致性与显式一阶项](../research/reconnect/ORACLE_WEAK_INTERFACE_CONSISTENCY.md)（本轮）。',
        '48. [同一oracle核的弱界面一致性与显式一阶项](../research/reconnect/ORACLE_WEAK_INTERFACE_CONSISTENCY.md)。\n49. [质量平均投影弱预算与固定比例闭合反例](../research/reconnect/MASS_AVERAGE_PROJECTION_WEAK_BUDGET.md)（本轮）。')
    text=text[:text.index('本轮关键进展：')]+'''本轮关键进展：完成预定质量平均投影弱预算。理想单元对共享图精确对应P_rho L_h P_rho，不是连续场真实平均作用P_rho L_h。投影弱误差显式界为Gamma_max*V*C1*P1*((32/3)*Delta/h+18*(Delta/h)^2)；结合连续核定理，h趋零且单元最大直径Delta/h趋零是充分条件。

另有严格固定比例反例：均匀纯相规则方格的二次质量平均，理想精确积分图响应为D*(1+eta(ell/h))，eta严格为正且固定ell/h时不随h变化；可在固定光滑测试下保留偏差。它针对这一理想分片常数闭合，不是对已保存点候选的新数值判决。原ell/h=25/128，而最大直径比为25*sqrt(2)/128，不混用。eta只有解析严格界，尚未数值求积。

本构、点对求积、质量平均闭合、连续核到PDE四项以带符号恒等式区分，不按范数当可加份额。真实物质几何和完整rho仅为额外理论oracle，候选接口、原m/c0/身份、总作用和PDE保持。新增9组标准库精确代数证书独立复算一致，无新增数值/流动测试，重建检查累计仍352项；旧v0.8.0/246项未恢复。

782原文件、15历史附件和48份先前正式科学清单保持，4份历史诊断清单既有路径差异另核。下一步只做原N128/h0.04/D0.02的一行理想二次响应积分，预定法向/横向32/32与64/64两规则，整数单元宽度处拆分；不构全图、不传播、不扫描、不用eta修正权重。普通中心阻力FV基线保持。

数据入口：runs/reconnect-mass-projection/stage3-status.json。Stage 3 incomplete，fluid ORC关闭；无PDF/新版本、提交/推送、环境配置修改或发布。
'''
    p.write_text(text)
    print(json.dumps({'projection_weak_bound':status['projection_weak_bound'],'ideal_closure_fixed_ratio_excess':'strictly positive',
        'exact_algebra_groups':9,'recorded_checks_total':352,'next_task':status['next_task']},indent=2,ensure_ascii=False))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args();main(a.output)
