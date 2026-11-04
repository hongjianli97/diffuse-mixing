"""Record a theory-only outcome and its next scope; preserve old reports."""
from pathlib import Path
import argparse,json


def main(output):
    root=Path(__file__).resolve().parents[2];out=Path(output).resolve()
    assert not (out/'sha256.json').exists(),'never rewrite sealed evidence'
    r=json.loads((out/'proof-certificates.json').read_text());a=json.loads((out/'proof-audit.json').read_text())
    assert a['seven_exact_rational_certificate_groups_recomputed'] and a['standalone_standard_library_result_identical']
    next_plan={'scope':'theory-only weak bilinear consistency review of the same oracle path-resistance radial kernel',
        'source_operator':'research/reconnect/oracle_path_reference.py',
        'contact':'same flat two-phase positive bounded density contact; full analytic density remains extra oracle information',
        'fields':'interface-continuous, phasewise C3 with bounded derivatives on the compact observation support; test function globally smooth and compactly supported',
        'PDE_transmission':'state concentration and normal Gamma*gradient continuity requirements explicitly',
        'tasks':['derive the exact symmetric nonlocal bilinear form including factors, sign and mass measure',
            'bound separated pure-phase Taylor remainder and the O(h)-width contact contribution under explicit regularity assumptions',
            'distinguish local adjacent-cell bias, frozen weak consistency, global norm and transient convergence; do not infer the last from the first three'],
        'not_yet_proved':'general piecewise-smooth weak error rate; present report only bounds local quadratic model',
        'same_original_kernel_and_path_coefficient':True,'no_new_density_closure_or_candidate':True,
        'ordinary_central_resistance_FV_baseline_retained':True,'new_candidate_graphs':0,'new_candidate_transient_runs':0,
        'new_reference_transient_runs':0,'parameter_or_mesh_matrix_scan':False,'fluid_orc_enabled':False}
    groups=r['certificate_groups']
    status={'updated':'2026-11-01','timezone':'Etc/UTC','stage3_physical_acceptance':'incomplete','fluid_orc_enabled':False,
        'scope':'completed predeclared second-order interface consistency theory review',
        'reference_only':True,'uses_additional_oracle_density_information':True,'candidate_and_saved_total_and_mass_states_unchanged':True,
        'unit_common_curvature_trace_ratios':{'low_density':'5/2','high_density':'5/8'},
        'pure_transverse_adjacent_mass_average':'low-density cell too large and high-density cell too small; finite Delta/h local bias persists',
        'original_Delta_over_h':'25/128',
        'exact_adjacent_cell_separation_bounds':groups['adjacent_mass_average_strict_defects_at_original_ratio'],
        'fixed_cell_and_weak_quadratic_limits':'fixed Delta mean O(h/Delta), local fixed-window weighted L1 O(h) and L2 O(sqrt(h)); do not infer general convergence failure',
        'new_numerical_or_flow_tests':0,'exact_algebra_certificate_groups':7,'recorded_checks_total':352,
        'old_v080_or_246_tests_recovered':False,'new_candidate_graphs':0,'new_candidate_transient_runs':0,'new_reference_transient_runs':0,
        'new_operator_quadrature_evaluations':0,'finite_node_candidate_admitted':False,'contact_transient_admitted':False,
        'current_cosine_directional_errors_reinterpreted_as_quadratic_predictions':False,
        'next_weak_interface_review_predeclared':next_plan,'next_task':'same-kernel general weak bilinear consistency theory review',
        'remaining':['finite-node density closure not identified','general weak/transient convergence not proved','geometry, shocks, walls, ledger, nonplanar/DNS physical acceptance'],
        'report':'research/reconnect/ORACLE_INTERFACE_SECOND_ORDER.md','pdf_generated':False,'software_version_created':False,'published':False}
    (out/'stage3-status.json').write_text(json.dumps(status,indent=2,ensure_ascii=False)+'\n')
    p=root/'docs/RESEARCH_STATUS.md';text=p.read_text().replace(
        '46. [同一oracle核的法向优先方向残差预算](../research/reconnect/ORACLE_DIRECTIONAL_RESIDUAL_BUDGET.md)（本轮）。',
        '46. [同一oracle核的法向优先方向残差预算](../research/reconnect/ORACLE_DIRECTIONAL_RESIDUAL_BUDGET.md)。\n47. [同一oracle核的二阶界面一致性理论审查](../research/reconnect/ORACLE_INTERFACE_SECOND_ORDER.md)（本轮）。')
    text=text[:text.index('本轮关键进展：')]+'''本轮关键进展：完成预定二阶界面一致性理论审查。同一oracle沿段阻力核保持恒法向通量，但共同法向/横向单位曲率的一侧变化率/PDE为5/2、5/8。纯横向二次场直接给出相邻物质质量平均偏差：低密度侧偏大、高密度侧偏小；固定有限Delta/h时局部平均偏差持续，原比值25/128有精确正分离界。

该局部结论不等于全域不收敛：对单接触二次模型，固定Delta平均误差O(h/Delta)，固定观察窗质量加权L1为O(h)、L2为O(sqrt(h))。点态迹、局部缩小单元和弱/全域极限分开，不能把2.5/0.625当成原余弦态5.42%/3.99%的定量归因。原候选、总作用、物质身份/质量/c0与PDE结果保持。

本轮只做解析推导及7组标准库精确代数证书，无新增数值或流动测试；重建检查累计仍352项，旧v0.8.0/246项未恢复。782原文件、15历史附件和46份先前正式科学清单保持；4份历史诊断清单已有路径差异另核。

下一步只审查同一核的一般弱双线性形式，明确分相光滑性/传输条件，分开纯相Taylor余项和O(h)界面层，区分冻结弱一致性与瞬态收敛。完整rho仍为额外oracle输入，不补密度闭合，不改系数、建图、传播或扫描；普通中心阻力FV基线保留。

数据入口：runs/reconnect-oracle-interface/stage3-status.json。Stage 3 incomplete，fluid ORC关闭；无PDF/新版本、提交/推送、环境配置修改或发布。
'''
    p.write_text(text)
    print(json.dumps({'trace_ratios':status['unit_common_curvature_trace_ratios'],'exact_algebra_groups':7,
        'recorded_checks_total':352,'next_task':status['next_task']},indent=2,ensure_ascii=False))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args();main(a.output)
