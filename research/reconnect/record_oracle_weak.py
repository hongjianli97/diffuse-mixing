"""Record the scoped weak-form theorem; do not promote a candidate."""
from pathlib import Path
import argparse,json


def main(output):
    root=Path(__file__).resolve().parents[2];out=Path(output).resolve()
    assert not (out/'sha256.json').exists(),'never rewrite sealed evidence'
    r=json.loads((out/'proof-certificates.json').read_text());a=json.loads((out/'proof-audit.json').read_text())
    assert a['eight_exact_rational_certificate_groups_recomputed'] and a['standalone_standard_library_result_identical']
    next_plan={'scope':'theory-only mass-average projection weak budget for the same continuous oracle kernel',
        'source_report':'research/reconnect/ORACLE_WEAK_INTERFACE_CONSISTENCY.md',
        'state':'fixed original material masses and rho-weighted cell averages; never trajectory point values',
        'ideal_reference':'analytic cell-pair integral of the unchanged K_path*B_h, with true material partition explicitly extra geometry oracle',
        'tasks':['derive the ideal shared-exchange and mass-inner-product bilinear identities without computing any new edge weights',
            'bound the rho-weighted cell-average projection difference in the nonlocal weak form; state cell-size Delta and h assumptions',
            'keep projection, finite-node pair quadrature and unavailable subcell-density closure distinct; do not apply fixed-C3 bounds to shrinking discontinuous projections'],
        'no_new_candidate_geometry_inputs_or_backend':True,'no_new_kernel_or_density_reconstruction':True,
        'ordinary_central_resistance_FV_baseline_retained':True,'new_candidate_graphs':0,'new_candidate_transient_runs':0,
        'new_reference_transient_runs':0,'new_operator_quadrature_evaluations':0,
        'parameter_or_mesh_matrix_scan':False,'fluid_orc_enabled':False}
    status={'updated':'2026-11-01','timezone':'Etc/UTC','stage3_physical_acceptance':'incomplete','fluid_orc_enabled':False,
        'scope':'completed predeclared same-oracle weak interface consistency review',
        'reference_only':True,'uses_additional_oracle_density_information':True,'candidate_and_saved_total_and_mass_states_unchanged':True,
        'weak_consistency_proved_within_scope':True,
        'assumptions':['flat phase-constant positive density','c continuous and phasewise C3 with fixed derivative bounds',
            'normal Gamma*gradient continuous for the local PDE without an interface Dirac','fixed globally C3 compact or periodic test'],
        'error_bound':'12*Gamma_max*C1*P1*interface_length*h + pure_phase_B(h), B(h)=O(h^2)',
        'sharp_expansion':'W_h-W = h*kappa_tau*integral_interface(c_tau*phi_tau) + R_h',
        'remainder_bound':'abs(R_h)<=B(h)+48*Gamma_max*interface_length*(C2*P1+C1*P2)*h^2',
        'kappa_tau_original_density_ratio':'(128/4125*log(4)-16/275)/pi < 0',
        'generic_Oh_weak_order_is_sharp':True,
        'shrinking_cell_mean_or_transient_convergence_proved':False,'finite_node_density_closure_proved':False,
        'old_cosine_errors_recomputed_or_refitted':False,
        'new_numerical_or_flow_tests':0,'exact_algebra_certificate_groups':8,'recorded_checks_total':352,
        'old_v080_or_246_tests_recovered':False,'new_candidate_graphs':0,'new_candidate_transient_runs':0,
        'new_reference_transient_runs':0,'new_operator_quadrature_evaluations':0,
        'finite_node_candidate_admitted':False,'contact_transient_admitted':False,
        'next_mass_average_projection_review_predeclared':next_plan,'next_task':'same-kernel mass-average projection weak budget, theory only',
        'remaining':['mass-average projection and finite-node quadrature budgets','subcell density closure and finite-input identifiability',
            'transient convergence and general geometry, shocks, walls, ledger, nonplanar/DNS acceptance'],
        'report':'research/reconnect/ORACLE_WEAK_INTERFACE_CONSISTENCY.md','pdf_generated':False,'software_version_created':False,'published':False}
    (out/'stage3-status.json').write_text(json.dumps(status,indent=2,ensure_ascii=False)+'\n')
    p=root/'docs/RESEARCH_STATUS.md';text=p.read_text().replace(
        '47. [同一oracle核的二阶界面一致性理论审查](../research/reconnect/ORACLE_INTERFACE_SECOND_ORDER.md)（本轮）。',
        '47. [同一oracle核的二阶界面一致性理论审查](../research/reconnect/ORACLE_INTERFACE_SECOND_ORDER.md)。\n48. [同一oracle核的弱界面一致性与显式一阶项](../research/reconnect/ORACLE_WEAK_INTERFACE_CONSISTENCY.md)（本轮）。')
    text=text[:text.index('本轮关键进展：')]+'''本轮关键进展：完成预定同一oracle核的弱形式审查。对界面连续、分相C3且法向Gamma梯度连续的场，以及固定全局光滑测试，纯相误差O(h²)、一般界面弱误差O(h)。精确对称化保留质量测度与原生成元因子2，双向配对积分不再乘1/2。

进一步推导显式界面主项：W_h-W=h*kappa_tau*integral_Sigma(c_tau*phi_tau)+O(h²)，给出带常数余项界；原1:4密度反差下kappa_tau严格为负。纯横向光滑解析见证使主项非零，故一般一阶界可达到。上轮2.5/0.625点态迹和相邻物质平均缺口保留；固定光滑测试弱收敛与局部偏差可同时存在。

完整rho仍为额外oracle信息；没有把该弱定理转称为有限点图、质量平均投影、耗散或瞬态收敛，也未重写原余弦态误差。候选、原总作用、身份/质量/c0和PDE均保持。新增8组标准库精确代数证书独立复算一致，无新增数值/流动测试，重建检查累计仍352项；旧v0.8.0/246项未恢复。

782原文件、15历史附件和47份先前正式科学清单保持，4份历史诊断清单已有路径差异另核。下一步只做同一核质量平均投影的弱预算：理想单元对积分仅为额外几何oracle理论参照，不计算新边权/构图，不将真实几何加入点输入；区分投影、有限点求积及密度闭合。普通中心阻力FV基线保持。

数据入口：runs/reconnect-oracle-weak/stage3-status.json。Stage 3 incomplete，fluid ORC关闭；无PDF/新版本、提交/推送、环境配置修改或发布。
'''
    p.write_text(text)
    print(json.dumps({'weak_consistency_proved_within_scope':True,'sharp_generic_order':'O(h)',
        'kappa_tau':status['kappa_tau_original_density_ratio'],'exact_algebra_groups':8,
        'recorded_checks_total':352,'next_task':status['next_task']},indent=2,ensure_ascii=False))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args();main(a.output)
