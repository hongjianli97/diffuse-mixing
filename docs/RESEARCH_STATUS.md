# LTN 重建研究当前状态

更新：2026-11-02（Asia/Shanghai）。本文件为可更新的进度入口；各轮报告、运行数据和原始导入材料分别保留哈希，不通过重写旧结果更新进度。

当前第三阶段总体状态：**incomplete**；fluid ORC：**关闭**。

已建立的证据链：

1. [Becker 物理定义与独立重建](../research/rebuild/REPORT.md)。
2. [共同状态的瞬时通量分解](../research/rebuild/FLUX_DIAGNOSIS.md)。
3. [全过程离散误差预算](../research/rebuild/TIME_ERROR_BUDGET.md)。
4. [固定 Becker 问题的三档物质空间细化](../research/rebuild/SPATIAL_CONVERGENCE.md)。
5. [独立接触的精确参考、跨界物质单元与双模态对照](../research/rebuild/CONTACT_TRANSMISSION.md)。
6. [初始浓度阶跃、早期层与交换量可识别性](../research/rebuild/CONTACT_STEP.md)。
7. [固定物质子单元与投影闭合对照](../research/rebuild/SUBCELL_CONTROL.md)。
8. [预先固定的局部网格分辨早期接触层](../research/rebuild/LOCAL_LAYER.md)。
9. [固定网格的实际隐式Euler时间控制与连续路径](../research/rebuild/FIXED_TIME_CONTROL.md)。
10. [新一维核交换同输入对照与固定邻居比例的一致性反例](../research/rebuild/KERNEL_ORACLE.md)。
11. [固定核宽控制、联合细化与接触面局部反例](../research/rebuild/KERNEL_JOINT_LIMIT.md)。
12. [界面残差弱矩与实际误差强迫分解](../research/rebuild/INTERFACE_LAYER.md)。
13. [连续内层、远场跳跃与无拟合预测](../research/rebuild/INNER_MATCHING.md)。
14. [FV非oracle平均密度与物质体积辨识](../research/rebuild/DENSITY_GEOMETRY.md)。
15. [固定物质面接口与离散采样输运](../research/rebuild/FACE_SAMPLING.md)，配套[输入约定](../research/rebuild/MATERIAL_FACE_INPUT.md)。
16. [二维固定单元面积与共享边输运](../research/rebuild/POLYGON_TRANSPORT.md)，配套[二维输入约定](../research/rebuild/MATERIAL_POLYGON_INPUT.md)。
17. [非正交物质单元的线性面通量与正性反例](../research/rebuild/NONORTHOGONAL_FLUX.md)。
18. [逐顶点多点通量与正系数模板方向限制](../research/rebuild/VERTEX_MPFA.md)。
19. [质量平均矩锥、宽邻域与横向漂移限制](../research/rebuild/MASS_MOMENT_CONE.md)。
20. [一次横向物质二等分的内点矩可行性](../research/rebuild/HALF_TRANSVERSE_CONE.md)。
21. [共享质量交换的全内点耦合不可行证书](../research/rebuild/SHARED_EXCHANGE.md)。
22. [固定共享图的显式矩缺陷预算](../research/rebuild/MOMENT_DEFECT_BUDGET.md)。
23. [独立零通量制造态的瞬时物理检验](../research/rebuild/NOFLUX_MANUFACTURED_PROBE.md)。
24. [真实面混合有限体积基线：准确性与正性分开检验](../research/rebuild/HYBRID_MATERIAL_FACE.md)。
25. [LTN主线回接：规范、实现与物理断点](../research/reconnect/MAINLINE_RECONNECTION.md)。
26. [二维点输入正交换核：固定oracle密度瞬时控制](../research/reconnect/POINT_KERNEL_2D_ORACLE.md)。
27. [二维点核一次预定联合细化与同物质单元比较](../research/reconnect/POINT_KERNEL_2D_REFINEMENT.md)。
28. [二维点核光滑变密度oracle状态与密度漂移检验](../research/reconnect/POINT_KERNEL_2D_VARIABLE_DENSITY.md)。
29. [同一变密度状态的连续核参照与三项误差预算](../research/reconnect/POINT_KERNEL_2D_CONTINUOUS_BUDGET.md)。
30. [光滑变密度点核一次联合细化及共同物质观测](../research/reconnect/POINT_KERNEL_2D_VARIABLE_REFINEMENT.md)。
31. [实际标量传播与质量掺混的有限时间对照](../research/reconnect/POINT_KERNEL_2D_DENSITY_TRANSIENT.md)。
32. [同质量同初态的光滑压缩非均匀点云冻结检验](../research/reconnect/POINT_KERNEL_2D_COMPRESSION.md)。
33. [同一压缩轨迹的中点冻结态与起止算子时间顺序](../research/reconnect/POINT_KERNEL_2D_COMPRESSION_MIDPOINT.md)。
34. [同一压缩流的独立时变材料PDE参考](../research/reconnect/COMPRESSION_TRANSIENT_REFERENCE.md)。
35. [同一压缩流的实际标量顺序传播与质量掺混对照](../research/reconnect/ORDERED_COMPRESSION_TRANSIENT.md)。
36. [同一压缩流的一档联合空间控制与共同物质观测](../research/reconnect/COMPRESSION_SPATIAL_CONTROL.md)。
37. [二维周期剪切冻结态：混合导数与密度漂移检验](../research/reconnect/PERIODIC_SHEAR_FROZEN.md)。
38. [同一二维剪切冻结态的单档空间控制](../research/reconnect/PERIODIC_SHEAR_SPATIAL_CONTROL.md)。
39. [同一二维剪切流的独立时变PDE参考](../research/reconnect/PERIODIC_SHEAR_TRANSIENT_REFERENCE.md)。
40. [同一二维剪切流的实际标量顺序传播与质量掺混](../research/reconnect/ORDERED_SHEAR_TRANSIENT.md)。
41. [同一二维剪切流的细档实际传播与共同物质区域比较](../research/reconnect/SHEAR_TRANSIENT_SPATIAL_CONTROL.md)。
42. [二维周期密度接触的一张冻结图检验](../research/reconnect/PERIODIC_DENSITY_CONTACT_FROZEN.md)。
43. [同一接触固定h连续核与物质平均误差预算](../research/reconnect/CONTACT_CONTINUOUS_KERNEL_CONTROL.md)。
44. [平面接触共享系数必要条件与有限输入可辨识性](../research/reconnect/CONTACT_CONSTITUTIVE_REVIEW.md)。
45. [同一接触oracle沿段串联阻力连续核对照](../research/reconnect/ORACLE_PATH_RESISTANCE_CONTROL.md)。
46. [同一oracle核的法向优先方向残差预算](../research/reconnect/ORACLE_DIRECTIONAL_RESIDUAL_BUDGET.md)。
47. [同一oracle核的二阶界面一致性理论审查](../research/reconnect/ORACLE_INTERFACE_SECOND_ORDER.md)。
48. [同一oracle核的弱界面一致性与显式一阶项](../research/reconnect/ORACLE_WEAK_INTERFACE_CONSISTENCY.md)。
49. [质量平均投影弱预算与固定比例闭合反例](../research/reconnect/MASS_AVERAGE_PROJECTION_WEAK_BUDGET.md)。
50. [原比例的一行理想质量平均二次响应](../research/reconnect/IDEAL_MATERIAL_ROW_RESPONSE.md)。
51. [同一二次探针的已存点图行比较](../research/reconnect/SAVED_POINT_ROW_COMPARISON.md)。
52. [已存接触两侧的零通量二次物质平均探针](../research/reconnect/SAVED_INTERFACE_PAIR_PROBE.md)。
53. [共享跨相单标量不相容证书与误差下界](../research/reconnect/CROSS_PHASE_SCALAR_COMPATIBILITY.md)。
54. [第三阶段候选路线决定](../research/reconnect/CANDIDATE_ROUTE_DECISION.md)，配套[条件性点密度闭合契约](../research/reconnect/CONDITIONAL_DENSITY_CLOSURE.md)。
55. [点密度重构的连续弱扰动界](../research/reconnect/DENSITY_RECONSTRUCTION_WEAK_BOUND.md)。
56. [点对求积与固定物质平均的条件性连接](../research/reconnect/POINT_PAIR_QUADRATURE_BRIDGE.md)。
57. [周期最近点密度的沿段逆密度积分原语](../research/reconnect/PERIODIC_DENSITY_PATH_PRIMITIVE.md)。
58. [三条封存路径的密度闭合核对与精确局部化](../research/reconnect/SAVED_DENSITY_PATH_CHECK.md)。
59. [普通中心阻力FV：原c0两行冻结对照](../research/reconnect/CARTESIAN_FV_TWO_ROW_CONTROL.md)。
60. [已存FV两行残差的面通量误差预算](../research/reconnect/FV_SAVED_FACE_ERROR_BUDGET.md)。
61. [普通FV的物质平均界面展开](../research/reconnect/FV_MASS_MEAN_INTERFACE_EXPANSION.md)。
62. [四物质单元线性界面重构：精确性与非负交换](../research/reconnect/FOUR_MEAN_INTERFACE_ADMISSIBILITY.md)。
63. [同一接触原c0：普通FV全域冻结对照](../research/reconnect/FV_GLOBAL_FROZEN_COMPARISON.md)。
64. [T=0.05独立质量平均参考](../research/reconnect/CONTACT_FINITE_TIME_REFERENCE.md)，配套[加权谱与尾项证明](../research/reconnect/CONTACT_PERIODIC_SPECTRAL_BASIS.md)。
65. [同一接触T=0.05：普通FV有限时间对照](../research/reconnect/FV_CONTACT_FIXED_TIME_COMPARISON.md)。
66. [普通FV：一档空间控制与共同物质观测](../research/reconnect/FV_CONTACT_COMMON_OBSERVATION.md)。
67. [点输入候选：闭合条件复核与下一道判别门槛](../research/reconnect/POINT_CANDIDATE_INFORMATION_REVIEW.md)。
68. [原c0完整两行：最近点密度路径候选](../research/reconnect/RECONSTRUCTED_PATH_PAIR_CONTROL.md)。
69. [同一快照全域冻结门槛：精确复用、范围与成本](../research/reconnect/GLOBAL_FROZEN_GATE_SCOPE_AND_COST.md)。
70. [原N128接触：路径候选的全域冻结比较](../research/reconnect/RECONSTRUCTED_PATH_GLOBAL_FROZEN.md)。
71. [全域冻结后的保存残差：有符号预算与解释边界](../research/reconnect/RECONSTRUCTED_PATH_RESIDUAL_SCOPE.md)。
72. [原c0两行：理想单元对积分与有符号误差分离](../research/reconnect/IDEAL_ORIGINAL_C0_PAIR_REFERENCE.md)。
73. [理想两行之后：路线判断与有限时间研究门槛](../research/reconnect/POST_IDEAL_ROUTE_DECISION.md)。
74. [原N128接触：保存路径点图的有限时间对照](../research/reconnect/POINT_PATH_FIXED_TIME_COMPARISON.md)。
75. [有限时间后的掺混偏差解释与路线收束](../research/reconnect/POINT_ENDPOINT_ROUTE_REVIEW.md)。
76. [点定义辅助面闭合：质量相容与平均缺陷](../research/reconnect/ORTHOGONAL_MATERIAL_MEAN_CLOSURE.md)。
77. [质量约束Laguerre分区：存在与正面距限制](../research/reconnect/LAGUERRE_MASS_INTERIORITY_REVIEW.md)。
78. [保留原质量的容量定理](../research/reconnect/CAPACITY_DEFECT_SEMIGROUP_BOUND.md)，配套[保存输入诊断与Stage 3验收路径](../research/reconnect/CAPACITY_INPUT_AND_STAGE3.md)。
79. [无激波优先：光滑压缩中点的辅助FV作用](../research/reconnect/SMOOTH_AUXILIARY_FV_FROZEN.md)。
80. [无激波二维剪切：非张量Voronoi辅助FV判别](../research/reconnect/SHEAR_VORONOI_FV_FROZEN.md)。

81. [同一光滑剪切：辅助FV实际时间推进与I/H](../research/reconnect/SHEAR_VORONOI_TIME_COMPARISON.md)。

82. [同一光滑剪切N256：共同物质区域空间控制](../research/reconnect/SHEAR_VORONOI_SPATIAL_COMPARISON.md)。

83. [光滑剪切16区间控制：空间收益已与时间指标区分](../research/reconnect/SHEAR_VORONOI_TIME16_COMPARISON.md)。

84. [光滑剪切N64物质并集：三档原生与共同误差控制](../research/reconnect/SHEAR_UNION64_COMPARISON.md)。

85. [光滑压缩：同一普通Voronoi FV实际4/8/16推进](../research/reconnect/COMPRESSION_VORONOI_TIME_COMPARISON.md)。

86. [光滑压缩：必要32区间时间控制](../research/reconnect/COMPRESSION_TIME32_COMPARISON.md)。

87. [光滑压缩N256：同一普通Voronoi FV空间对照](../research/reconnect/COMPRESSION_SPATIAL_FV_COMPARISON.md)。

88. [同一光滑压缩N64物质并集：三档原生与共同控制](../research/reconnect/COMPRESSION_UNION64_COMPARISON.md)。

89. [同一普通Voronoi后端：T1均匀介质解析基准](../research/reconnect/T1_UNIFORM_COMPARISON.md)。

90. [同一普通Voronoi后端：T2刚体平移客观性](../research/reconnect/T2_TRANSLATION_COMPARISON.md)。

91. [T2：D0非刚体物质输运与严格恒等检查](../research/reconnect/T2_D0_COMPARISON.md)。

92. [T2刚体旋转：周期域契约与最小边界适配方案](../research/reconnect/T2_ROTATION_DOMAIN_CONTRACT.md)。

93. [T2：精确旋转周期域的连续刚体旋转对照](../research/reconnect/T2_ROTATION_COMPARISON.md)。

94. [T3平面仿射压缩：恒D/恒Γ成对模型对照](../research/reconnect/T3_AFFINE_COMPARISON.md)。

95. [T3仿射压缩：保存误差预算复核与空间对照方案](../research/reconnect/T3_AFFINE_BUDGET_REVIEW.md)。

96. [T3仿射压缩：同一后端三档空间对照](../research/reconnect/T3_AFFINE_SPATIAL_COMPARISON.md)。

97. [同一普通FV后端：T1–T4与T3剪切覆盖复核](../research/reconnect/T3_COVERAGE_REVIEW.md)，配套[仿射剪切滑移周期域书面契约](../research/reconnect/AFFINE_SHEAR_DOMAIN_CONTRACT.md)。

98. [T3仿射剪切：给定滑移周期域的普通欧氏FV判别](../research/reconnect/AFFINE_SHEAR_COMPARISON.md)（本轮）。

执行上轮唯一N32/rho1/D=.02/T=.05/s=5t/K4/8/16仿射剪切登记项，新增可选精确滑移周期world欧氏Voronoi，旧矩形/正交适配/普通阻力/时间代码字节不改，s0调用原矩形分支。28图先封存再读原m/c0/参考，3实际及3完整独立路径，两个倾斜模态参考和原始观测不拟合、不归一化，原初态桥保留。

数值判别T3_affine_shear_control_passed。终时相对物质参考误差0.000342588344562，实际H0.146718819074761、参考H0.1473139037725；两非零时刻原始方向变化差均超过固定正项指标且与PDE相容，4/8到8/16状态/H差下降可分辨。单N32不声明新空间曲线或一般非正交/T3接纳。28图独立world顶点/面/阻力/面积/探针与逐单元全半平面检查、精确率/尾项/权重/完整独立序列及80/100位参考审计通过。

四项新软件检查首次通过，记录487，旧483未重跑。主计算149.100秒/95.71MiB至结果写JSON因NumPy布尔值序列化失败，原失败日志与不完整results.json保留；追加补全器仅从保存数据转换JSON标量类型，写results-completed.json，197个原文件和冻结实现保持，未重算传播或改判据。报告/图/重放均使用有效completed结果，不能称主流程首次全成功。782原文件、15附件、97份旧正式清单及旧交付保持，4份历史诊断差异保留，旧v0.8.0/246未恢复。

下一项只登记同一仿射剪切N16/32/64必要空间控制：复用N32三路径，只补N16/N64及原K4/8/16，56图、6实际/6完整独立路径，原N16共同区域与原生浓度/H、细档方向门槛均预定正项比较，1500秒/1024MiB；本轮未执行，不自动扩矩阵。Stage3 incomplete，fluid ORC关闭，kappa未计算，一般T3/光滑流/非正交/有限节点接纳false。报告：research/reconnect/AFFINE_SHEAR_COMPARISON.md；入口：runs/reconnect-affine-shear/stage3-status.json；下载：exports/ltn-affine-shear-downloads.md。原README/METHOD/CODEX_TASK/PHASE、附件与旧代码/数据/报告不改，仅更新本入口。无PDF、版本、提交推送、环境配置更改或发布，CRC/SHA和全新解压重放以交付JSON为准。


99. [仿射剪切：同一普通FV三档物质空间对照](../research/reconnect/AFFINE_SHEAR_SPATIAL_COMPARISON.md)（本轮）。

执行上轮固定N16/32/64、K4/8/16登记项，只补N16/N64，56图全部先封存再读原m/c0/参考或旧状态；6实际/6完整独立路径，N32三路径仅加载、原通过记录保持。几何/普通阻力/时间后端原字节不改，原生/共同浓度与H相邻改善、各档时间/符号/参考和细N64原始方向按预定正项指标核验。决定T3_three_space_level_affine_shear_control_passed，原生/共同/H空间收益分辨=True，细档方向分辨=True。原物质均值和初态/参考投影桥保留，不拟合、裁剪、归一化或改物理/H阈值。

首次软件检查1项预期布尔错误保留源码/日志，只更正测试预期，第二次3项通过；旧487不重跑，累计490。主计算首次669.281s/244.83MiB，独立几何/全部单元半平面/80–100位参考/原始权重和完整序列/固定母单元质量积分及正项预算审计通过，782原文件、15附件、98份旧正式清单保持；父轮JSON失败和补全前文件不改。

通过本轮即补齐原表T3给定压缩/剪切登记覆盖，不追加角度/模型/更细格扫描，不宣称通用非正交点云定理。下一项只登记同一最终点几何/时间后端N32周期接触oracle判别，1静态图复用、K16实际/独立路径，复用原128模态90位系数评价新N32有限均值并独立Cartesian半离散控制；本轮未生成T4输入/图/参考/状态。Stage3剩同后端T4和三密度来源闭合，incomplete，fluid ORC关闭，κ未算。无PDF、版本、提交推送、配置更改或发布。有效入口runs/reconnect-affine-shear-spatial/results.json；下载exports/ltn-affine-shear-spatial-downloads.md。


100. [同一普通FV周期接触oracle判别](../research/reconnect/CONTACT_ORACLE_COMPARISON.md)（2026-11-04）。

执行上轮N32/ρ1:4/D=.02/T=.05登记项，一张静态点图先封存再读m/c0/参考，K16实际和完整独立序列；原几何/阻力/时间后端不改。决定T4_same_backend_static_contact_oracle_control_passed，终时相对PDE误差0.000434110789229、相对Cartesian差7.14224526911e-15，原H和正项指标通过。复用128模态90位系数评价新真实有限均值，不插值或伪称恢复v080；独立80/100位、两个时间尾项、17端点Padé、精确率/尾项/权重和原始状态审计通过。

新增3检查第二次通过，首次N4审计夹具半径边界错误保留后仅换N8，不改阈值；记录493，旧490未重跑。主计算首次通过，首次完整审计因JSON读回Pythonfloat/原NumPyfloat求和末位差失败，追加类型恢复入口后精确复现原JSON；48个恢复前文件/首次代码与日志不改，第二次审计通过。782原文件、15附件、99份既有正式清单保持。

下一项仅登记同接触N16/32/64必要空间控制，只补N16/N64两静态图并复用N32；尚未执行。Stage3仍缺同后端完整T4和三密度来源闭合，fluid ORC关闭，κ未算。按用户本轮明确要求生成独立PDF与代码结果ZIP，未发布环境/版本或提交推送。
