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


101. [同一周期接触oracle三档物质空间对照](../research/reconnect/CONTACT_SPATIAL_COMPARISON.md)（2026-11-04）。

执行原N16/32/64登记项，只补N16/N64两张静态点图及K16实际/完整独立序列；两图先封存再读m/c0/参考/旧状态，N32仅加载原结果保持。决定T4_three_space_level_static_contact_oracle_control_passed，原生/共同浓度与原生/共同H两相邻档两个时刻改善均超过原正项指标。终时共同误差0.00176306246144、0.000417790234342、0.000102495677257，非一般阶证明。全部原几何/普通阻力/时间/H源码字节不改，128模态90位系数复用且新有限均值按100/120位评价，独立80/100位及新网格各17端点Padé审计通过。

三项新检查、主计算和完整审计首次通过；旧493未重跑，记录496。782原文件、15附件、100份旧正式清单和原N32失败恢复保持。下一项只登记原δ.08/D.02/T.1/32列Becker oracle接口与最小判别：先写移动外边界x零通量/y周期、T显式时序包装并在T.05夹具匹配旧时序，再按封存协议构图/传播，复用原质量/c0/参考；本轮没有新Becker点/图/状态/参考评价。Stage3仍incomplete，fluid ORC关闭，三密度尚未闭合。按已授权生成PDF/代码结果ZIP和GitHub备份，不发布环境/版本。


102. [移动Becker点FV oracle：外边界/显式时长接口与未决判别](../research/reconnect/BECKER_POINT_COMPARISON.md)（2026-11-04）。

执行原δ.08/D.02/T.1/32列×8行、K4/8/16单例；新增可选有界x/周期y精确点几何与T显式包装，旧构造器/指数/统一化/H代码不改，T.05夹具逐位匹配旧路径。28图全部先封存再读原m/c0/参考；3实际/完整独立路径，参考不新解。原生K16误差0.000413042910329。状态时间控制已分辨，H下降5.76279599929e-07未超过原指标4.85814408707e-06，决定moving_Becker_control_failed_or_unresolved，不宣告通过，不扩N/K或改H/阈值。

4项新软件检查及完整独立审计首次通过，记录500，旧496不重跑。主流程在体积诊断有理数解析时失败，全部科学结果已存；原失败/源码/JSON保留，追加补全只修解析，109个原文件保持、不重算传播。原782文件、15附件、101旧正式清单保持。下一项只登记保存H/参考预算分量只读复核，0新图/候选/PDE/软件检查。Stage3 incomplete，fluid ORC关闭，κ未算；按授权提供PDF/代码ZIP与GitHub备份，不发布环境/版本。


103. [移动Becker保存H预算只读复核](../research/reconnect/BECKER_BUDGET_REVIEW.md)（2026-11-04）。

保持原未决/H公式与阈值；原状态预算99.818674%来自参考空间差，非参考项单独原H诊断值5.21511144e-9。按同一公式反解，参考空间+时间指标约需缩小8.50倍才有分辨余地。此为工作规划，不追认旧通过、不认证PDE误差界。0新图/候选作用/PDE/状态/软件测试，累计仍500；主复核与独立80/100位账目审计首次通过。782原文件、15附件、102旧正式清单和历史诊断差异保持。

仅登记独立参考L形三点(16384,8192)/(32768,8192)/(32768,16384)，候选N/K/行数/中心/参数不变，尚未执行。保留参考空间/时间/投影正项与原H公式；失败或未决即停，不自动加密或改阈值。Stage3 incomplete，fluid ORC关闭；本轮本地PDF/代码ZIP交付，公开推送待明确授权，不发布环境/版本。


104. [固定Becker参考L形核验：完整H判别仍未决](../research/reconnect/BECKER_REFERENCE_CHECK.md)（2026-11-06）。

执行已登记三组参考(16384,8192)/(32768,8192)/(32768,16384)，原参考/Becker源码不变，0候选图/作用/重跑。参考空间+时间指标5.956372e-9，但预先固定初值投影正项使原公式H指标9.468091e-7仍高于下降5.762796e-7；保持两次未决并停止加密。主汇总因旧H逐位一致断言失败（复算差2.22e-16），21文件/失败源码保持，追加汇总使用旧H并记录复算差，无额外PDE。独立精确投影/80与100位预算审计通过；原782文件、15附件、103份旧正式清单保持，累计500。下一项仅初值投影含义只读复核，Stage3 incomplete、fluid ORC关闭；仅本地交付，不发布/推送。


105. [初值物质平均投影只读复核](../research/reconnect/INITIAL_PROJECTION_REVIEW.md)（2026-11-06）。

31个内部物质边界均切在细参考格内；解析边界缺陷恒等式解释5.89553258e-9/1.08801426e-9表示差，浮点贡献约1e-17到1e-16。80/100位主复核与独立sinc部分积分100/120位审计首次通过，整格均值不确定部分积分的有理数见证成立。不改变物理初值、旧正项或两次未决，0新PDE/候选/软件测试，累计500；782原文件、15附件、104旧正式清单保持。仅提出完整子物质区间积分的参考观测契约，下一项隔离非均匀参考原型及两个小型软件夹具，尚未执行新Becker参考。Stage3 incomplete，fluid ORC关闭；仅本地交付。


106. [完整物质子单元参考原型](../research/reconnect/ALIGNED_REFERENCE_PROTOTYPE.md)（2026-11-06）。

隔离非均匀质量参考实现完成，两项预定小型软件检查首次通过，累计502，旧500未重跑。均匀极限与旧参考逐位一致；完整子积分/嵌套/常数/D0及CN两种拒绝策略通过。7次合成调用、33次浮点线性求解，含2次预期拒绝；monitor大步长原始最小值−0.734546并保留。独立有理数重建25个保存方程及80/100位初值审计首次通过，没有再次时间积分。任意非均匀二阶或CN无条件极值保持均不声称。0新Becker参考/候选，旧H未决与预算保持；782原文件、15附件、105旧正式清单保持。下一项仅登记对齐参考L形三点及原正项/H核验，尚未执行；Stage3 incomplete、fluid ORC关闭。仅本地PDF/代码ZIP交付与Git备份，不发布/推送。


107. [物质对齐Becker参考：保存候选时间改善已分辨](../research/reconnect/ALIGNED_BECKER_REFERENCE_COMPARISON.md)（2026-11-06）。

完成冻结的(16395,8192)/(32790,8192)/(32790,16384)三组参考，原32列×8行K4/8/16候选复用，0新候选/软件检查、累计502。完整子积分使初值桥约1.7–1.9e-16；原H指标4.668887e-8低于保存下降5.762796e-7，单网格时间改善可分辨，K16相对误差4.130620e-4。两次旧未决/H预算保持，不追认。三组原守恒/极值守卫通过，但CN充分非负条件在所有步未满足，monitor不作无条件最大值声明；最高质量漂移8.87e-11。独立精确积分/几何质量与80/100位预算审计首次通过，主流程90.439秒首次通过。782原文件、15附件、106旧正式清单保持。下一项仅登记同后端32/64/128列最小空间控制，固定8行，原32路径复用，只补56图/6实际及独立路径和以128父区间为基的3个参考解，空间指标必须计入时间差；本轮未执行。Stage3与三密度闭合仍未完成，fluid ORC关闭。仅本地PDF/代码ZIP和Git恢复备份，不发布/推送。


108. [移动Becker同后端三列档空间对照](../research/reconnect/BECKER_SPATIAL_COMPARISON.md)（2026-11-06）。

64→128列浓度改善尚未分辨，空间控制未通过，决定T4_moving_Becker_spatial_control_failed_or_unresolved。原32列路径复用，只补64/128列、固定8行及K4/8/16；56新图先封存，6实际/6独立路径，原128父区间3个对齐参考。两相邻档原生/共同浓度和H改善按固定正项比较，显式计入K8→16时间差；不改原H/阈值、质量/状态或旧代码。主计算和独立几何证明/共享面率/完整序列/精确积分/80与100位预算审计首次通过；0新软件检查、累计502。782原文件、15附件、107旧正式清单保持。下一项仅只读分解保存的未决空间/时间预算，定位时间、参考、数值或空间改善符号障碍；不新增N/K/行数/参考/密度，不修改本轮判据或追认通过。 Stage3 incomplete，fluid ORC关闭。仅本地PDF/代码ZIP与Git恢复备份，不发布/推送。


109. [Becker空间未决预算只读复核](../research/reconnect/BECKER_SPATIAL_BUDGET_REVIEW.md)（2026-11-07）。

时间项占未决原生/共同浓度预算99.9808%/99.9800%，原六项分辨、两项未决保持；0新图/状态/PDE/密度/软件检查，累计502。机械后移H判据受旧预算下限阻碍，不能承诺K32使完整Stage3通过。两档更新的充分界曾错误用于单档方案，原代码/结果/算术审计保留，以精确反例和追加审计纠正。下一项登记仅128/K32一条实际/独立路径和32图，64/K16控制及参考复用；固定正项、H和严格比较不改，混合时间诊断不等于三档空间验收，尚未执行。782原文件、15附件、108旧正式清单保持。Stage3 incomplete，fluid ORC关闭；仅本地PDF/精选ZIP和Git恢复备份，不发布/推送。


110. [单一128列K32浓度时间诊断](../research/reconnect/BECKER_FINE_TIME_COMPARISON.md)（2026-11-07）。

单档浓度时间诊断已分辨，决定single_finest_K32_concentration_diagnostic_resolved。仅32新图先独立核验封存，1实际/1完整独立路径，原64/K16对照及全部旧参考复用，0旧重跑/新参考/密度/软件检查，累计502。新旧共有33采样帧逐位一致，旧全部正项/H公式/阈值保留。原三列档空间未决、旧K4/8/16通过及后移H下限障碍保持；混合时间诊断不等于完整空间或Stage3验收。主流程与独立保存序列/精确积分/80与100位预算审计完成；782原文件、15附件、109旧正式清单及全部历史修正保持。只读整理已保存的混合时间浓度/H空间证据、原K4/8/16时间资格与后移H下限障碍，逐条对应原Stage3要求；明确仍欠缺的证据，不能把本轮诊断追认为原三档空间通过或豁免H判据。本项0新图/传播/参考/软件测试；只有证据支持后另行登记最小下一项。 Stage3 incomplete，fluid ORC关闭。仅本地PDF/精选ZIP与Git恢复备份，不发布/推送。


111. [Stage3证据与H障碍只读复核](../research/reconnect/STAGE3_EVIDENCE_REVIEW.md)（2026-11-07 UTC）。

64份封存来源、12条原文和10组案例状态核对，80/100位正项比较独立复算；原K16空间6项分辨/2项未决、旧K4/8/16时间通过、混合K16/K32四项分辨及后移H未决均保持。同后端独立层厚度与三密度来源仍未闭合。0新图/路径/参考/密度/软件测试，累计502；782原文件、15附件、110旧正式清单及四组历史差异保持。下一项已冻结共同K32三档空间诊断，只补32/64两条路径，复用128/K32和全部参考；本轮未执行，不豁免后移H、不追认旧空间门。仅本地PDF/精选ZIP与Git恢复点；Stage3 incomplete，fluid ORC关闭，不发布/推送。


112. [共同K32三档空间诊断](../research/reconnect/BECKER_COMMON_K32_COMPARISON.md)（2026-11-07 UTC）。

共同K32三档空间诊断已分辨，决定common_K32_three_column_spatial_diagnostic_resolved。仅32/64两条新路径，128/K32、旧K4/8/16和全部参考复用。64新图先审计封存，2实际/2完整独立路径，0旧重跑/新参考/密度/软件检查，累计502。八项空间分辨=True；后移H结果={'32': True, '64': False, '128': False}。全部旧正项/H公式/阈值及原K16空间未决保留，不自动升级完整Stage3。保存序列、完整物质观测和80/100位预算独立审计通过；782原文件、15附件、111旧正式清单及历史修正保持。下一项尚未执行：只读定义一个最小独立层厚度δ对照：从原Becker场和已保存物质标签/质量平均定义出发，明确变化δ时哪些物质标签、初值、边界和物理量可固定、哪些须重建；冻结参考/时间/空间误差计账和资源预算，不预设δ变薄后误差必须降低，不因缺失H后移分辨而改原阈值。最多提出一个新增δ值，不运行新路径或参考PDE。 仅本地PDF/精选ZIP与Git恢复点；Stage3 incomplete，fluid ORC关闭，不发布/推送。


113. [固定物质质量坐标的层厚度对照定义](../research/reconnect/BECKER_DELTA_CONTROL_DEFINITION.md)（2026-11-07 UTC）。

只读定义完成，未采样或计算新δ。固定旧μ标签/M*/m/c0，重建新δ轨迹、密度、物质边界与参考终态；不能同时固定原Eulerian区间/初场。明确旧flow.total断言和N32旧H复用分支不能用于新δ驱动。冻结下一项唯一δ=.04、32列×8行、K8/16/32，最多56图、3实际/3独立与3新参考；旧参考网格/c0复用，主900秒/1024MiB、参考300秒。原状态/H时间资格与全部正项不改，误差增减均报告，不预设变薄改善，不宣称新δ三档h或δ→0。26封存来源、旧输入及代数系数/量纲独立核验；782原文件、15附件、112旧正式清单和历史修正保持。当前0新图/路径/参考/密度/软件检查，累计502。下一项尚未执行；Stage3 incomplete、fluid ORC关闭，仅本地PDF/精选ZIP与Git恢复点，不发布/推送。


114. [单一δ=.04固定物质质量对照](../research/reconnect/BECKER_DELTA004_COMPARISON.md)（2026-11-07 UTC）。

新δ=.04的状态时间差下降已分辨，H时间差下降未分辨；合并时间资格=False，H响应方向一致=True。 56图封存后加载标量，3实际/3独立路径与3新参考终态；参考网格/m/c0逐位复用。旧全部正项/H/阈值不变，误差增减及响应幅值如实报告；新δ无三档h或δ→0验收。保存序列、精确物质汇总和80/100位预算审计通过；782原文件、15附件、113旧正式清单保持。0新软件测试，累计502，旧502未重跑。下一项：只读定位新δ时间/H或响应未决项：逐项列出状态、H、参考与数值正项，不减旧预算、不改阈值；先判断现有数据能否区分原因，不自动增加K、网格、δ或参考。 Stage3 incomplete，fluid ORC关闭，仅本地PDF/精选ZIP与Git恢复点，未发布/推送。


115. [δ=.04 H时间预算只读复核](../research/reconnect/BECKER_DELTA004_BUDGET_REVIEW.md)（2026-11-07 UTC）。

参考空间/时间项占B约87.217%，但机器项H下界7.9587e−9已高于H下降4.3935e−9；只细化参考不能解锁。保留全B时，后移K16/32/64下降上限3.0348e−9低于原指标6.2263e−8，不能通过。精确保存状态方差恒等式显示空间线性贡献抵消，高精度H差仍未决；不把余量等同实测浮点误差、不作物理误差因果分配。0新图/路径/参考/密度/软件测试，累计502；独立有理数和80/100位审计通过，782原文件/15附件/114旧正式清单保持。下一项仅定义已通过oracle的T1均匀平滑基准同后端三密度标定契约，不产生新数组或关闭Becker缺口。Stage3 incomplete，fluid ORC关闭；仅本地PDF/精选ZIP和Git恢复点，未发布/推送。


116. [平滑T1三密度来源契约定义](../research/reconnect/T1_DENSITY_SOURCE_DEFINITION.md)（2026-11-07 UTC）。

复用已通过oracle的N32静态周期T1，34来源哈希、旧1024节点/2048面/m/c0/参考和三条旧路径核对；原二维径向核固定h=1/16，不离散归一化，45项rho*仅符号未求值；新固定有向环ledger代理锚质量为2m，不能替换m，不执行旧F拟合/修补/回退驱动。下一项仅2字段/2系数包，ledger须通过逐项系数身份后引用旧oracle路径；只新增kernel的3实际/3独立路径，28区间，0新几何/参考PDE。两项软件组未来通过才502→504，当前仍502；当前0新密度/系数/路径。静态半群时间控制沿用原T1，不豁免Becker；物理偏差与实现控制分列。独立符号积分/来源/旧输入审计通过，782原文件/15附件/115旧正式清单保持。Stage3 incomplete，fluid ORC关闭，仅本地PDF/精选ZIP和Git恢复点，未发布/推送。


117. [平滑T1同后端三密度单例](../research/reconnect/T1_DENSITY_CALIBRATION.md)（2026-11-07 Asia/Shanghai）。

固定N32、D=.02、T=.05、K4/8/16，原m/c0/参考/几何复用。2密度/2系数包先独立核验封存，新增kernel的3实际/3独立路径、28区间；ledger2048精确系数身份后复用旧oracle三档。核密度1.0004157504550328有正偏差，终时共同分母浓度误差2.058734e-4低于oracle2.364108e-4约12.917%，两个非零时刻浓度/H改善均分辨；独立符号说明偏高密度部分补偿原FV衰减偏慢，不能宣称密度更准。18时间/27符号控制、两组必要软件检查及28区间/80和100位独立审计首次通过，累计504，旧502未重跑。代理锚质量2m原样保留，无F/归一化/拟合/修补。782原文件/15附件/116旧清单保持；Stage3 incomplete，fluid ORC关闭。下一项只读核对已有平滑压缩N64运动/非均匀密度推广契约，不自动算新场或扩展扫描。仅本地交付/恢复点，未发布或推送。


118. [平滑压缩N64三密度运动契约](../research/reconnect/COMPRESSION_DENSITY_DEFINITION.md)（2026-11-07 Asia/Shanghai）。

136来源哈希及48份保存几何/回执核对，4096节点、每图8192面，K16的33帧是K32的65帧逐位子集。两档均有64个x周期穿越，K32最后中点128面镜像标签改变，需显式整数绕行及初始固定环镜像；不修补原坐标。形式推导显示ledger有限间距伸缩含sinc(2pi/64)，不等于点oracle；非均匀初始代理质量比不再恰为2。原oracle/参考预算固定部分终时浓度3.30807e-6、H2.09946e-6，80/100位审计通过，未来还须加新来源正项。下一项冻结h=1/32、复用48旧几何，96密度/96系数包、kernel和ledger各K16/32，共4实际/4独立路径、96区间；0新几何/PDE/旧路径重跑。两组未来软件检查通过才504→506，当前0新物理数组/测试，仍504。782原文件/15附件/117旧清单保持，Stage3 incomplete、fluid ORC关闭。本轮Markdown及精选恢复包，不重复PDF，未发布/推送。


119. [平滑压缩N64同后端三密度单例](../research/reconnect/COMPRESSION_DENSITY_CALIBRATION.md)（2026-11-07 Asia/Shanghai）。

原m/c0/参考/轨迹/48几何及2oracle路径不变，h=1/32、D=.02、T=.05、K16/32。96新密度/系数先独立核验封存，再4实际/4独立路径、96区间；0新几何/PDE/旧路径重跑。kernel终时浓度误差5.265825e-5，比oracle6.361973e-5小17.23%，中时反大12.63%，两次相反方向变化均超过完整正项预算；H两时刻改善。ledger终时6.219366e-5，其浓度/H四项小幅改善均未分辨，状态来源差已分辨不等于更准。整数周期提升与初始固定环保持，有限环sinc偏差不修正；代理锚质量比不强制归一。两组必要软件检查及96区间/80和100位独立审计首次通过，累计506，旧504未重跑。主阶段207.465秒/115.70MiB。782原文件/15附件/118旧清单保持。下一项只读拆解保存误差交叉项、预算主项及无参考误差时的剩余下界，不自动扩展N/K/h/参考/模型。全部Becker缺口保持，Stage3 incomplete，fluid ORC关闭；PDF/精选恢复包及本地Git，未发布/推送。


120. [平滑压缩保存误差与预算诊断](../research/reconnect/COMPRESSION_DENSITY_DIAGNOSTICS.md)（2026-11-07 Asia/Shanghai）。

只读两来源×K16/32×两时刻的8行浓度/H恒等式，精确有理矩及80/100位4行完整预算核验通过。kernel两时刻交叉项都负，中时正平方项占优使误差增大，终时负交叉项占优使误差减小；H与浓度交叉项对象不同，不替代准确性验收。固定ledger观测收益时，假设参考空间/时间指标为零后的剩余预算仍超过全部4项收益，终时浓度/H为2.498/4.847倍；反向参考空间单项也已超过终时收益，不能将单独时间或参考精化当成充分方案。旧分类与全部正项不改，假设余量不作新判据。下一项只读清点更细资料、形成唯一ledger/oracle联合时间/参考控制契约，不自动增加N/K/h/参考/模型。0新物理字段/求解/软件测试，累计506；782原文件、15附件、119旧正式清单保持，Stage3 incomplete，fluid ORC关闭。本轮Markdown/精选恢复包，不重复PDF，未发布/推送。


121. [唯一N64 ledger/oracle联合时间与参考控制契约](../research/reconnect/COMPRESSION_JOINT_DEFINITION.md)（2026-11-07 Asia/Shanghai）。

168来源、原四参考和四对保存路径核对，N128→N64原物质平均层次经独立审核。冻结下一项：只补oracle/ledger K64，64新中点位置/几何，128密度/系数、2实际/2独立路径、128区间；只补n2048-k1024和n2048-k2048两参考，复用n1024-k1024，同基底空间/时间差，参考完整B记录。两项是给定相邻档下最少缺项，不是全局最优。新行与旧K16/K32判定分列，正项/阈值不变；必要三级时间检查中K32余量双计，不预言时间阶或通过。先冻结实现/两组软件/审计，未来两组通过才506→508；主1600秒/1024MiB，独立数值/参考审计400秒，软件120秒。当前0新物理字段/求解/测试，仍506；本轮契约与来源审核通过，782原文件/15附件/120旧清单保持。下一步实施唯一单例；失败/未分辨停止，不自动K128、参考4096或参数矩阵。Stage3 incomplete，fluid ORC关闭，Markdown及精选恢复包，无新PDF、未发布/推送。


122. [唯一N64联合时间与参考控制执行结果](../research/reconnect/COMPRESSION_JOINT_CONTROL_RESULTS.md)（2026-11-07 Asia/Shanghai）。

oracle/ledger各新增K64实际及完整独立路径，共128作用区间；64几何和128密度/系数预封存核验。仅新增n2048-k1024、n2048-k2048两参考，3072区间和完整B历史保存，原N128→N64物质观测不变。两组必要测试首次通过506→508；独立保存数据/80和100位预算审核通过。旧结果及预算不重判。

- t=0.025：浓度变化-8.368176243e-07，预算5.915658797e-07，error_smaller_resolvably；H变化-4.786166449e-07，预算4.094863222e-07，error_smaller_resolvably。
- t=0.05：浓度变化-1.478448638e-06，预算1.274228994e-06，error_smaller_resolvably；H变化-3.324187166e-07，预算7.699403564e-07，unresolved。

时间下降全部分辨：True；停止扩展：True。不自动K128/参考4096/扫描。782原文件、15附件、121旧清单保持。Stage3未通过，fluid ORC关闭，仅本地Markdown和精选恢复包，未发布/推送。


123. [联合控制后的证据归纳](../research/reconnect/JOINT_EVIDENCE_REVIEW.md)（2026-11-07 Asia/Shanghai）。

只读新K64保存场、128既有密度与新旧正预算。两个时刻的ledger浓度改善包含负交叉项抵消；最后系数时刻点相对Linf约0.0528514%。终时H预算为收益2.31618倍，固定收益下只消除参考/候选时间项均不足；保持原未分辨与停止决定，不减项改判。原规范证据账本更新，10旧状态及6后续记录核对，保留已有T1–T3/静态接触oracle/Becker共同K32空间证据，密度来源与H/层厚度缺口不隐藏。下一项只读核对并冻结同一光滑流t=63/1280、N64/128/256的密度一致性快照；N64字段复用，未来四个字段尚未生成，不自动运行标量或加密。精确矩及80/100位独立审核通过，0新物理/软件测试，仍508；782原文件/15附件/122旧清单保持。Stage3 incomplete，fluid ORC关闭，仅本地Markdown与精选恢复包，未发布/推送。


124. [单快照点密度空间诊断定义](../research/reconnect/DENSITY_SNAPSHOT_DEFINITION.md)（2026-11-07 UTC）。

47来源哈希和N64/128/256原m/ID/rho0/完整时钟/初始邻居及周期提升条件核验通过，N64字段原字节引用，细网格负y穿越不能遗漏。frame63实际时钟比名义63/1280大约5.55e−18，保留原浮点。固定h=2/N的均匀格点核求和与N无关，旧T1正偏差不随此比例细化消失；不将静态值移植为运动极限。下一项四字段契约已冻结，可直接实施N128/256各kernel/ledger一个，不再只做定义。点密度质量L2/Linf与生产—独立差指标、精确矩/80和100位核验、各600秒1024MiB及不降/未决停止规则固定，0新图/系数/路径/PDE；不改旧H/正预算和Stage3要求。本轮0新物理/软件测试，仍508，独立输入审计通过；782原文件/15附件/123旧清单保持。Stage3 incomplete、fluid ORC关闭；Markdown/精选ZIP/本地Git恢复，不发布/推送、不生成PDF。


125. [四字段单快照点密度空间对照](../research/reconnect/DENSITY_SNAPSHOT_COMPARISON.md)（2026-11-07 UTC）。

仅N128/256各kernel/ledger一个新字段，N64两旧字段字节复用；保存frame63时钟和h=2/N不变，初始环/镜像及双轴提升固定。全部六字段独立核验封存后才读未来oracle。kernel的四项网格比较中，下降0项、增大4项、未决0项；ledger的四项网格比较中，下降4项、增大0项、未决0项。决定stop_no_expansion，不增加网格/带宽/时间/参考或删项调阈值。六行精确矩、80/100位及八项网格比较独立通过，数值指标不冒称连续PDE总界。0新图/系数/路径/PDE/软件测试，仍508；54来源、782原文件/15附件/124旧清单保持。下一项仅只读分析保存有符号误差结构和固定h/Delta一致性机制，不重跑字段。Stage3 incomplete、fluid ORC关闭，旧H/Becker缺口保留；Markdown/精选ZIP/本地Git恢复，不发布/推送或生成PDF。


126. [保存密度误差结构与核一致性](../research/reconnect/DENSITY_ERROR_STRUCTURE.md)（2026-11-07 UTC）。

只读六个旧字段；kernel所有点偏高，均值偏差略降而中心化范数2.2448e−4→2.9021e−4增大，膨胀区平方误差占比85.1%→91.5%，两个E2²增长恒等式由空间项增幅超过均值项降幅解释，不当成因果分配。ledger全部点膨胀正/压缩负，两区条件均值约四倍下降。六场/896列/12分区/四恒等式独立展开精确矩和80/100位核对；原核归一化1、径向二阶矩14/33、平滑系数7/66由独立Beta积分核验。固定h/Delta有限格点极限及光滑条件下h→0、Delta/h→0充分尺度推导写明，未求新极限/卷积场、不声称激波适用。下一项只读定义固定旧h=1/32的材料求积对照与独立连续卷积参考，0新场/系数/路径/PDE/测试，不扫描调参，旧stop_no_expansion和所有预算保持。仍508；70来源、782原文件/15附件/125旧清单保持。Stage3 incomplete、fluid ORC关闭，Markdown/PNG/SVG/精选ZIP/本地Git恢复，未发布/推送或生成PDF。


127. [LTN 理论主线、实现与验证总览](PROJECT_OVERVIEW.md)（2026-11-07；文档整理，不是新实验）。

按用户“先不急着研究”和“重新整理”的要求，新增统一阅读入口，逐项对应原理论、数学对象、实际研究代码、保存证据与 Stage3 缺口。区分真物质面 FV 与点轨迹辅助交换面候选、图层性质与流体精度、原型未核对数字与本地重建结果、原验收范围与后续应用；近期密度诊断回归输入环节。当前暂停新增研究，126 条记录和封存 next-gate-spec 中的历史下一步不再自动执行。旧数值结果、预算和未决判定保持，0 新场/求解/软件测试，仍为累计 508 项记录；Stage3 incomplete、fluid ORC 关闭。原科学文件与附件不改，仅更新主报告、交接、追加历史索引和本地恢复元数据；不生成 PDF/版本或新结果 ZIP，不发布/推送。


128. [完整 PDF、代码整理与下一步计划](../START_HERE.md)（2026-11-07；研究交付）。

按用户明确要求生成综合 PDF 和小型源码包，增加理论证据附录、代码地图、统一完整性/检查入口与排版脚本，并准备保留远端历史的 GitHub 更新。原算法目录和科学结果不变；既有小图 10 项、物质 FV 5 项重跑通过，累计科学检查记录仍 508，不重复计数。下一项拟在一个平滑压缩流内验证 ledger 的瞬态空间一致性，再依次处理接触和 Becker 来源/层厚度缺口；此次只规划，0 新物理实验，旧固定带宽提案不自动重启。Stage3 incomplete，fluid ORC 关闭。原 782 文件保持，源码包与 GitHub 精选恢复的不同范围明确列出；历史大 ZIP 留在本地，不发布云环境。最终远端状态以交付回执为准。
