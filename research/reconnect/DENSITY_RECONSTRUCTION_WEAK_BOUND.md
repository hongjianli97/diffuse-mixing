# 点密度重构的连续弱扰动界

2026-11-01（UTC）。执行[路线决定](CANDIDATE_ROUTE_DECISION.md)中预定的条件性理论工作。Stage 3 **incomplete**；fluid ORC **关闭**。

本轮证明了所需的 **h 一致弱扰动界**。对固定 Lipschitz 浓度和测试函数，重构误差通过逆扩散系数的 L1 范数进入连续双线性型；在明示的分片光滑密度、界面邻域面积和点云填充距离条件下，最近点重构产生 O(epsilon) 的弱扰动。

这补上连续密度闭合的一项误差控制，**没有证明有限点图收敛或瞬态通过**。已有二阶界面反例、物质平均投影缺陷和节点求积缺项保留。本轮不构图、不计算新边权或数值算子积分。

## 1. 固定模型、测度与核

保持[条件性输入契约](CONDITIONAL_DENSITY_CLOSURE.md)：二维周期矩形 Omega，单位物理厚度 b=1 m，空间常数 D>0，核支撑 2h < min(periods)/2。真密度 rho 和重构密度 rho_hat 都满足

\[
0<\rho_-\le \rho,\hat\rho\le\rho_+<\infty.
\]

记 Gamma=D rho，Gamma_hat=D rho_hat，Gamma_+=D rho_+，a=1/Gamma，a_hat=1/Gamma_hat。密度只须可测且满足上述正界，就足以得到第2节的估计。rho_-、rho_+ 是真场的先验全局界，不能只用有限样本极值代替。给定周期欧氏最近点重构的密度值均来自真点密度，因此沿用相同正界；等距按固定身份取值。重构分区不是物质单元。

浓度 c 与测试函数 phi 是固定的全局周期 Lipschitz 函数，常数分别为 L_c、L_phi，不随 h 或采样尺度改变。它们是用于一致性分析的连续场，**图的未知量仍是原固定物质单元的质量平均**，不是 c(x_i)。

对唯一最小镜像位移 r，定义

\[
K_a(x,x+r)=\left[\int_0^1a(x+\theta r)\,d\theta\right]^{-1},
\qquad
W_h^a(c,\phi)=b\int_\Omega\int B_h(r)K_a(x,x+r)
\delta_r c(x)\delta_r\phi(x)\,dr\,dx.
\]

重构形式 W_h^{a_hat} 只把 a 换成 a_hat。保留原 B_h、原因子2及支撑，引用既有核矩

\[
B_h\ge0,\qquad \int B_h(r)|r|^2\,dr=2.
\]

双向配对积分前不再乘1/2，见[既有弱形式证明](ORACLE_WEAK_INTERFACE_CONSISTENCY.md)。若写生成元，两个形式都可用**同一真质量测度** b rho dx：令 L_h^{a_hat;rho}c=2/rho * integral K_{a_hat}B_h delta c，则 -b integral rho phi L_h^{a_hat;rho}c=W_h^{a_hat}。rho_hat 仅改变阻力系数，不替换固定质量测度，更不重归一化质量。

## 2. 对 h 一致的弱扰动定理

令 A(x,r)=integral_0^1 a(x+theta r)dtheta，A_hat 同理。它们均不小于1/Gamma_+，故

\[
|K_{\hat a}-K_a|
=\frac{|\hat A-A|}{\hat A A}
\le\Gamma_+^2\int_0^1|\hat a-a|(x+\theta r)\,d\theta.
\]

由 |delta c delta phi| <= L_c L_phi |r|^2 得

\[
|W_h^{\hat a}-W_h^a|
\le b\Gamma_+^2L_cL_\phi\int B_h(r)|r|^2
\int_0^1\int_\Omega|\hat a-a|(x+\theta r)\,dx\,d\theta\,dr.
\]

每个固定 theta、r 的周期平移保持面积测度。非负被积函数允许 Tonelli 交换积分；有界密度和有限核二阶矩保证结果有限。因此

\[
\boxed{|W_h^{\hat a}(c,\phi)-W_h^a(c,\phi)|
\le 2b\Gamma_+^2 L_cL_\phi\,\|\hat a-a\|_{L^1(\Omega)}.}
\tag{1}
\]

常数不含 h，也不含 epsilon/h。这来自差分提供的 |r|^2 与原核二阶矩；若先丢掉差分，再用 integral B_h 的 O(h^-2) 界，会得到本处不需要的发散粗界。没有因此修改核归一化或边权。

另有较粗的密度版本

\[
\boxed{|W_h^{\hat a}-W_h^a|
\le 2bD\left(\frac{\rho_+}{\rho_-}\right)^2
 L_cL_\phi\,\|\hat\rho-\rho\|_{L^1(\Omega)},}
\tag{2}
\]

因为 |a_hat-a|=|rho_hat-rho|/(D rho_hat rho)。D=0 独立定义为零交换，两种形式都为零，不计算无穷阻力。D>0 的常数随 D 线性缩放，不是对 D 或 rho 的拟合。

## 3. 最近点重构的 L1 控制及所需先验

设 Sigma 包含所有相界面，点不位于未定义密度的界面上；点密度是原真场值。设对0<t<=epsilon_0，周期管状邻域 T_t={dist(x,Sigma)<=t} 满足

\[
|T_t|\le C_\Sigma t.
\]

相内 a 有沿不跨界短线段成立的 Lipschitz 界 L_a，可由相内 rho 的梯度/Lipschitz 界 L_rho 给出 L_a <= L_rho/(D rho_-^2)。要求点云填充距离不大于 epsilon<=epsilon_0：任何位置到最近输入点的周期距离至多 epsilon。

当 x 不在 T_epsilon 中时，从 x 到最近点的短线段不会碰到 Sigma，故该点与 x 同相，|a_hat(x)-a(x)|<=L_a epsilon。管状邻域内只使用正密度界，令

\[
\operatorname{osc}_a=\frac1{D\rho_-}-\frac1{D\rho_+},
\]

则误差至多 osc_a。最近点等距集合的取值不影响面积积分。因此

\[
\|\hat a-a\|_1
\le L_a\epsilon\,|\Omega\setminus T_\epsilon|
+\operatorname{osc}_a|T_\epsilon|
\le\bigl(|\Omega|L_a+C_\Sigma\operatorname{osc}_a\bigr)\epsilon.
\tag{3}
\]

合并得到

\[
\boxed{|W_h^{\hat a}-W_h^a|
\le C_{rec}\epsilon,\quad
C_{rec}=2b\Gamma_+^2L_cL_\phi
\bigl(|\Omega|L_a+C_\Sigma\operatorname{osc}_a\bigr).}
\tag{4}
\]

这些常数须在所讨论的采样族中统一有界。填充距离不是物质单元直径；有限点密度也不能自行证明相内正则性和界面面积控制。亚采样结构若随采样加密而使这些常数增长，就不能据(4)宣称误差趋零。有限输入不可辨识性结论没有被推翻。

### 原1:4平面周期接触的纯解析特化

仅将原已存单位周期背景的 rho_-=1、rho_+=4、D=1/50 代入上述**理论常数**。两条长度为1的平面界面在 epsilon<1/4 时有 |T_epsilon|=4 epsilon，且相内 L_a=0。因此 Gamma_+=2/25、osc_a=75/2，并有

\[
|W_h^{\hat a}-W_h^a|
\le\frac{48}{25}\,bL_cL_\phi\,\epsilon.
\tag{5}
\]

坐标和物性单位沿用原背景。48/25 是相应单位下的保守解析系数，**不是测得误差、收敛率拟合或验收阈值**；本轮未计算任何点云 epsilon、路径系数、算子作用或旧 NPZ 的新指标。

## 4. 与局部PDE相接：可合并的结论及边界

只在[既有完整密度弱定理](ORACLE_WEAK_INTERFACE_CONSISTENCY.md)的平面、相内常密度、连续 c、法向 Gamma 梯度连续、分相 C3 浓度和固定全局 C3 测试假设下，引用其结果而不重证：

\[
|W_h^a-W_{PDE}|\le b\{12\Gamma_+L_cL_\phi\ell_\Sigma h+\mathcal B(h)\},
\quad\mathcal B(h)=O(h^2).
\]

所以该模型中两项连续误差满足

\[
\boxed{|W_h^{\hat a}-W_{PDE}|
\le C_{rec}\epsilon+
b\{12\Gamma_+L_cL_\phi\ell_\Sigma h+\mathcal B(h)\}.}
\tag{6}
\]

这是固定测试下的**条件性连续弱一致性 O(epsilon+h)**；没有要求 epsilon/h 趋零。它没有把原完整密度平面弱定理扩展到任意曲界面、变密度 PDE 或运动问题，也没有宣称常数未知的有限算例已满足某个数值容差。

若以后连接物质平均图，仍须区分

\[
E_{point,rec}-W_{PDE}
=(E_{point,rec}-E^*_{rec})
+(E^*_{rec}-W_h^{\hat a})
+(W_h^{\hat a}-W_h^a)
+(W_h^a-W_{PDE}).
\tag{7}
\]

两种离散形式均作用于 cbar_i=(P_rho c)_i、phibar_i=(P_rho phi)_i。E_point,rec 使用输入契约中的点对电导与原 A_i^q=m_i/(rho_i b)；E^*_{rec} 只在理论上将电导换成 2b*integral_Ki integral_Kj K_{a_hat} B_h dx dy，故等于 W_h^{a_hat}(P_rho c,P_rho phi)。本轮没有生成任一新图。

第一项仍是点对求积及质量/点密度面积权缺陷，第二项仍是物质平均投影闭合。既有投影界依赖最大真实单元直径 Delta/h，其证明只需核系数正对称有上界，重构系数也满足这一条件；引用该预算不等于已经控制 Delta 或第一项。

必须先单列投影项：P_rho c、P_rho phi 通常是有跳跃的分片常数，不能直接将它们代入(1)并沿用原连续场的 Lipschitz 常数。

这四项有符号相加；其范数不能当作相加份额，更不能利用抵消宣布通过。固定比例的投影反例和界面强二阶偏差仍保留。若测试函数随 h 集中到单元/界面，L_phi 可能发散，(1)不提供统一强局部结论。

## 5. 为什么连续 L1 界不直接控制抽样边

一个纯范数例子说明边界：在单位二维周期域取 a_0=1，令 a_tau=2 仅在矩形 31/64<x<33/64、|y-1/2|<tau/2 内，其他地方为1，0<tau<1/2。则

\[
\|a_\tau-a_0\|_1=\tau/32\longrightarrow0.
\]

但固定线段 gamma(theta)=(15/32+theta/16,1/2) 有一半参数区间落在该矩形中，沿段阻力平均恒为3/2，所以 K_tau=2/3，而 K_0=1；差恒为1/3。该线段长度1/16小于原支撑2h=0.08，最小镜像唯一。**面积 L1 控制不能单独控制指定线段上的迹/路径平均。** 连续配对积分的平移平均正是第2节能够绕过这个问题的原因。

这个例子不是所选最近点重构在既定点云上的反例，也不声称给出同固定质量的两套完整流体状态；它只排除“把连续 L1 界直接逐边代入”的无根据推断。实际点对求积仍需其采样与闭合条件，不能用本轮定理替代。

## 6. 实现、核验与下一步

新增标准库 `density_weak_certificates.py`，核对3组精确代数：逆平均敏感性的正多项式余量、原平面背景的新界系数及D缩放、L1/固定路径区别的解析分数。测度、Tonelli、最近点同相性与函数空间证明写在本文；程序不是完整自动定理证明器，也没有重新证明旧核矩/弱界面/投影定理。

来源审计和独立进程结果一致。782个原文件、15个历史附件及54份先前正式科学清单保持，4份历史诊断清单既有差异另核。新增3组代数核验另列，无新增数值/流动测试，历史记录仍 **363项**；旧v0.8.0/246项未恢复。原固定质量、物质身份、c0、原图、参考、FV基线和已封存文件均未改。

下一步只作**有限节点求积连接的条件审查**：以(7)前两项为对象，明确允许的点输入和已有先验究竟能控制什么；尤其不能把 L1 密度误差当成逐边迹界，不能用填充距离代替真实物质直径。先给出可证明的条件或障碍，再决定是否值得编码候选。不重复旧 LP、强二阶反例或物理算例，不构图、不调整权重，不进行新算子数值求积或瞬态。

```bash
PYTHONDONTWRITEBYTECODE=1 /workspace/.venvs/ltn-reference/bin/python research/reconnect/density_weak_certificates.py --output /tmp/ltn-density-weak
python research/reconnect/density_weak_certificates.py --certificates-only
```

第一条要求完整历史工作区且输出不存在；第二条只需当前标准库脚本。数据及审计入口为 `runs/reconnect-density-weak/`。未生成PDF/新版本、提交、推送、更改环境配置或发布。

[本轮理论与审计包](../../exports/ltn-density-weak-review.zip)；[交付范围与哈希](../../exports/ltn-density-weak-delivery.json)。
