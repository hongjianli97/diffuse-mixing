# 点对求积与固定物质平均的条件性连接

2026-11-01（UTC）。执行上一轮预定的求积连接审查。Stage 3 **incomplete**；fluid ORC **关闭**。

本轮得到两个结论：

1. 对下述分片常密度背景，若轨迹点位于其代表的真实物质单元内，真实单元直径 Delta 与核宽满足 Delta/h 趋零，则可以控制点对求积，再结合既有投影和连续弱预算，得到**固定测试下的条件性离散弱一致性**。
2. 点云填充距离趋零本身不够。一个由细长物质条带组成的解析例子满足精确质量/面积权、稠密点云，甚至填充距离/核宽趋零，但物质平均仍丢失横向浓度变化。不能用点间距代替真实物质单元尺度。

没有新图、边权、数值算子求积、LP 或瞬态。该定理不是通用点输入后端的物理接纳，也不声称原有限算例的误差已改善。

## 1. 原输入与新增的条件性几何先验

保持[输入契约](CONDITIONAL_DENSITY_CLOSURE.md)：周期二维矩形、单位厚度 b=1 m、空间常数 D>0、给定点位置/身份、固定 m 和真点密度 rho_i。密度重构仍为周期欧氏最近点插值，分区只用于密度闭合。

本轮求积定理限于真密度**分片正常数**的背景。设所有界面组成 Sigma，0<rho_-<=rho<=rho_+，R_rho=rho_+/rho_-，Gamma_+=D rho_+，且 |T_s(Sigma)|<=C_Sigma s 在所用尺度内成立。该密度界及管状控制是先验，不能仅由有限样本极值保证。传到局部PDE时只使用既有平面接触定理，不扩展到一般变密度或运动问题。

真实物质区域 {K_i} 是周期域的可测分割，V_i=|K_i|，并假设

\[
x_i\in K_i,\qquad
m_i=b\int_{K_i}\rho\,dx>0,\qquad
\operatorname{diam}K_i\le\Delta.
\tag{1}
\]

点属于其单元的陈述忽略零测边界。使用小尺度局部提升，要求2h+4Delta小于周期注入半径 min(periods)/2；界面管状控制覆盖2h+4Delta+epsilon。epsilon 是最近点填充距离的上界。证明中若采用实际填充距离，(1)保证 epsilon<=Delta，但不能反向推出 Delta<=C epsilon。

(1)是明确的**模型先验**，不是新增真实面/F/C输入，也不是有限点数据已识别出的几何事实。允许的候选接口不变；如果不能保证该先验，下面的定理就不能用于接纳该数据的离散后端。

## 2. 两个离散形式及面积权

状态与测试都采用原真质量平均：u_i=(P_rho c)_i，v_i=(P_rho phi)_i。c、phi为固定全局Lipschitz连续场，常数为L_c、L_phi。

记 q_i=m_i/(rho_i b)，K_hat为声明的最近点密度路径系数。保持原B_h、因子2和支撑，理论上定义

\[
E_{pt}=b\sum_{i,j}q_iq_jK_{\hat\rho}(x_i,x_j)B_h(r_{ij})
(u_j-u_i)(v_j-v_i),
\]

\[
E_* =b\sum_{i,j}\int_{K_i}\int_{K_j}
K_{\hat\rho}(x,y)B_h(y-x)(u_j-u_i)(v_j-v_i)\,dy\,dx.
\tag{2}
\]

两式采用双向配对，i=j因状态差为零。等价无向图电导为2b q_iq_j K_hat B_h；本轮只定义这些形式，没有生成边权。E_*=W_h^{a_hat}(P_rho c,P_rho phi)，其中a_hat=1/(D rho_hat)。

q_i不是自动识别的真实面积，但由真质量和正密度界有

\[
R_\rho^{-1}V_i\le q_i\le R_\rho V_i.
\tag{3}
\]

在完整单元处于同一常密度相时，q_i=V_i。还可逐单元展开而不重归一化，得到

\[
\sum_i|q_i-V_i|
\le(R_\rho-1)|T_\Delta(\Sigma)|
\le(R_\rho-1)C_\Sigma\Delta.
\tag{4}
\]

理由是 |q_i-V_i| <= rho_-^-1 integral_Ki |rho(x)-rho_i|dx；若x距界面大于Delta，连接x与x_i的短线段不跨界，两密度相等。其余点只用密度差上界。界面切入单元并未被改成单相，也没有调整质量或密度使q_i=V_i。

## 3. 本轮所需的新核正则性常数

原核写为B_h(r)=h^-4 beta(r/h)，支撑外为零，

\[
\beta(q)=\frac9{4\pi}\frac73(2+5q)(1-q/2)^5,\quad 0\le q\le2.
\]

直接求导得 beta'(q)=-35*(9/(4pi))*q*(1-q/2)^4。因此beta单调非增，且

\[
M_0=\sup\beta=\frac{21}{2\pi},\qquad
L_\beta=\sup|\beta'|=\frac{8064}{625\pi}.
\tag{5}
\]

后者利用q(1-q/2)^4在q=2/5达到最大值；其导数为(1-q/2)^3(1-5q/2)。延拓为零后仍有全局Lipschitz界，所以

\[
|B_h(r)-B_h(s)|\le L_\beta h^{-5}|r-s|.
\]

这里证明的是新增求积所需的核导数界，既有核矩、投影和弱界面证明不重跑。

## 4. 纯相与界面目标分开控制

令 t=Delta/h、R=2h+2Delta、w=2h+3Delta+epsilon。按目标代表点分成

\[
I=\{i:\operatorname{dist}(x_i,\Sigma)\le w\},\qquad I^c.
\]

这只是证明分区，不改变支撑、图或状态。若点核或某个单元对核非零，代表点距离必不超过R。对i不在I，相关K_i、K_j及其所有短路径都在距界面大于epsilon的同一相内；最近节点也在该相，所以**点路径和单元内路径系数均恰为该相Gamma，q_iq_j=V_iV_j**。这里没有对一般重构密度使用L1到逐边迹的错误推断。

保持同一u_j-u_i，有

\[
|u_j-u_i|\le L_c(r_{ij}+2\Delta),\qquad
|v_j-v_i|\le L_\phi(r_{ij}+2\Delta).
\]

节点距离与实际配对距离相差不超过2Delta；所有相关邻居单元总面积不超过pi(R+Delta)^2。用(5)逐项积分，纯相目标的求积差不超过

\[
b\Gamma_+ V L_cL_\phi C_B(t)t,\qquad
C_B(t)=2\pi L_\beta(2+4t)^2(2+3t)^2,
\tag{6}
\]

其中V=|Omega|。这保留了物质平均状态，没有用c(x_i)替换u_i。

对i在I，其单元并集U_I位于T_(w+Delta)，故面积S<=C_Sigma(2h+4Delta+epsilon)。点形式使用q_iq_j<=R_rho^2 V_iV_j、点支撑2h、邻居单元面积至多pi(2h+Delta)^2；理想形式使用实际点支撑内面积4pi h^2。两者状态差均由(2h+2Delta)L_c控制，系数只用0<K_hat<=Gamma_+。分别取绝对值后，界面目标差不超过

\[
b\Gamma_+ L_cL_\phi C_I(t,R_\rho)S,
\quad
C_I(t,R_\rho)=\pi M_0(2+2t)^2\{R_\rho^2(2+t)^2+4\}.
\tag{7}
\]

由(6)、(7)得到本轮点对求积连接：

\[
\boxed{|E_{pt}-E_*|\le b\Gamma_+L_cL_\phi
\left[V C_B(t)t+
C_I(t,R_\rho)C_\Sigma(2h+4\Delta+\epsilon)\right].}
\tag{8}
\]

无需假定一般路径系数在每条采样边上光滑。界面附近直接保留有界误差及其区域面积，纯相内才使用核的Lipschitz界。这是保守充分界，常数较大，不是原有限算例的误差预测或验收阈值。

## 5. 合并已封存预算：能证明什么

引用[既有投影预算](MASS_AVERAGE_PROJECTION_WEAK_BUDGET.md)，其仅需系数正对称有上界，给出

\[
|E_*-W_h^{\hat a}(c,\phi)|
\le b\Gamma_+VL_cL_\phi(32t/3+18t^2).
\]

再引用[重构弱扰动界](DENSITY_RECONSTRUCTION_WEAK_BOUND.md)及其平面接触连续预算，有符号分解仍是

\[
E_{pt}-W_{PDE}=(E_{pt}-E_*)+(E_*-W_h^{\hat a})
+(W_h^{\hat a}-W_h^a)+(W_h^a-W_{PDE}).
\tag{9}
\]

在各报告共同的平面接触、传输/正则性假设下，(8)和后三项预算一起说明：**h趋零且Delta/h趋零是该条件性点图固定测试弱一致性的充分条件**。由真实单元覆盖及代表点条件可取实际填充距离epsilon<=Delta。用固定参考长度L0无量纲化后，总保守界为O(h/L0+Delta/h)；简记无量纲h时即O(h+Delta/h)。对纯相无界面光滑问题，可引用纯相连续预算，得到相应O((h/L0)^2+Delta/h)充分界。

该结论没有证明条件必要，没有要求以扫描寻找某个h，也不支持单个有限h算例通过。它不是生成元强一致性、Mosco/演化收敛、有限时间掺混或fluid ORC验收。已有局部二阶界面和固定比例投影反例均保留。

**这个弱预算不能证明路径系数优于端点谐和系数。** 求积证明只用了纯相正确、正对称有界和局部支撑；其他满足这些性质的系数也可得到同类弱结论。不能凭该定理替换现有后端或改写保存的误差数字。普通中心阻力物质FV基线保留。

## 6. 解析障碍：稠密点云不保证物质平均分辨率

取单位周期方形、rho=b=1，静止背景。对每个正整数N，设N^2个固定水平物质条带

\[
K_{i,j}=[0,1)\times[(iN+j)/N^2,(iN+j+1)/N^2),
\quad 0\le i,j<N.
\]

每个条带面积与质量均为1/N^2，选其内部代表点

\[
x_{i,j}=\left((j+1/2)/N,\ (iN+j+1/2)/N^2\right).
\]

对任意位置，先选最近的横坐标j；固定j后纵坐标是间距1/N的周期点列，因此填充距离epsilon<=sqrt(2)/(2N)，而每个物质条带的周期直径至少1/2。q_i=V_i完全正确，最近点密度重构也精确为1。

令c=phi=1/2+(1/5)cos(2pi x)。其物理浓度在[3/10,7/10]，每个条带的**真质量平均均为1/2**，故任何仅作用这些平均的图都有E_pt=E_*=0。但是

\[
W_{PDE}(c,c)=D\int_\Omega|\nabla c|^2dx
=\frac{2}{25}\pi^2D>0.
\]

即使在充分大N时选符号尺度h=N^-1/2，也有epsilon/h趋零，而这个投影缺口不消失。这里没有改变平均定义、质量或初场来配合算法；该例只说明细长物质单元没有解析横向浓度变化。它不是已经保存算例的数值失败，也不证明所有质量平均观测不正确；在这些条带上，真实平均本来就无法分辨该横向模态。

因此，点云填充距离、正密度及精确总权重不能替代(1)中的物质局部性和Delta控制。真实几何只在这个解析证明/参照中出现，不加入候选运行接口；一般有限输入几何不可辨识性结论仍保留。

## 7. 代码、审计与下一步

新增标准库 `point_quadrature_certificates.py`，3组精确代数核验分别检查新核导数及最大值常数、求积界形状多项式、条带例的精确平均/能量系数。积分、区域估计和几何证明在本文，不把程序称为完整自动定理证明器。独立进程结果一致；无新增数值/流动测试，历史记录仍363项，旧v0.8.0/246项未恢复。

782个原文件、15个附件和55份先前正式科学清单不变，4份历史诊断清单既有差异另核。原质量、身份、c0、核、图、参考及所有封存报告保持，仅更新可变进度入口。

下一步可以进入一个有限编码任务：实现**周期最近点密度的沿段逆密度积分原语**，只接受点位置/身份、点密度、周期长度和节点对。采用线段上周期点距离平方的仿射下包络，保留区间来源，不使用真实物质面、质量、浓度或参考PDE。用解析分段阻力、互易性、周期平移、身份排列及非法输入作必要单元检查。它只是让已经声明的密度闭合可执行，不生成电导或新图，不构造传播子，不据此接纳后端。

```bash
PYTHONDONTWRITEBYTECODE=1 /workspace/.venvs/ltn-reference/bin/python research/reconnect/point_quadrature_certificates.py --output /tmp/ltn-point-quadrature
python research/reconnect/point_quadrature_certificates.py --certificates-only
```

第一条需要完整历史工作区且输出不存在；第二条只需当前标准库脚本。结果位于 `runs/reconnect-point-quadrature/`。未生成PDF/版本、提交、推送、更改环境配置或发布。

[本轮理论与核验包](../../exports/ltn-point-quadrature-review.zip)；[交付范围与哈希](../../exports/ltn-point-quadrature-delivery.json)。
