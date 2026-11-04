# 同一核的质量平均投影弱误差预算

2026-11-01（UTC）。Stage 3 **incomplete**；fluid ORC **关闭**。

预定理论预算完成。理想单元对积分共享图精确对应 **P_rho L_h P_rho**，即先将连续场替换成物质质量平均的分片常数，再作用原核并取质量平均。它一般不等于真实连续场作用的平均 **P_rho L_h**；中间仍有闭合误差。

本轮给出显式投影弱误差界，Delta/h趋零是一个充分尺度条件。还证明：**固定单元宽度/核宽，即使单元对积分完全精确，理想质量平均图也可在纯相二次场上得到偏大的有效扩散率D(1+eta)，eta>0。** 因而上一轮连续oracle的弱一致性不能无条件传给该闭合图。

这是同核、额外真实几何/完整密度oracle的理论参照，没有实现新边权或图，也没有给已保存点输入候选作出新的数值判决。普通中心阻力物质FV基线保持。

## 1. 固定状态、投影与额外信息

执行 `runs/reconnect-oracle-weak/stage3-status.json` 中已封存的 `next_mass_average_projection_review_predeclared`，先核验1017项科学清单。原m、c0、物质身份、候选图、PDE和保存总作用均不改。

在原二维周期域的固定时刻，设真实物质区域组成不重叠分割{K_i}，且

\[
m_i=\int_{K_i}\rho\,dx>0,\quad
(P_\rho c)|_{K_i}=\bar c_i=\frac{1}{m_i}\int_{K_i}\rho c\,dx.
\]

设0<rho_min≤rho≤rho_max，V为域面积，Delta为**最大单元直径**。c、phi分别有全局Lipschitz常数C1、P1。传到局部PDE弱形式时，再使用上一轮的平面接触、分相C³、法向Gamma梯度连续与固定光滑测试假设。

P_rho是L²(rho dx)的正交投影，逐单元保持质量积分，保常数且L²质量范数不增：

\[
\int_{K_i}\rho(P_\rho c-c)dx=0,\qquad
\|c\|_{L^2(\rho)}^2=\|P_\rho c\|_{L^2(\rho)}^2+\|P_\rho c-c\|_{L^2(\rho)}^2.
\]

这里的正交测度是rho dx，不是dx；原核双线性式的配对积分使用dx dy，不能据正交性直接删除其交叉项。证书中的分相解析单元例子保留1:4密度接触，质量残差为零而体积残差为−3/32，说明两种测度不能混用；该例子未生成替代初场。

此外，**真实分割和沿段完整rho是本参照的额外oracle信息**，不能从现有轨迹点、m、节点rho自动获得，也不加入候选接口。

## 2. 理想共享交换的精确恒等式

记A_h(x,y)=K_path(x,y)B_h(y−x)，沿用原核和原正对称路径系数。仅在理论上定义i≠j时

\[
G^*_{ij}=2\int_{K_i}\int_{K_j}A_h(x,y)\,dy\,dx.
\]

它正对称，并满足支撑限制。对分片常数u、v，定义

\[
(Q^*u)_i=\frac1{m_i}\sum_{j\ne i}G^*_{ij}(u_j-u_i).
\]

由逐单元积分及两方向配对，精确得到

\[
\boxed{Q^*\bar c=P_\rho L_h(P_\rho c),}
\]

\[
\boxed{-\sum_i m_i\bar\phi_i(Q^*\bar c)_i
=\sum_{i<j}G^*_{ij}(\bar c_j-\bar c_i)(\bar\phi_j-\bar\phi_i)
=W_h(P_\rho c,P_\rho\phi).}
\]

这里G*定义中的2、无向配对求和及上一轮双向积分的因子完全一致。单元内配对因分片常数差为零而不贡献。图层因此守恒、常数保持，且质量内积下对称、能量非负；这些代数性质不证明其扩散率正确。

连续场的实际平均作用则为P_rho L_h c。两者差为

\[
Q^*\bar c-P_\rho L_hc=P_\rho L_h(P_\rho c-c).
\]

这正是本轮要保留的单元内部状态闭合项，没有将cbar当作轨迹点值，也没有把理想图称为无误差的原PDE有限体积离散。

## 3. 不对分片常数套光滑性：投影误差的直接界

设e=P_rho c−c、f=P_rho phi−phi。由正质量平均及单元直径，|e|≤C1 Delta、|f|≤P1 Delta，无需单元凸性或投影后连续。原核精确径向矩为

\[
\mu_0=\int B_h\,dr=\frac9{2h^2},\quad
\mu_1=\int B_h|r|\,dr=\frac8{3h},\quad
\mu_2=\int B_h|r|^2\,dr=2.
\]

对于任意L²误差，利用两端对称性还得到能量估计

\[
W_h(e,e)\le\frac{18\Gamma_{max}}{h^2}\|e\|_{L^2(dx)}^2
\le\frac{18\Gamma_{max}}{\rho_{min}h^2}\|e\|_{L^2(\rho)}^2.
\]

保留有符号恒等式

\[
W_h(P_\rho c,P_\rho\phi)-W_h(c,\phi)
=W_h(e,\phi)+W_h(c,f)+W_h(e,f).
\]

三个量都未强制为零，也不作可加的因果份额。用|delta e|≤2C1 Delta、|delta phi|≤P1|r|和原mu1，每个线性交叉项绝对值不超过(16/3)Gamma_max V C1P1 Delta/h；二次项由mu0界定。故令epsilon=Delta/h，有

\[
\boxed{|W_h(P_\rho c,P_\rho\phi)-W_h(c,\phi)|
\le\Gamma_{max}VC_1P_1\left(\frac{32}{3}\epsilon+18\epsilon^2\right).}
\]

这是真正覆盖分片常数投影的界，没有假设P_rho c具有固定C³导数，也未借质量正交性消掉dx dy配对项。

结合上一轮连续核弱界，在同样的传输与测试条件下得到

\[
|W_{G^*}(\bar c,\bar\phi)-W(c,\phi)|
\le\Gamma_{max}VC_1P_1(32\epsilon/3+18\epsilon^2)
+12\Gamma_{max}C_1P_1\ell_\Sigma h+\mathcal B(h),
\]

其中B(h)=O(h²)及接触总长度ell_Sigma沿用上一轮定义。因此h趋零且Delta/h趋零，是此理想图弱一致性的**充分条件**。若所有长度先除以固定参考长度，要让这个保守界保持O(h)总阶，一个充分但未证明必要的条件是Delta=O(h²)。该界不是实际算例误差估计，也不是建议扫描或修改核宽的验收阈值。

原N128方格的边长ell=1/128，而Delta=sqrt(2)/128；在h=0.04下，ell/h=25/128，**Delta/h=25sqrt(2)/128**。不能将上一轮法向宽度比当成单元直径比。单个已保存算例也不是已经完成的联合极限序列。

## 4. 固定比例的严格反例：精确积分仍不消除平均闭合

仅有非零上界还不能证明不收敛。下面另给解析反例，不用新的数值算例。

取均匀密度纯相内的规则方格，边长ell，Gamma=rho D。原接触算例远离界面的纯相内部也满足这些局部条件。先在无限均匀网格推导，再用固定物理邻域内相同的二次场作光滑延拓；当h、ell足够小时，测试支持内的作用只看到该二次片。

对法向位移s，写s/ell=k+theta，k=floor(s/ell)，0≤theta<1。将目标点均匀平均在一个单元内，邻居单元中心的法向位移d只有ell*k和ell*(k+1)两种，比例分别为1−theta、theta。因此

\[
\mathbb E[d]=s,\qquad
\boxed{\mathbb E[d^2]=s^2+\ell^2\theta(1-\theta).}
\]

对负s也成立，theta为通常的小数部分。它是单元对积分中的真实几何相位平均，不是随机生成一份数据，也不是节点密度拟合。

取c(a,b)=a²/2。其**正确质量平均**为cbar_i=a_i²/2+ell²/24；均匀密度下这也是体积平均。理想图的线性中心项由配对对称性消失，生成元因子2与二次差中的1/2抵消，二阶响应精确为

\[
\boxed{(Q^*\bar c)_i=D[1+\eta(\lambda)],\qquad \lambda=\ell/h,}
\]

\[
\boxed{\eta(\lambda)=\lambda^2\int_{|q|<2}\beta(|q|)
\operatorname{frac}(q_a/\lambda)[1-\operatorname{frac}(q_a/\lambda)]\,d^2q.}
\]

正确局部PDE变化率恒为D。对每个固定lambda>0，积分非负且在正测度上严格为正，故eta(lambda)>0；它只取决于lambda和原始beta，**与h无关**。因此固定宽度/核宽联合缩小时，该理想图在纯相内部仍可保留非零扩散率偏差。

这也给出固定光滑测试下的反例，而不只是一侧点态：在固定二次片内部选紧支撑phi且integral phi>0，所有与phi相交单元及其核邻居均留在片内。图弱式与正确PDE弱式之差为−rho D eta(lambda) integral phi，不随h趋零。该结论针对此分片常数理想闭合，**不是声称所有正交换图或已存点候选都失败**。

## 5. 原宽度比的无求积分离界

原lambda=25/128时，用0≤theta(1−theta)≤1/4和mu0即得eta<9lambda²/8。为给出显式严格正下界，在无量纲位移中取

lambda/4≤|q_a|≤lambda/2，1/2≤|q_b|≤1，

包含两种法向及两种横向符号。矩形总面积lambda/2；其上theta(1−theta)≥3/16，|q|<3/2。原beta及pi<22/7给出beta>147/45056，因此

\[
\boxed{\frac{6890625}{3023656976384}<\eta(25/128)
<\frac{5625}{131072}.}
\]

这是保守的严格分离区间，**没有数值计算eta，也不能将区间端点当作其近似值**。它仅证实原比值下偏差不会恰为零；不把它代入原余弦态5.42%/3.99%或3.12%的残差解释。

本轮没有除以1+eta重归一化边权，没有改核矩、质量平均或阈值来消除反例。普通中心阻力物质FV基线不受该理想核闭合反例替换。

## 6. 与有限点候选的四项有符号预算

固定相同cbar、phibar，令E_hat为原端点系数点候选的双线性式；E_point-path为同一点位置和原给定求积权重、但采用完整路径oracle系数的**理论中间量**；E_star为上述理想单元对积分图。则

\[
E_{hat}-W=(E_{hat}-E_{point-path})
+(E_{point-path}-E_{star})
+(E_{star}-W_h(c,\phi))+(W_h(c,\phi)-W).
\]

| 项 | 含义和本轮边界 |
|---|---|
| 端点系数−完整路径系数 | 需要额外内部rho信息；既有有限输入不可辨识性仍未解决 |
| 点对求积−真实单元对积分 | 包含采样、体积权重等近似；m/rho_node不被自动称为真实体积 |
| 理想图−连续核 | 本轮质量平均投影/闭合界与固定比例反例 |
| 连续核−局部PDE | 上一轮有条件弱一致性及显式O(h)界面主项 |

这是有符号恒等式，未计算任何中间新图或边权。各项可能抵消，不能按范数当作相加份额，也不能用其抵消制造验收结果。真实几何、完整路径rho都只作为理论参照，不进入当前候选必要输入。

## 7. 精确证书、复现与完整性

新增标准库 `mass_projection_certificates.py` 核对9组精确代数：原核0/1/2径向矩、rho加权投影及测度差异、共享交换配对因子、投影预算常数、宽度与直径比、一般k/theta跳跃均值/方差多项式、二次质量平均与生成元系数、原比例eta的严格界、四项带符号消去恒等式。积分与极限证明在本文中；证书程序不是完整自动定理证明器。

独立标准库执行输出与带来源核验的输出完全一致。本轮无新数值/流动测试，重建检查累计仍 **352项**，9组代数证书另列；旧v0.8.0及246项未恢复。782原始文件、15历史附件和48份先前正式科学清单保持字节一致；4份历史诊断清单的既有路径差异保留另核。原README/METHOD/CODEX_TASK、后端、状态与旧结果不改，仅更新 `docs/RESEARCH_STATUS.md`。

```bash
PYTHONDONTWRITEBYTECODE=1 /workspace/.venvs/ltn-reference/bin/python research/reconnect/mass_projection_certificates.py --output /workspace/mass-projection-reproduction
python research/reconnect/mass_projection_certificates.py --certificates-only
```

第一条输出目录须尚不存在；第二条仅需标准Python，不依赖历史流体数组。结果在 `runs/reconnect-mass-projection/`，包括预定方案、精确证书、独立执行输出和审计。本轮无新边权/图、算子数值求积、瞬态、参数扫描、F/C拟合、裁剪、重归一化、阈值修改、PDF、版本、提交/推送、环境配置修改或发布。

## 8. 下一步只量化原比例的一行理想响应

下一步限定为一个数值控制：固定原N128、h=0.04、D=0.02、lambda=25/128，以本轮解析方差公式计算eta和一行理想二次响应；只用预定的两组法向/横向求积(32,32)、(64,64)，法向积分在s为单元宽度整数倍处拆开，以检查小数部分的折点。最细原始规则预先指定，保留两规则的eta及行响应差，规则差不称严格误差界。

这项单行积分尚未运行。它不构造全图、不传播、不改原输入、不扩大扫描、不用eta修正任何边权。只有量化这个已证明的闭合偏差，才决定下一项工作；不能直接把它归因为已存点候选误差。

Stage 3 **incomplete**；fluid ORC **继续关闭**。

- [报告、代码与精确证书审阅包](../../exports/ltn-mass-projection-review.zip)。
- [完整增量包及十二份历史依赖](../../exports/ltn-mass-projection-downloads.md)。
- [交付哈希](../../exports/ltn-mass-projection-delivery.json)。
