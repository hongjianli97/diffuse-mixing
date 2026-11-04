# 保留原固定质量的容量缺陷：固定图的一致时间预算

2026-11-01（UTC）。承接[Laguerre可行性审查](LAGUERRE_MASS_INTERIORITY_REVIEW.md)。本轮不再强制辅助分区与原质量精确匹配；保留原m、原rho和原点，用辅助密度积分得到的m_tilde作**比较量**，不替换真实物质质量。

主要结果是固定共享图的传播界：若r_i=m_tilde_i/m_i>0，eta=max|r_i-1|，R=max(r_i)/min(r_i)，则对所有t>=0、任意初态u0，

\[
\boxed{\|(S_M(t)-S_{\widetilde M}(t))u_0\|_M
\le\eta(2+R)\|u_0-\mu_M(u_0)1\|_M.}\tag{1}
\]

它不依赖图的最大特征值，不要求u0具有网格一致的光滑性，也不假设两生成元可交换。**它只比较同一固定L下两个容量矩阵，绝不是完整PDE误差或运动轨迹验收。** 后续将使用预先指定的一个保存快照核验输入量，不用其结果选择参数。

## 1. 两个容量矩阵保持各自定义

由原允许点输入构造无权周期Voronoi辅助单元C_i，既定最近点密度rho_hat在C_i中为给定rho_i。定义m_tilde_i=b integral_Ci rho_hat=b*rho_i*|C_i|>0。实际状态始终是原真实物质单元K_i的质量平均，实际质量是m_i。

令M=diag(m)，M_tilde=diag(m_tilde)，同一共享非负电导定义半正定对称L。符号约定为

\[
A=-M^{-1}L,\quad B=-\widetilde M^{-1}L,\quad
S_M(t)=e^{tA},\quad S_{\widetilde M}(t)=e^{tB}.
\]

这里只在理论上比较，未构图或运行任一传播。两个容量分别定义不同模型；不能因二者均守恒就混称同一PDE。

令r_i=m_tilde_i/m_i、E=diag(r_i-1)。逐行恒等式为

\[
\boxed{A-B=EB.}\tag{2}
\]

没有把m或rho乘上补偿因子，也未按c0/PDE响应拟合E。总质量相同仅给sum m_i(r_i-1)=0，不给eta=0。

## 2. 范数换算与直接Duhamel预算

设r_-=min r_i>0、r_+=max r_i，C_r=sqrt(r_+/r_-)，则

\[
\sqrt{r_-}\|z\|_M\le\|z\|_{\widetilde M}
\le\sqrt{r_+}\|z\|_M,\qquad \|E\|_{M\to M}=\eta.
\]

A在M内积自伴非正，B在M_tilde内积自伴非正，故

\[
\|S_M(t)\|_{M\to M}\le1,\quad
\|S_{\widetilde M}(t)\|_{M\to M}\le C_r.
\]

B的半群不应在原M范数中直接当作收缩。固定有限图上的Duhamel公式为

\[
D(t):=S_M(t)-S_{\widetilde M}(t)
=\int_0^t S_M(t-s)EB S_{\widetilde M}(s)\,ds.\tag{3}
\]

于是对给定初态可保留直接预算eta integral_0^t ||B S_tilde(s)u0||_M ds。但只代入O(1/s)谱界会在s=0得到不可积估计；不能据此宣称容量缺陷无界，也不能靠补一个时间截断阈值解决。

## 3. 分半区间分部积分，消除伪奇异

有限图谱定理给出，对s>0，

\[
\|A S_M(s)\|_{M\to M}\le\frac1{es},\qquad
\|B S_{\widetilde M}(s)\|_{M\to M}\le\frac{C_r}{es},
\]

因为sup_(lambda>=0) lambda exp(-s lambda)=1/(es)。零模态的该读数为0；这是解析性质，不是修改任何数值特征值。

将(3)分为[0,t/2]与[t/2,t]。前半令F(s)=S_M(t-s)E S_tilde(s)，则

\[
F'(s)=-A S_M(t-s)E S_{\widetilde M}(s)
+S_M(t-s)EB S_{\widetilde M}(s).
\]

因此前半积分等于F(t/2)-F(0)+integral_0^(t/2) A S_M(t-s)E S_tilde(s)ds。其范数分别不超过eta*C_r、eta、eta*C_r*log(2)/e。后半直接使用B的谱界，不超过eta*C_r*log(2)/e。相加得到更细的界

\[
\|D(t)\|_{M\to M}\le
\eta\left[1+C_r\left(1+\frac{2\log2}{e}\right)\right].\tag{4}
\]

整个推导中E在A和B之间，未将它与任一生成元交换。因为log2<1、e>2，且1+2sqrt(R)<=2+R，(4)被纯代数量eta(2+R)控制。

两个半群都保常数，故D(t)1=0；对u0只需应用于u0-mu_M(u0)1，得到(1)。t=0直接成立。该中心化只用于证明/误差度量，**不调整输入初态、实际传播或任何保存库存**。

该界对每个固定有限图成立。若讨论网格族，要使容量项消失仍需eta->0且R受控；不能从点云稠密或全域质量差为零直接推得。界可能保守，不能拿界的大小冒充实测PDE误差或新通过阈值。

## 4. 辅助平均使用自己的积分质量，但实际m不改

定义

\[
(A_Kf)_i=\frac b{m_i}\int_{K_i}\rho f,\qquad
(A_Cf)_i=\frac b{\widetilde m_i}\int_{C_i}\hat\rho f.
\]

第二个分母是**比较模型自身平均的定义**，不是重新归一化原质量或把辅助状态送回实际求解器。两个概率测度nu_i^K=b*rho*1_Ki/m_i与nu_i^C=b*rho_hat*1_Ci/m_tilde_i各有总量1。因此原W1观察界仍可写成

\[
\|(A_K-A_C)f\|_M\le\operatorname{Lip}(f)
\left[\sum_i m_iW_1(\nu_i^K,\nu_i^C)^2\right]^{1/2}.
\]

其中范数和外层权重始终用实际m。A_C从L²(b*rho_hat)到原M范数的算子界为r_-^(-1/2)，不再无条件为1。

库存联系必须保留：

\[
\sum_i m_i(A_Cf)_i-b\int\hat\rho f
=\sum_i(m_i-\widetilde m_i)(A_Cf)_i.\tag{5}
\]

即使sum m=sum m_tilde，右侧对一般非恒定f也不为零，不能将辅助平均按实际质量聚合后的值称为辅助PDE精确库存。

## 5. 完整静止问题保留五项，容量项不吞掉其他误差

令S_hat为固定辅助密度PDE半群，S为真实静止PDE半群，对同一连续初场c有

\[
S_M A_Kc-A_KSc
=(S_M-S_{\widetilde M})A_Kc
+S_{\widetilde M}(A_K-A_C)c
+(S_{\widetilde M}A_C-A_CS_{hat})c
+A_C(S_{hat}-S)c
+(A_C-A_K)Sc.\tag{6}
\]

五项依次是容量差、初观察差、辅助空间离散、密度模型传播、终观察差。第一项可用(1)，第二项需额外乘C_r再用观察界；后三项没有被容量定理消除。真实材料几何仅用于明确的条件性理论，不加入运行接口。有限点的可识别性限制和原质量平均协方差项保持。

该定理固定L、M和M_tilde。随时间改变的Voronoi区域不是物质区域，其均值还涉及相对边界运动的输运；不能简单把本证明按时刻拼接成移动激波/一般轨迹的验收。时变问题仍需独立残差与有序传播控制。

## 6. 本轮限定的保存输入诊断

理论和输入协议先记录，再只读取既有压缩流中点：`runs/reconnect-kernel2d-compression-midpoint/midpoint-observations.npz`，N=128，索引1、t=0.025。仅加载ID、m、位置、给定rho及时间标签；不读取c0、候选图、参考变化率、真实材料面、J或真实面积来构造诊断。

该保存点阵若通过逐成员验证为周期张量积，则辅助Voronoi单元可由相邻点中垂线直接得到，无需数值剖分。对一个轴的相邻正周期间隔d_left、d_right，宽度=(d_left+d_right)/2、质心相对点偏移=(d_right-d_left)/4、关于原点的二阶矩=(d_left²-d_left*d_right+d_right²)/12。两个轴组成矩形。

所有几何、m_tilde、比值和容量上界用保存float64的精确Fraction计算；保留周期间隔及面积的原精确和，不调整原点或密度。二维质心偏移采用L1范数作为欧氏范数上界，标签明确为上界。得到的eta、R及eta(2+R)只属于这一输入的**同固定L容量比较**，不设新物理通过阈值。

若保存输入不具备声明结构、出现非法质量/密度/周期次序，保留失败并停止，不换快照或加容差。本轮只做这一个输入诊断，不生成共享电导、图或PDE/标量演化。实际数值与Stage 3缺口另列于[本轮输入与验收路线](CAPACITY_INPUT_AND_STAGE3.md)。

## 7. 验证范围

新增8组标准库代数核验：两个质量矩阵的生成元差、范数比例平方、非交换Duhamel分部积分、谱平滑/最终系数的代数、五项消去、库存差、矩形均值矩、全域质量抵消不消除局部容量差。谱定理和积分估计的证明在本文，不冒充自动PDE定理证明。

软件检查记录仍428，8组新代数另列，旧证明/套件不重跑。输入诊断另计一次，非新增流动算例或物理验收。原README/METHOD/CODEX_TASK、历史代码/数据/报告和附件不改，Stage 3 incomplete，fluid ORC关闭。
