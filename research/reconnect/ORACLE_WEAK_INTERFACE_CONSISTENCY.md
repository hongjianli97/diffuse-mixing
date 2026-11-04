# 同一oracle沿段阻力核的弱界面一致性

2026-11-01（UTC）。Stage 3 **incomplete**；fluid ORC **关闭**。

预定弱形式审查完成。在以下明确的分相光滑性和传输条件下，同一核对固定光滑测试函数具有弱一致性：**纯相贡献为O(h²)，界面贡献一般为O(h)**。进一步得到显式的一阶界面项；对非零密度反差，它可由纯横向变化激发，因此一般不能把全局弱误差提升为二阶。

这与上一轮的局部二阶迹/相邻物质平均缺口相容。本轮证明的是完整密度oracle连续算子的冻结弱形式，尚未证明有限点图、质量平均投影或瞬态的收敛，没有新增数值运行或改进已保存的误差数字。

## 1. 固定范围、质量与正则性

执行已封存 `runs/reconnect-oracle-interface/stage3-status.json` 的 `next_weak_interface_review_predeclared`，先核验999项科学清单。原核、K_path、候选图、质量m、质量平均c0、物质身份、PDE及保存总作用保持。完整解析rho仍为额外oracle信息。

先在二维平面、单个平面接触Sigma={a=0}证明。两相rho分别为正常数，Gamma_p=rho_p D，0<Gamma_min≤Gamma_p≤Gamma_max，D固定。原例rhoL=1、rhoR=4、D=1/50。取0<h≤h0，核支撑半径2h。

设c在界面连续，分相C³并将导数连续延拓到各自的界面；phi为固定的全局C³紧支撑测试函数。选固定矩形Omega覆盖supp(phi)的2h0邻域及相关界面条带，面积A、切向长度ell；所有需要的线段均在略大的固定矩形内。记C_j、P_j为其中c各相、phi的j阶导数多线性算子范数上界，j=1,2,3。这些界及Omega均不随h变化。

连续c与有界分相梯度保证c在该区域全局Lipschitz：直线段至多穿过接触一次，分段积分梯度后仍有|c(y)−c(x)|≤C1|y−x|。切向迹导数也一致。

另外要求物理传输条件

\[
[c]_\Sigma=0,\qquad [\Gamma\partial_a c]_\Sigma=0.
\]

第二式使局部PDE的变化率不含界面Dirac源。若只满足第一式，下面的双线性弱极限仍可定义，但局部散度包含[Gamma c_a]delta_Sigma，不能把它误写成只有分相体积项的D Delta c。浓度初始阶跃、固壁及一般运动界面不在本定理范围内。

原周期问题的两条平面接触可分开求和；h<1/8时条带不重叠，用接触总长度替换ell即可，周期光滑测试代替紧支撑测试。本轮没有重新求解该周期问题。

## 2. 对称化：符号、质量测度与因子

保持原算子

\[
L_hc(x)=\frac2{\rho(x)}\int K(x,x+r)B_h(r)\delta_r c(x)\,dr,
\quad \delta_r c(x)=c(x+r)-c(x),
\]

K为原沿段倒阻力平均的倒数，满足Gamma_min≤K≤Gamma_max、K(x,y)=K(y,x)。B_h为原径向核；没有归一化行、核矩或状态。

由交换x、y并取两式平均，得到精确恒等式

\[
\boxed{W_h(c,\phi):=-\int\rho\phi L_hc\,dx
=\iint K(x,x+r)B_h(r)\delta_r c(x)\delta_r\phi(x)\,dr\,dx.}
\]

右侧积分包括两个方向的配对，**不再带1/2**；原生成元中的因子2已抵消对称化的1/2。rho只在左侧质量内积出现，右侧不能再乘一份rho。W_h对称，W_h(c,c)≥0；在可积或周期情形，常数测试给出组分质量守恒。若确有相应演化，则d(∫rho c²/2)/dt=−W_h(c,c)；这是条件能量恒等式，不是本轮对演化解存在或收敛的证明。

局部形式为

\[
W(c,\phi)=\int\Gamma\nabla c\cdot\nabla\phi\,dx
=-\int\rho\phi Lc\,dx,
\quad Lc=\rho^{-1}\nabla\cdot(\Gamma\nabla c).
\]

最后一个等号用到了无外边界项与上述传输条件。

## 3. 原核矩与纯相Taylor余项

原B_h=h⁻⁴ beta(|r|/h)，beta(q)=(9/(4pi))(7/3)(2+5q)(1−q/2)^5，0≤q<2，支撑外为零。由其原多项式精确积分：

| 原始矩 | 值 |
|---|---:|
| integral B_h r_i r_j | delta_ij |
| integral B_h |r|² | 2 |
| integral B_h |r|⁴ | (56/33)h² |
| integral B_h |r|⁵ | (256/143)h³ |
| integral B_h |r|⁶ | (288/143)h⁴ |
| integral_(s>0) B_h s z² | 32h/(55pi) |

以下将中心分为纯相区域|a|>2h及条带|a|≤2h。纯相中心的全部支撑均在本相，故K=Gamma_p。沿线段展开

delta c = u1+u2+Ru，u1=grad c·r，u2=H_c[r,r]/2，|Ru|≤C3|r|³/6；phi同理。

u1*v1积分精确给出grad c·grad phi；u1*v2及u2*v1为三次奇项，在完整径向支撑内积分为零。余下项按原矩逐项估计，得到纯相总误差不超过

\[
\mathcal B(h)=\Gamma_{max}A\left\{
h^2\left[\frac{14}{33}C_2P_2+\frac{28}{99}(C_1P_3+C_3P_1)\right]
+h^3\frac{64}{429}(C_2P_3+C_3P_2)
+h^4\frac8{143}C_3P_3\right\}.
\]

这些常数分别来自二次×二次、一次×三阶余项、二次×三阶余项、余项×余项；没有用拟合或新求积求得。因此B(h)=O(h²)，条件是固定C³界，而非任意随h振荡的场。

## 4. 界面条带的一般界

条带面积至多4h ell。由全局Lipschitz界，非局部双线性密度绝对值至多2 Gamma_max C1 P1，局部密度至多Gamma_max C1 P1。因此

\[
\boxed{|W_h(c,\phi)-W(c,\phi)|
\le12\Gamma_{max}C_1P_1\ell h+\mathcal B(h).}
\]

这给出**一般O(h)冻结弱一致性界**，不是从两次求积差推测出的经验阶，也不是一个物理验收阈值。该粗界只需c连续，传输条件用于将极限解释为无Dirac的原局部PDE。

## 5. 一阶界面项及其明确符号

还可利用传输条件分清一阶项来自哪里。在每个界面切向位置b，令

g(b)=partial_b c(0,b)，J(b)=Gamma_p partial_a c(0^p,b)，p_a=partial_a phi(0,b)，p_b=partial_b phi(0,b)。

在条带内用保持界面传输的分相一次jet

\[
c^{jet}_p(a',b')=c(0,b)+g(b)(b'-b)+J(b)a'/\Gamma_p
\]

及phi的全局一次jet。对给定目标b，将g、J、p视为该jet的常数。法向项满足原路径恒等式K delta c_J=J s，所有混合s*z项因z的奇性消失。因此非局部减局部的jet密度恰为

\[
g(b)p_b(b)[M_t(a)-\Gamma(a)],\qquad
M_t(a)=\int K(a,s)B_h(s,z)z^2\,ds\,dz.
\]

这一步只说明**弱界面主项在这些传输/测试条件下的结构**，不将上一轮有序方向范数当作因果份额。

对固定s>0，跨界中心a在(−s,0)；置alpha=−a/s，其K平均为

\[
\overline H=\int_0^1\left(\alpha/\Gamma_L+(1-\alpha)/\Gamma_R\right)^{-1}d\alpha
=\frac{\Gamma_L\Gamma_R}{\Gamma_R-\Gamma_L}\log\frac{\Gamma_R}{\Gamma_L}.
\]

负s对应的跨界中心平均相同。两侧合起来，使用第3节原半空间三阶混合矩，得到精确条带积分

\[
\int_{-2h}^{2h}[M_t(a)-\Gamma(a)]\,da=h\kappa_\tau,
\quad
\boxed{\kappa_\tau=\frac{32}{55\pi}(2\overline H-\Gamma_L-\Gamma_R).}
\]

当GammaL=GammaR时用连续极限，kappa_tau=0；有密度反差时，逐alpha的加权调和平均严格小于加权算术平均，积分后Hbar<(GammaL+GammaR)/2，故 **kappa_tau<0**。

在原1:4、D=1/50背景中，

\[
\kappa_\tau=\frac1\pi\left(\frac{128}{4125}\log4-\frac{16}{275}\right),
\qquad -\frac{208}{37125}<\kappa_\tau<-\frac{392}{81675}<0.
\]

严格区间只用4/3<log4<25/18以及3<pi<22/7；前者由log4=4 sum_(k≥0)(1/3)^(2k+1)/(2k+1)及正几何尾界得到，不含数值求积。该区间用于证实符号，不是已保存误差或耗散的预测。

为控制jet余项，条带目标及其邻居距(0,b)至多sqrt(20)h<5h，所以两种场减去各自jet的梯度界分别为5C2h、5P2h。非局部余量密度至多10 Gamma_max h(C2P1+C1P2)，局部余量至多2 Gamma_max h(C2P1+C1P2)。再乘条带宽度，有完整展开

\[
\boxed{W_h-W=h\kappa_\tau\int_\Sigma
\partial_b c\,\partial_b\phi\,d\sigma+R_h,}
\]

\[
\boxed{|R_h|\le\mathcal B(h)
+48\Gamma_{max}\ell(C_2P_1+C_1P_2)h^2.}
\]

这给出带常数的O(h²)余项。它仍要求本节的平面、常相密度、连续浓度、法向通量连续及固定光滑测试。

## 6. 一阶通常不可消去，局部与弱结论保持区分

取chi(a)为紧支撑光滑函数且在界面附近恒为1；取非恒定f(b)为紧支撑光滑函数，令c=phi=chi(a)f(b)。界面法向通量为零，所有传输与正则性条件均满足，而

\[
\int_\Sigma \partial_b c\,\partial_b\phi\,d\sigma=\int(f')^2db>0.
\]

故密度反差下的一阶系数严格为负，W_h-W=h*kappa_tau*integral(f')²+O(h²)，一般O(h)界确实可达到。若要将c限制为物理浓度，可用c_*+epsilon*chi*f并取足够小epsilon，测试仍取chi*f；同一非零主项结论保持。这是解析见证，没有生成一份新的流动数据。

| 判断对象 | 本轮可支持的结论 |
|---|---|
| 同一连续oracle、固定光滑测试 | 弱误差趋零；一般首项O(h)，纯相O(h²) |
| 界面一侧变化率迹 | 上轮2.5/0.625缺口保留 |
| Delta/h固定的相邻物质单元 | 上轮局部平均偏差保留；单元指标函数不属于固定光滑测试类 |
| 已保存h=0.04余弦态误差 | 仍按原NPZ解释，不据新渐近式重写数值或宣称新改善 |
| 一般质量掺混耗散 | 此处测试phi须全局C³；不能任意令phi=c并忽略c的梯度跳跃 |
| 有限点图与实际标量传播 | 求积、质量平均投影、密度闭合和瞬态收敛仍需独立论证 |

随着观察单元缩小，用光滑函数逼近其指标会使P_j增长，本界不再给出统一的小误差。因此弱一致性不能覆盖此前缩小物质单元的局部反例，也没有否定它。守恒、非负能量和固定光滑测试一致性尚不足以自动得到演化收敛；还需适当的紧性、稳定性、解/形式收敛和离散误差控制。

## 7. 实现、精确证书与完整性

新增标准库 `oracle_weak_certificates.py` 核对8组代数证书：原核高阶矩、质量内积符号及配对因子、传输平衡、纯相Taylor系数、条带/jet常数、路径平均及严格符号、显式切向界面系数、非零主项见证的代数前提。积分交换、光滑性与渐近证明在本文中，程序不被称为完整自动定理证明器。

独立进程的标准库执行结果与带来源核验的证书完全一致。没有新增流动/数值单元测试；重建检查累计仍为 **352项**，8组代数证书单列。旧v0.8.0及246项测试未恢复。

782个原始文件、15份历史附件及47份先前正式科学清单保持字节一致；4份历史诊断清单的既有路径差异单独保留，不计作当前目标全部匹配。原README/METHOD/CODEX_TASK、后端、状态、旧结果不改，仅更新可变索引 `docs/RESEARCH_STATUS.md`。

```bash
PYTHONDONTWRITEBYTECODE=1 /workspace/.venvs/ltn-reference/bin/python research/reconnect/oracle_weak_certificates.py --output /workspace/oracle-weak-reproduction
python research/reconnect/oracle_weak_certificates.py --certificates-only
```

第一条命令的输出目录必须尚不存在；第二条仅需标准Python，不依赖历史流体数组。数据在 `runs/reconnect-oracle-weak/`，包含预定方案、精确证书、独立执行输出和审计记录。本轮没有新图、算子数值求积或瞬态，没有拟合F/C、裁剪、重归一化、阈值更改、PDF、版本、提交/推送、环境配置修改或发布。

## 8. 下一步：质量平均投影的弱误差预算

连续oracle的弱一致性已经有条件成立，下一步只审查它如何传到**固定物质质量平均未知量**：以同一核的理想单元对积分共享交换作为额外几何oracle参照，推导质量内积/双线性恒等式及质量平均投影误差界，明确单元尺度Delta与h的关系。

此参照只作理论账本，不实现新边权或图，不把真实物质几何加入候选接口。将质量平均投影、有限点求积及节点rho无法唯一识别内部阻力的问题分开；不把cbar直接称为轨迹点c。该投影预算尚未完成，也不构成FV替换或候选接纳。

普通中心阻力物质FV基线保持。Stage 3 **incomplete**；fluid ORC **继续关闭**。

- [报告、代码与精确证书审阅包](../../exports/ltn-oracle-weak-review.zip)。
- [完整增量包及十一份历史依赖](../../exports/ltn-oracle-weak-downloads.md)。
- [交付哈希](../../exports/ltn-oracle-weak-delivery.json)。
