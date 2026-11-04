# 同一oracle沿段阻力核的二阶界面一致性审查

2026-11-01（UTC）。Stage 3 **incomplete**；fluid ORC **关闭**。

预定理论审查完成。**同一核的恒法向通量保真，不保证二阶曲率一致性。** 对法向、横向的单位共同曲率，1:4密度接触的一侧变化率与PDE正确值之比均为 **5/2、5/8**。纯横向二次场还给出直接的物质平均反例：界面相邻、分相对齐的低密度单元偏大，高密度单元偏小；在固定有限Delta/h下，该局部平均偏差不随h缩小而消失。

这些结论针对当前“标量沿段阻力×径向核”的完整密度oracle。它们不等于保存余弦态5.42%/3.99%误差的定量归因，也不否定固定单元平均、弱形式或全域质量加权范数随h趋零收敛。没有据此更换系数或宣布所有正交换图都不可能。

## 1. 原问题和理论审查的边界

执行已封存 `runs/reconnect-oracle-directional/stage3-status.json` 中的 `next_second_order_interface_review_predeclared`，先核验其981项科学清单。候选图、m、c0、物质身份、原PDE和两档oracle总作用均不变。完整解析rho及界面位置仍只属于额外oracle信息。

状态定义继续是固定物质单元质量平均：

\[
m_i=\int_{K_i}\rho\,dV,\qquad
\bar c_i=\frac{1}{m_i}\int_{K_i}\rho c\,dV.
\]

以下连续二次函数只用于解析检验算子，再观察到物质单元；没有生成或替换一份数值初值数据，没有将点态作为图状态。

将同一平面接触平移到a=0，rhoL=1、rhoR=4，D=1/50，Gamma_p=rho_p D。原周期接触相距1/2，h=1/25、支撑半径2h，局部支撑不会遇到另一接触。取界面附近与下面多项式相同的连续场；局部纯横向二次场可在切向作光滑周期延拓，并取足够小振幅维持物理浓度范围。结论只依赖覆盖支撑与观察单元的该邻域。

## 2. 保持原核和全部常数

原算子为

\[
\mathcal L_hc(a,b)=\frac{2}{\rho(a)}\int B_h(s,z)K(a,s)
 [c(a+s,b+z)-c(a,b)]\,ds\,dz,
\]

\[
K(a,s)=\left[\int_0^1\frac{d\theta}{\Gamma(a+\theta s)}\right]^{-1}.
\]

原径向核满足 B_h(r)=h^{-4} beta(|r|/h)，

\[
\beta(q)=\frac{9}{4\pi}\frac73(2+5q)(1-q/2)^5\quad(0\le q<2),
\]

支撑外为零。由原W核多项式直接积分，未重归一化：

\[
\int B_hs^2=\int B_hz^2=1,\qquad
\int_{s>0}B_hs^2=\int_{s>0}B_hz^2=\frac12,
\quad \int B_hsz=0.
\]

常数系数纯相中的二次函数因而精确给出D(c_aa+c_bb)。系数K正、对交换端点对称；这些性质仍不足以单独给出界面曲率一致性。

## 3. 恒通量与二次曲率分别推导

在两相内取局部场

\[
c_p(a,b)=c_*+\frac{J}{\Gamma_p}a+\frac12 Q_p a^2+\frac12 T b^2.
\]

浓度在界面连续，Gamma c_a的两侧迹均为J，故不存在由通量跳跃产生的Dirac源。取共同Q_L=Q_R=Q时，PDE变化率的两侧迹也同为D(Q+T)，避免用变化率自身的跳跃构造反例。

线性阻力项满足

\[
K(a,s)[c_J(a+s)-c_J(a)]=Js,
\]

其完整对称支撑积分为零，对所有目标a均精确保留恒通量。下面只需分析二次项。

对固定h以及几乎处处的非零s，当目标a从任意一侧趋于0，路径中目标相所占比例趋零，故

\[
K(a,s)\longrightarrow
\begin{cases}\Gamma_L,&s<0,\\\Gamma_R,&s>0.\end{cases}
\]

s=0的纯切向路径是积分零测集，不影响结果。K夹在两相Gamma之间；固定h下局部被积函数有统一可积界，因此可交换该一侧极限与积分。在b=0处，二次差趋于(Q_side s²+T z²)/2。结合前节原半空间矩，密度加权作用的共同迹为

\[
\boxed{S_*:=\lim_{a\to0^-}\rho_L\mathcal L_hc
=\lim_{a\to0^+}\rho_R\mathcal L_hc
=\frac{\Gamma_LQ_L+\Gamma_RQ_R+(\Gamma_L+\Gamma_R)T}{2}.}
\]

对共同曲率Q，S_*=Gamma_avg(Q+T)，Gamma_avg=(GammaL+GammaR)/2。真正变化率分别还需除以rhoL、rhoR。对单位法向曲率或单位横向曲率，两种情形的表相同：

| 一侧量，单位Q或T | 低密度侧 | 高密度侧 |
|---|---:|---:|
| rho | 1 | 4 |
| 正确PDE变化率 | 1/50 | 1/50 |
| 同一oracle核变化率迹 | 1/20 | 1/80 |
| oracle/PDE | 5/2 | 5/8 |
| 变化率偏差 | 3/100 | −3/400 |

这些值与h无关，因为原二阶矩与h无关。若要求共同二次曲率的两侧迹均正确，就需同时Gamma_avg=GammaL=GammaR；当前密度反差不满足。结论针对本核形式，不能推广到所有位置/方向相关的正对称交换。

## 4. 纯横向场的物质平均严格反例

取J=Q=0、T>0，即c=c_*+Tb²/2。在支撑内该场与a无关，正确PDE作用处处为DT，与密度接触无关。切向奇项积分为零，当前oracle核却给出

\[
\mathcal L_hc(a,b)=\frac{T}{\rho_p}\int K(a,s)B_h(s,z)z^2\,ds\,dz.
\]

低密度侧 −2h<a<0：同相位移K=GammaL，跨界位移K>GammaL。跨界支撑有正测度，B_h z²在其内部几乎处处为正，故L_h c>DT。高密度侧0<a<2h同理有K<GammaR，故L_h c<DT。支撑不跨界时则精确等于DT。

于是任何紧邻接触、分别位于各相内的固定物质单元，其平均变化率

\[
\bar r_K=\frac{1}{m_K}\int_K\rho\mathcal L_hc\,dV
\]

都分别严格高于或低于DT。这里使用原质量定义，未用目标点值或重归一化代替物质观察。它直接说明有限h下的局部平均一致性缺口。

进一步令a=h xi、s=h u、z=h v，得到无量纲函数

\[
\eta_p(\xi)=\frac{1}{\Gamma_p}\int K(\xi,u)\beta(\sqrt{u^2+v^2})v^2\,du\,dv,
\qquad \mathcal L_hc/(DT)=\eta_p(a/h).
\]

对于法向宽度Delta=lambda h的相邻单元，由单元内rho常量，

\[
\frac{\bar r_L}{DT}=\frac1\lambda\int_{-\lambda}^0\eta_L(\xi)d\xi>1,\qquad
\frac{\bar r_R}{DT}=\frac1\lambda\int_0^\lambda\eta_R(\xi)d\xi<1.
\]

对任意固定有限lambda>0，两比值与h无关，故该界面相邻单元的局部平均偏差可在联合缩小h、Delta时持续。原N128/h0.04对应 **lambda=25/128**，不是另选参数。

为给出不用数值求积的严格证书，对所有|xi|≤25/128，取跨界位移矩形 1/2≤|u|≤1、1/2≤|v|≤1，u符号指向另一相。目标相路径比例alpha≤25/64，因此

\[
K/\Gamma_L\ge256/139>1\quad(\text{低密度侧}),\qquad
K/\Gamma_R\le64/181<1\quad(\text{高密度侧}).
\]

矩形上q≤sqrt(2)<3/2。用pi<22/7、2+5q≥2、1−q/2>1/4及v²≥1/4，积分该正矩形得到

\[
\int_{\rm rectangle}\beta(q)v^2\,du\,dv>\frac{147}{360448}.
\]

因此，逐点不等式再作同一正质量平均，有

\[
\boxed{\frac{\bar r_L}{DT}>1+\frac{17199}{50102272},\qquad
\frac{\bar r_R}{DT}<1-\frac{17199}{65241088}.}
\]

这是保守解析分离界，不是实际平均偏差的近似值，更不是原保存余弦态误差的预测。没有求积扫描、拟合或新候选计算。

## 5. 极限顺序与弱/全域结论必须分开

固定非零a再令h趋零，支撑最终完全在纯相内，共同二次场精确得到D(Q+T)，之后取界面一侧极限仍正确。反过来，先取界面迹再令h趋零，仍保留第3节的5/2、5/8比值。**这两种迭代极限不交换。**

同一结论也不能直接当作固定物质单元平均的h趋零失败。以下只考虑无限平面单接触上的分相二次解析模型，或支撑完全落在局部二次片内的观察区域；不把结论套到周期延拓的其他区域或原余弦场。对共同二次曲率，恒通量项精确抵消，变化率误差只支撑在|a|<2h。由原核绝对一阶矩integral B_h|s|=16/(3pi h)、|a|≤2h和二阶矩，可得独立于h的保守界

\[
|\mathcal L_hc-\mathcal Lc|
\le D\left(\frac{301}{9}|Q|+5|T|\right)=:M,
\]

这里用了pi>3及Gamma_max/rho_min=4D。对切向长度为ell的固定观察窗，

\[
\|E_h\|_{L^1(\rho)}\le4h\rho_{max}\ell M=O(h),\qquad
\|E_h\|_{L^2(\rho)}\le M\sqrt{4h\rho_{max}\ell}=O(\sqrt h).
\]

固定单元K若质量m_K>0不变，其平均变化率误差至多rho_max*4h*ell*M/m_K，故是O(h/Delta)并趋零；对固定光滑紧支撑测试函数的弱作用差也为O(h)。这些是上述局部二次场/单接触模型的界，不是一般传输解的完整收敛定理。

固定lambda时，界面相邻缩小单元的平均偏差持续，**仍可与全域质量加权L2误差趋零同时成立**：有偏单元总体质量随h缩小。不能把局部偏差持续写成全域不收敛，更不能直接推断有限时间掺混错误。这也是下一步必须审查弱形式的原因。

## 6. 对当前路线的决定

本轮阻止的是“仅凭恒通量保真，将当前完整密度oracle核提升为通用点输入后端”的推断。当前核在有限h下既有二阶点态迹缺口，也有纯横向场的相邻物质平均缺口，故不进入新的候选瞬态或Stage 3验收。原普通中心阻力物质FV基线保留。

原余弦初场在法向的两相曲率不同，并且方向预算含预定顺序及交叉项；不能把这里共同曲率的2.5/0.625比值当作其5.4237%/3.9850%残差的定量解释。上一轮3.1229%总误差及其抵消原样保留，本轮没有新的数值误差下降。

完整rho场仍是额外输入；一般有限点位置/质量/节点rho/cbar不能唯一确定内部沿段阻力的既有反例仍成立。本轮没有补出该闭合，也没有调整B_h、路径系数、F/C、单元平均、阈值或误差定义。

## 7. 精确证书、复现和完整性

新增 `oracle_interface_certificates.py` 只用标准库Fraction核对7组精确代数：原核多项式及半空间二阶矩、二次迹系数、两侧变化率比、不同迭代极限常数、原Delta/h及路径界、相邻质量平均严格分离界、二次层统一界的常数。极限交换、积分正性、缩放与平均证明在本文中；程序不是完整自动定理证明器。

没有新增数值单元测试或流动测试，重建检查累计仍为 **352项**；7组代数证书另列，不增加或重跑旧测试计数。旧v0.8.0/246项未恢复。

782个原始文件、15份历史附件和46份先前正式科学清单保持字节一致；4份历史诊断清单已有路径差异另核，不宣称当前目标全部匹配。原README/METHOD/CODEX_TASK、旧后端与结果均不改，仅更新可变入口 `docs/RESEARCH_STATUS.md`。

```bash
PYTHONDONTWRITEBYTECODE=1 /workspace/.venvs/ltn-reference/bin/python research/reconnect/oracle_interface_certificates.py --output /workspace/oracle-interface-reproduction
python research/reconnect/oracle_interface_certificates.py --certificates-only
```

第一条命令核验历史来源，输出目录必须尚不存在；第二条仅复算代数，标准Python即可，不需要历史流体数据。`runs/reconnect-oracle-interface/` 保存预定方案、精确证书、独立标准库执行输出、执行记录和完整性审计。

## 8. 下一步只审查同一核的一般弱形式

预定下一步：对平面接触两相正密度、界面连续且分相光滑的场，写出当前同一核的对称双线性弱形式，推导其与局部PDE弱形式的差异界；分开支撑远离界面的Taylor误差与宽O(h)接触层，说明所需正则性及Gamma界。特别区分局部质量平均一致性、弱一致性和瞬态收敛，后者不能由一个冻结弱形式结论自动获得。

该一般弱形式审查尚未完成。它不引入新核/方向系数或密度重构，不扫描参数、不建图、不作瞬态，不修改普通FV。原图、保存总量与误差定义继续固定。

本轮无PDF、新软件版本、提交/推送、环境配置修改或发布。Stage 3 **incomplete**，fluid ORC **继续关闭**。

- [报告、精确证书与代码审阅包](../../exports/ltn-oracle-interface-review.zip)。
- [本轮完整增量包和十份历史依赖](../../exports/ltn-oracle-interface-downloads.md)。
- [交付哈希](../../exports/ltn-oracle-interface-delivery.json)。
