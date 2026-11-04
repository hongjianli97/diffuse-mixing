# 平面接触通量闭合：双侧迹必要条件与有限输入可辨识性

2026-11-01（UTC）。本轮为预定的**纯理论审查**，未构造候选图或运行流体/参考瞬态。Stage 3 **incomplete**，fluid ORC **关闭**。

得到三点结论：

1. 原核若以一个仅依赖两相扩散系数的共享常数表示全部跨相作用，就无法在密度跳跃时同时保持局部恒通量场的双侧稳态迹。谐和系数可精确保持跨面总通量，仍不满足这个局部条件。
2. 这个限制不能推广为所有共享非负交换都不可能。依赖完整沿段密度分布的串联阻力系数，在这一局部恒通量检验上可同时保持稳态和通量；它目前只是需要额外信息的解析本构参照，未实现为候选后端。
3. 构造了两套**点位置/点密度、逐单元质量、原质量平均初场及真实固定几何完全相同**的密度场，其跨界线段阻力严格不同。因此，沿段阻力不能在无附加先验时由这些有限输入唯一确定。

以下证明限定于有限h的局部一致性与有限输入信息；不构成一般正交换不可能、h趋零弱极限失败或现有FV基线失败的证明。它解释需要审查什么，未把上轮14.47%误差逐点或按因果份额重新归因。

## 1. 接续对象与假设

本轮执行 `runs/reconnect-contact-continuous/stage3-status.json` 的 `next_contact_constitutive_review_predeclared`，原909项科学清单保持不变。接续[固定h连续核对照](CONTACT_CONTINUOUS_KERNEL_CONTROL.md)、[二维接触冻结检验](PERIODIC_DENSITY_CONTACT_FROZEN.md)及[旧一维界面反例](../rebuild/KERNEL_JOINT_LIMIT.md)。

局部平面接触为a=0，法向位移记s，切向位移记z。两侧Gamma=rho D分别为ΓL、ΓR，均正。保留原径向核、完整圆盘支撑和因子2：

\[
\rho(x)\mathcal L_hc(x)
=2\int H(x,x+r) B_h(|r|)[c(x+r)-c(x)]\,dr,
\quad B_h=-W'_h/r.
\]

同相H=ΓL或ΓR；在本节受限模型类中，跨相H为一个共享常数K，可依赖ΓL、ΓR和h，却不依赖边两端距界面的相对位置。未假定负权，也没有通过拟合选择K。

使用局部透射场

\[
c(a)=\begin{cases}c_*+s_La,&a<0,\\c_*+s_Ra,&a>0,\end{cases}
\qquad \Gamma_Ls_L=\Gamma_Rs_R=J\ne0.
\]

这里J=+rho D c_a，物理Fick通量F=−J。局部PDE的加权作用为零。只需该场在核支撑覆盖的局部条带内成立，可取足够小斜率使浓度落在[0,1]；不声称在周期或封闭无通量域上存在全局非零恒通量稳态。

## 2. 原二维核的半空间一阶矩

原代码给出

\[
W_h(r)=\frac{9}{4\pi h^2}F(r/h),\quad
F(q)=(1-q/2)^6(1+3q+35q^2/12),\quad 0\le q\le2.
\]

定义正半空间法向矩

\[
A_h=\int_{s>0} B_h(\sqrt{s^2+z^2})s\,ds\,dz.
\]

由极坐标角积分与分部积分，

\[
A_h=2\int_0^{2h}B_h(r)r^2dr
=2\int_0^{2h}W_h(r)dr
=\frac{8}{3\pi h}>0.
\]

其中精确多项式积分为

\[
\int_0^2F(q)dq=16/27,\qquad
\int_0^2qF(q)dq=2/9.
\]

后者给出原始归一化及二阶矩

\[
\int_{\mathbb R^2} W_h(|r|)\,d^2r=1,\qquad
\int_{\mathbb R^2} B_h(|r|)s^2\,d^2r=1,\qquad
\int_{s>0}B_h(|r|)s^2\,ds\,dz=1/2.
\]

这些是原核的解析性质，没有重归一化核矩。多项式导数与原 `slope_over_radius` 的系数恒等式也用有理数精确核对。

## 3. 双侧迹要求不相容

对上面的局部场，分别取a趋于0的两侧极限。负/正半空间的奇矩给出

\[
S_h^-:=\lim_{a\uparrow0}\rho\mathcal L_hc
=2A_h(-\Gamma_Ls_L+Ks_R)
=2A_hJ(K/\Gamma_R-1),
\]

\[
S_h^+:=\lim_{a\downarrow0}\rho\mathcal L_hc
=2A_h(-Ks_L+\Gamma_Rs_R)
=2A_hJ(1-K/\Gamma_L).
\]

这里是**密度加权作用的迹**；真实变化率迹还要分别除以rhoL、rhoR。因为A_h>0、J非零，局部稳态迹同时为零要求

\[
S_h^-=0\Rightarrow K=\Gamma_R,\qquad
S_h^+=0\Rightarrow K=\Gamma_L.
\]

当ΓL≠ΓR时，两条件不相容，即使允许K随h变化也不能在每个固定h上同时成立。不相容性针对受限的相位常系数形式，不能推广到依赖边位置或路径信息的所有共享非负系数。

对原谐和系数

\[
K_H=\frac{2\Gamma_L\Gamma_R}{\Gamma_L+\Gamma_R},
\quad
S_h^-=S_h^+=\frac{16J}{3\pi h}
\frac{\Gamma_L-\Gamma_R}{\Gamma_L+\Gamma_R}.
\]

在本项目ΓL=1/50、ΓR=2/25的1:4接触中，K_H=4/125，双侧S/(2A_hJ)均为−3/5，不为零。

## 4. 为什么跨面总通量仍可精确

对法向跨界位移s>0，所有跨面配对的起点a范围为[−s,0]。有

\[
\int_{-s}^0[c(a+s)-c(a)]da
=\frac{s_L+s_R}{2}s^2.
\]

按原非局部交换定义，每单位切向长度的物理跨面通量为

\[
F_h=-2K\int_{s>0}B_h(r)
\int_{-s}^0[c(a+s)-c(a)]da\,ds\,dz
=-\frac K2(s_L+s_R).
\]

因此谐和K_H给出F_h=−J=F，却不保持双侧局部稳态。这与旧一维反例的机制一致，二维矩常数按本节重新推导。

| 精确代数对照，ΓL:ΓR=1:4 | K | S−/(2A_hJ) | S+/(2A_hJ) | F_h/F |
|---|---:|---:|---:|---:|
| 原谐和系数 | 4/125 | −3/5 | −3/5 | 1 |
| 左侧迹为零所需系数 | 2/25 | 0 | −3 | 5/2 |
| 右侧迹为零所需系数 | 1/50 | −3/4 | 0 | 5/8 |

后两行只是必要条件的解析对照，没有作为调参候选运行。它们不能修复原模型；分别给左右端使用不同K也不再是当前的共享对称电导结构。

## 5. 受限类之外：沿段阻力的解析参照

若**另外知道整条线段上的正Gamma分布**，可定义

\[
K_{path}(x,y)=\left[\int_0^1\Gamma(x+\theta(y-x))^{-1}d\theta\right]^{-1}.
\]

它为正，且由theta替换为1−theta立即得K_path(x,y)=K_path(y,x)。对仅随法向变化的局部恒通量场c_a=J/Gamma，

\[
c(y)-c(x)=Js\int_0^1\Gamma(x+\theta(y-x))^{-1}d\theta,
\]

于是

\[
K_{path}(x,y)[c(y)-c(x)]=Js.
\]

完整对称核支撑内的加权作用成为2J integral B_h(r)s dr=0，跨面总通量也由半空间二阶矩得到F_h=−J。它同时通过这一局部恒通量检验，说明上一节的不相容性不能称为一般正交换不可能。

该结果没有证明完整二维瞬态精度，也没有证明把cbar当节点未知量后仍精确保持上述关系。通过恒通量检验不保证瞬态优于普通FV；普通中心阻力物质FV基线继续保留。**本轮只给出解析本构参照，没有计算新电导、构图、替换后端或接纳流体ORC。**

沿段Gamma不是目前候选的有限节点rho数组。必须先说明它如何获得；不能把真实界面、物质面、F/C或未提供的密度积分自动作为候选输入。

## 6. 同一有限输入、不同阻力的显式反例

下面甚至额外固定真实方格和接触位置，排除几何变化。令N=128、Delta=1/N，使用当前静止周期接触的rho0(a)=1或4，以及当前同一个连续标量

\[
c(a,b)=1/2+(1/5)f(a)\cos(2\pi b),\qquad
f(a)=\cos(2\pi\xi(a)+\pi/5).
\]

每相内lambda_p=2pi kappa_p，kappaL=8/5、kappaR=2/5。在第i个法向单元，以中心为原点t=a−a_i，定义

\[
\omega=2\pi/\Delta,\quad
r_p=(\lambda_p/\omega)^2=(\kappa_p/N)^2,\quad
\alpha_p=\frac{4-r_p}{2(1-r_p)},
\]

\[
g_i(t)=\sin(\omega t)+\alpha_p\sin(2\omega t),\qquad
\rho_\epsilon(a,b)=\rho_0(a)[1+\epsilon g_i(t)],\quad \epsilon=1/100.
\]

alpha仅用于构造满足观测约束的数学反例，不进入候选电导，也不替换已保存的密度数据。

### 6.1 正性与相同点数据

精确数值为alphaL=8533/4266、alphaR=136533/68266。两相均有|g|≤1+alpha<3.01，因此

\[
\rho_\epsilon/\rho_0>1-0.01(3.01)=0.9699>0.
\]

在所有单元中心t=0和面t=±Delta/2处g=0。故两背景的全部节点rho、接触两侧rho迹、固定节点位置及静止轨迹一致。两背景都满足u=0下的连续性方程。

### 6.2 相同逐单元质量和相同原cbar

g是关于单元中心的奇函数，所以每个单元integral g da=0，逐单元质量完全相同。对标量约束，令beta=lambda_p Delta/2，直接积分得

\[
I_1=\int_{-\Delta/2}^{\Delta/2}\sin(\omega t)\sin(\lambda_pt)dt
=\frac{2\sin\beta}{\omega(1-r_p)},
\]

\[
I_2=\int_{-\Delta/2}^{\Delta/2}\sin(2\omega t)\sin(\lambda_pt)dt
=-\frac{4\sin\beta}{\omega(4-r_p)}.
\]

由alpha的定义，I1+alpha I2=0。再将f(a_i+t)展开为cos(phi_i)cos(lambda_p t)−sin(phi_i)sin(lambda_p t)，偶/奇性与该恒等式给出

\[
\int g_i f\,da=0.
\]

所以对每个二维原物质区域，

\[
\int_{K_i}\rho_\epsilon dV=\int_{K_i}\rho_0dV=m_i,
\qquad
\int_{K_i}\rho_\epsilon c\,dV=\int_{K_i}\rho_0c\,dV.
\]

不仅质量平均cbar相同，连续初始c本身也没有改变。m/rho_node与真实单元面积仍相同，但内部倒密度积分不同。接触面上的g为零，因此原初场的浓度及法向通量透射条件也保持。

### 6.3 跨界线段阻力严格不同

取相邻中心a−=1/2−Delta/2与a+=1/2+Delta/2，D=1/50。每单位横截面积的线阻力为

\[
R_\epsilon=\int_{a_-}^{a_+}\frac{da}{D\rho_\epsilon(a)},\qquad
R_0=\frac{5\Delta}{8D}=125/512.
\]

两个半单元分别有integral g_L da=Delta/pi、integral g_R da=−Delta/pi。使用精确恒等式

\[
\frac1{1+\epsilon g}=1-\epsilon g+
\frac{\epsilon^2g^2}{1+\epsilon g},
\]

得到

\[
\frac{R_\epsilon}{R_0}=1-\frac{6\epsilon}{5\pi}+V,
\quad 0<V<\frac{\epsilon^2(3.01)^2}{1-3.01\epsilon}.
\]

只用3<pi<22/7及有理数运算就有严格区间

\[
\boxed{\frac{249}{250}<\frac{R_\epsilon}{R_0}
<\frac{1063813031}{1066890000}<1.}
\]

即0.996<R_epsilon/R0<0.997115945412，明确排除1；这些是解析界，不是数值求积估计。

两个背景在当前全部有限输入上无法区分，线阻力却不同。该证明甚至固定了真实几何；它比只说明点轨迹不能识别单元体积的[既有反例](../rebuild/DENSITY_GEOMETRY.md)多约束了原质量平均初场。

### 6.4 反例的边界

- 它针对未知单元内部rho分布的有限输入问题。若另行给出严格分段常量密度、已知界面和采样几何等先验，可以排除此反例；这些先验必须显式列出。
- 反例的单元内振荡随N变化，未提供与N无关的导数界；它不排除在固定光滑背景及明确正则性假设下随分辨率提高而收敛。
- 相同的是给定背景点轨迹、rho采样和初始cbar，不是声称两个扩散问题未来的完整标量观测都相同。
- 这是运动学/被动标量层面的信息反例，不声称两个背景满足同一额外给定的完整Navier–Stokes热力学约束。
- 沿段线阻力是参考物理量，尚不是面向一般质量平均未知量的通用电导公式；不能据此直接替换普通FV或原图。

## 7. 输入与后端决策

| 信息/路线 | 本轮可作出的结论 | 使用边界 |
|---|---|---|
| 当前有限点位置、固定m、节点rho、原cbar | 不唯一确定一般内部密度与沿段阻力 | 需要显式闭合先验，不能宣布精确恢复 |
| 原相位常数谐和K | 恒通量总交换可正确，有限h双侧稳态迹失败 | 暂停该接触候选细化/瞬态接纳 |
| 完整连续rho的沿段串联阻力 | 可作为正对称、保持局部恒通量的解析参照 | 完整rho只属于oracle参考，不进入当前节点候选接口 |
| 普通中心阻力物质FV | 已有受限基线继续保留 | 不把它的额外几何/密度信息称为一般点后端已有信息 |

下一步只预定**同一连续场的oracle沿段阻力本构对照**：复用当前N128/h0.04/D0.02的连续初场、质量和PDE参考，按已有两组求积规则评估上述K_path的连续作用并观察到原物质区域，与已保存原谐和连续核比较。它需要当前解析密度全场，明确为oracle本构控制；不修改候选、不构图、不作瞬态，不用结果拟合状态或边权。检验局部恒通量条件是否对完整二维冻结PDE有帮助，不能因该必要条件通过而自动接纳新后端。

## 8. 证书、完整性与交付

`contact_constitutive_certificates.py` 以标准库Fraction保存6组精确代数核对：核多项式导数、归一化及半空间矩、相位常系数条件、保持cbar的正交系数、正密度下界、阻力严格分离区间。小数仅用于阅读。解析积分、极限和可辨识性论证写在本报告中，不将程序称为完整自动定理证明器。

本轮没有新增数值单元测试或物理算例测试，既有重建检查累计仍为**341项**；6组代数证书另列，不重复计数，不恢复旧v0.8.0/246项。782个原始文件、15个历史附件和43份先前正式科学清单保持；4份历史诊断清单已有路径差异另核。原README/METHOD/CODEX_TASK、后端和旧结果未改，仅更新可变进度入口 `docs/RESEARCH_STATUS.md`。

```bash
PYTHONDONTWRITEBYTECODE=1 /workspace/.venvs/ltn-reference/bin/python research/reconnect/contact_constitutive_certificates.py --output /workspace/contact-constitutive-review-reproduction
```

只复算精确代数证书时，可独立使用标准Python运行 `python contact_constitutive_certificates.py --certificates-only`，不需要下载历史流体数据；完整来源审计仍需历史证据链。

数据在 `runs/reconnect-contact-constitutive/`，保存精确证书、预定方案、执行记录和完整性审计。本轮未实现新本构后端、未运行流动或参考瞬态、未生成PDF/版本、未提交、推送、修改环境配置或发布。**Stage 3仍未通过，fluid ORC继续关闭。**

- [报告、证书与代码审阅包](../../exports/ltn-contact-constitutive-review.zip)。
- [完整交付及历史依赖](../../exports/ltn-contact-constitutive-downloads.md)。
- [交付哈希](../../exports/ltn-contact-constitutive-delivery.json)。
