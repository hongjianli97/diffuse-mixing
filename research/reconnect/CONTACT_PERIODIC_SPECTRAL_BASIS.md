# 同一周期接触的质量加权特征基与参考误差界

2026-11-01（UTC）。这是[全域冻结对照](FV_GLOBAL_FROZEN_COMPARISON.md)预先限定的有限时间参考推导。只处理原单位方形、等长两相、相同常数D、rho_L=1和rho_R=4的静止周期接触；不将结论推广到任意界面或变D。

## 1. 自伴算子、透射条件与完备性

在H=L²([0,1],rho dx)中取闭二次型

\[
\mathfrak a(f,g)=D\int_0^1\rho f'\overline{g'}dx,\qquad
V=H^1_{per}(0,1),\quad \rho=1\ (0<x<1/2),\ 4\ (1/2<x<1).
\]

rho有正上下界，故该形式非负、闭、稠密。关联自伴算子为A=−rho⁻¹(rho D f')'；其定义域为周期H¹、相内H²，且在x=1/2和周期接缝x=0/1处满足[f]=[rho D f']=0。它由分部积分的边界项确定。有限区间H¹到L²的嵌入紧，因此(A+I)⁻¹紧；谱为离散实非负谱，全部特征函数构成H的完备正交基。

零特征值只有常数：能量为零意味着f'=0。对于lambda>0，令k=sqrt(lambda/D)，相内有f''+k²f=0。用状态(f,rho f')跨越长度1/2，传递矩阵为

\[
M_\rho=\begin{pmatrix}\cos(k/2)&\sin(k/2)/(\rho k)\\
-\rho k\sin(k/2)&\cos(k/2)\end{pmatrix},\quad \det M_\rho=1.
\]

总传递M_4 M_1的迹为

\[
2\cos^2(k/2)-(4+1/4)\sin^2(k/2)
=2-(2+4+1/4)\sin^2(k/2).
\]

周期解存在要求det(M_4 M_1−I)=2−tr(M_4 M_1)=0，故k=2pi*n，n≥1。此时两个半区间矩阵均为(−1)^n I，总矩阵为I，特征空间二维。可选

\[
u_n(x)=\cos(2\pi nx),\qquad
v_n(x)=\frac{\sin(2\pi nx)}{\rho(x)},\quad
\lambda_n=4\pi^2D n^2.
\]

u_n在两接触处法向导数为零；v_n在两接触处函数值为零且rho*v_n'连续。两者满足周期透射条件。上面的传递计算穷尽正特征值，并给出其完整二维空间；结合紧自伴谱定理，常数、u_n、v_n完备。这里没有用有限截断的拟合表现代替完备性证明。

## 2. 正交归一化和初始投影

半区间的整数频率积分给出

\[
\|1\|_\rho^2=5/2,\quad
\langle u_n,u_m\rangle_\rho=(5/4)\delta_{nm},\quad
\langle v_n,v_m\rangle_\rho=(5/16)\delta_{nm},\quad
\langle u_n,v_m\rangle_\rho=0.
\]

常数与所有正频模式也正交；交叉项中的rho恰好消去，是完整周期的cos*sin积分。

原数学初场写为c_0(x,y)=1/2+f_0(x)cos(2pi*y)，其中

\[
f_0(x)=\tfrac15\cos(\pi(a_px+\phi_p)),\quad
(a_L,\phi_L)=(16/5,1/5),\quad(a_R,\phi_R)=(4/5,7/5).
\]

这就是原阻力坐标公式；相位、振幅不由保存c0或误差拟合。D保留原二进制精确分数，只影响衰减率，初始阻力坐标中的D消去。

令

\[
a_0=\frac{\langle f_0,1\rangle_\rho}{5/2}
=\frac{3\sin(\pi/5)}{4\pi}\ne0,\quad
a_n=\frac{\langle f_0,u_n\rangle_\rho}{5/4},\quad
b_n=\frac{\langle f_0,v_n\rangle_\rho}{5/16}.
\]

主投影逐相用cos*cos和cos*sin积化和差，再用三角原函数精确积分，不进行数值求积。由原相位的半区间反射对称，奇数n的a_n、偶数n的b_n精确为零。实现通过有理倍pi的精确周期/对称约简得到这些零，没有按误差大小裁剪系数。

另一路使用逐相Green恒等式：若e''=−k²e，f_0''=−q_p²f_0，则

\[
(k^2-q_p^2)\int_l^r f_0e\,dx
=[f_0'e-f_0e']_l^r,\quad q_p=\pi a_p.
\]

这里q_p/(2pi)为8/5或2/5，不等于整数n，没有共振分母。相乘rho并归一化，独立获得同一a_n、b_n。这一路只用端点值和导数，不调用积化和差积分，实际核对所有保留模式。

## 3. 有限时间解与真实材料单元平均

令tau=4pi²Dt。分离横向频率后，唯一弱解为

\[
c(x,y,t)=\tfrac12+e^{-\tau}\cos(2\pi y)
\left[a_0+\sum_{n\ge1}e^{-\tau n^2}(a_nu_n(x)+b_nv_n(x))\right].
\]

法向常数a_0仍按横向频率衰减，不能丢掉。该解允许原初场不满足时间迹的更高阶初始兼容性；完备谱作用于原初场，不排除它或修改初始冻结诊断。t>0的衰减给出所需的正则性。

原N=128固定材料单元与两接触对齐。每格内rho常数，故质量平均等于几何平均，但仍使用原质量。若x_i=(i+1/2)/N，y_j=(j+1/2)/N，则

\[
\begin{aligned}
\bar u_{n,i}&=\operatorname{sinc}(\pi n/N)\cos(2\pi nx_i),\\
\bar v_{n,i}&=\operatorname{sinc}(\pi n/N)\sin(2\pi nx_i)/\rho_i,\\
Y_j&=\operatorname{sinc}(\pi/N)\cos(2\pi y_j).
\end{aligned}
\]

这里sinc(z)=sin(z)/z。截断到n≤K的真实质量平均即

\[
\bar c_{ij}^{K}(t)=\tfrac12+Y_j e^{-\tau}
\left[a_0+\sum_{n=1}^{K}e^{-\tau n^2}(a_n\bar u_{n,i}+b_n\bar v_{n,i})\right].
\]

这不是轨迹点浓度。特别是n=N时单元平均模态精确为零，中心点值一般并不为零。

## 4. 不依赖结果拟合的尾项界

加权正交性与完备性给出尾项的连续质量范数界；单元投影由Jensen不等式收缩质量范数：

\[
\|P_m(c-c^K)\|_m\le\|c-c^K\|_{L^2(\rho)}
\le e^{-4\pi^2Dt[(K+1)^2+1]}\|c_0-1/2\|_{L^2(\rho)}.
\]

由于振幅1/5、横向cos²平均1/2，总质量5/2，初始范数平方≤1/20<1/16，故可取严格上界1/4。

为避免把Decimal.exp读数当作区间认证，另给完全有理数上界。Machin公式的交错arctan级数用12项和下一项包围pi，验证pi>157/50。对h=231/100，正项和sum_{j=0}^{12}h^j/j!>10，故exp(h)>10。令

\[
q_K=\left\lfloor\frac{4(157/50)^2DT[(K+1)^2+1]}{231/100}\right\rfloor.
\]

则固定T=1/20下有

\[
\boxed{\|P_m(c-c^K)\|_m\le\frac1{4\,10^{q_K}}.}
\]

D在此取原二进制精确Fraction。该证书只依赖模型、截断和时间，不使用保存参考缺陷、FV误差或曲线拟合。高精度读数另外给出，明确不冒充区间证明。

## 5. 保存均值舍入桥接的限定解释

先独立计算数学初场的真实单元平均，再保留delta_i=c0_saved_i−(P_m c0_math)_i；不从保存值重新拟合f_0，不把delta减掉或修改源c0。

若要把此均值舍入差带到有限时间，可以明确采用一个常值扰动提升：在每个原单元令eta=delta_i。它满足P_m eta=delta、||eta||_{L²(rho)}=||delta||_m。因此由PDE半群和材料投影的收缩性，

\[
\|P_m S_t(c0_{math}+\eta)-P_m S_t c0_{math}\|_m\le\|\delta\|_m.
\]

这只证明该**显式扰动提升**的条件性界。均值本身不能识别任意连续初场的亚单元结构；本界不是对所有具有相同均值的连续初场的误差界。实现不传播eta、不生成校正后参考，只将桥接差独立记入不确定度。

谱、模型和这些界均在执行固定时间参考前建立；本轮不构建FV/核算子、不进行候选传播、不改物理验收阈值。数值读数和审计另见 `CONTACT_FINITE_TIME_REFERENCE.md`。
