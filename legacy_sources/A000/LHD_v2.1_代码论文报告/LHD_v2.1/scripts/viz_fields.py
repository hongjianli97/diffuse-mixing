"""Field visualisations per case: material labels (transport), position-only density, network diffusion scalar,
position-only vorticity, shock-crossing map, velocity from positions, stretch / compression fields (mixing)."""
import numpy as np, pickle, sys, time, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
fm.fontManager.addfont("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")
plt.rcParams.update({"font.family": ["Noto Sans CJK JP"], "font.size": 7, "axes.titlesize": 7.5, "figure.dpi": 130, "axes.unicode_minus": False})
import cases, lhdx as X, netdiff as ND, stretch_est as SE, classify as C
from bcom import ledger
from lhdp import step_readouts, tri_area, tri_to_node
OUT = "/home/claude/w/viz/"
import os
EN = os.environ.get("VIZ_EN", "0") == "1"
TR = [("物质标签与界面示踪点", "material labels and interface tracers"), ("网络扩散标量（质量形式）", "network diffusion scalar (mass metric)"), ("只凭位置的步涡量", "step vorticity from positions"),
      ("位置差分得到的速度", "velocity from position differences"), ("拉伸场", "stretch field"), ("链式张量，$x_0$ 等值线族", "chained tensor, iso-lines of $x_0$"), ("各向异性", "anisotropy"),
      ("压缩场", "compression field"), ("账本；不可压应为 0", "ledger; zero if incompressible"), ("双环流（不可压，解析速度场）：输运、扩散、涡量与掺混的可视化", "Double gyre (incompressible): transport, diffusion, vorticity and mixing"),
      ("物质标签（左右半区）与绕涡圆环", "labels (left/right) and ring around the vortex"), ("网络扩散标量", "network diffusion scalar"), ("速度（位置差分）", "velocity (position differences)"), ("绕左涡圆周族", "circles around the left vortex"), ("不可压应为 0", "zero if incompressible"),
      ("Lamb–Oseen 共转涡对（不可压）：输运、扩散、涡量与掺混", "Lamb-Oseen co-rotating pair: transport, diffusion, vorticity and mixing"), ("物质标签（驱动/被驱动气体）", "labels (driver / driven gas)"), ("账本密度相对误差（对 DNS）", "ledger density, relative error vs DNS"), ("账本密度相对误差", "ledger density, relative error"), ("账本密度", "ledger density"),
      ("C 通道检出的穿越时刻（初始位置）", "detected shock-crossing time (initial positions)"), ("步膨胀率", "step dilatation"), ("伪涡量 $\\hat\\omega$（跨激波三角形）", "pseudo-vorticity (triangles straddling the shock)"), ("Sod 激波管（可压，精确轨迹）：输运、密度、扩散与激波检测", "Sod shock tube: transport, density, diffusion and shock detection"),
      ("物质标签（内/外气体）与圆环", "labels (inner / outer gas) and ring"), ("穿越时刻（初始位置）", "crossing time (initial positions)"), ("圆周族", "circles"), ("柱面爆炸波（可压，精确轨迹）", "Cylindrical blast wave"),
      ("涡核物质与圆环", "vortex-core material and ring"), ("DNS 涡量", "DNS vorticity"), ("（位置）", " (positions)"), ("激波–等熵涡（可压，DNS）：涡核物质输运、密度与涡量", "Shock-vortex (DNS): core transport, density and vorticity"),
      ("重气体示踪点与界面", "heavy-gas tracers and interface"), ("DNS 密度", "DNS density"), ("DNS 扩散标量", "DNS diffusing scalar"), ("$\\phi_0$ 等值线族", "iso-lines of $\\phi_0$"), ("激波–重气泡（可压，DNS）：物质输运、密度、扩散与掺混", "Shock-bubble (DNS): transport, density, diffusion and mixing"),
      ("粘性：重气体与界面", "viscous: heavy gas and interface"), ("密度相对误差（对粘性 DNS）", "density error vs viscous DNS"), ("粘性激波–重气泡", "Viscous shock-bubble"), ("重气体与界面示踪点", "heavy gas and interface tracers"), ("密度相对误差（对 DNS）", "density error vs DNS"), ("单模 Richtmyer–Meshkov（可压，DNS）", "Single-mode Richtmyer-Meshkov (DNS)")]
def tr(s):
    if not EN: return s
    for a, b in TR: s = s.replace(a, b)
    return s
def sc(ax, P, v, cmap="viridis", vmin=None, vmax=None, s=0.8, title="", aspect="equal", cb=True):
    h = ax.scatter(P[:, 0], P[:, 1], c=v, s=s, cmap=cmap, vmin=vmin, vmax=vmax, lw=0, rasterized=True)
    ax.set_aspect(aspect); ax.set_xticks([]); ax.set_yticks([]); ax.set_title(tr(title))
    if cb: plt.colorbar(h, ax=ax, fraction=0.035, pad=0.02)
    return h
def labels(ax, P, lab, iface=None, title="", aspect="equal", s=0.8):
    ax.scatter(P[~lab, 0], P[~lab, 1], s=s, c="#aed6f1", lw=0, rasterized=True); ax.scatter(P[lab, 0], P[lab, 1], s=s, c="#e74c3c", lw=0, rasterized=True)
    if iface is not None: ax.scatter(P[iface, 0], P[iface, 1], s=s*1.5, c="k", lw=0, rasterized=True)
    ax.set_aspect(aspect); ax.set_xticks([]); ax.set_yticks([]); ax.set_title(tr(title))
def quiver(ax, P, U, title="", aspect="equal", n=900, scale=None):
    idx = np.random.default_rng(1).choice(len(P), min(n, len(P)), replace=False)
    ax.quiver(P[idx, 0], P[idx, 1], U[idx, 0], U[idx, 1], np.hypot(*U[idx].T), cmap="viridis", scale=scale, width=0.003)
    ax.set_aspect(aspect); ax.set_xticks([]); ax.set_yticks([]); ax.set_title(tr(title))
def omega_hat(fr, s, dt):
    P0, P1 = fr[s], fr[s+1]; tri = X.complex_peeled(P0)["tri"]; r = step_readouts(P0, P1, tri)
    Am = tri_area(0.5*(P0+P1), tri); om = 2*r["A0"]*r["p"].imag/(Am*dt); th = np.log(np.abs(r["J"]))/dt
    return tri_to_node(om, Am, tri, len(P0)), tri_to_node(th, Am, tri, len(P0))
def stretch_field(fr, h, bad, t0, s):
    F = SE.chained_F(fr, h, bad=bad, idx=np.arange(fr.shape[1]), keep={s})[s][0]
    ll, A = SE.vector_lnlam(F, t0); return ll, A
def crossing_map(hot):
    N, S1 = hot.shape; first = np.where(hot.any(1), hot.argmax(1), -1); return first
def run_net(fr, dt, c0, kap, m, rho, keep=()):
    r = ND.run(fr, dt, c0, kap, "mass", m=m, rho=rho, keep=set(keep)); r = dict(r); r.setdefault(0, np.asarray(c0, float)); return r

def fig_dg():
    vel, vort, grad = cases.dg_fields(); c = cases.case_dg(nx=160, ny=80, T=10.0, S=51); fr, t, dt = c["frames"], c["t"], c["dt"]; N = fr.shape[1]
    ss = [0, 17, 33, 50]; P0 = fr[0]; lab = P0[:, 0] < 1.0; d, _ = __import__("scipy.spatial").spatial.cKDTree(P0).query(P0, 2); h = np.median(d[:, 1])
    iface = np.abs(P0[:, 0]-0.5) < 0.6*h
    lnJ, hot, bad, tau = ledger(fr); m = X.star_area(P0)/3; c0 = 0.5*(1+np.tanh((P0[:, 0]-1)/0.1)); cn = run_net(fr, dt, c0, 1e-3, m, np.ones((len(t), N)), ss)
    fig, ax = plt.subplots(4, 4, figsize=(8.6, 9.2))
    for j, s in enumerate(ss):
        labels(ax[0, j], fr[s], lab, iface, "物质标签与界面示踪点 $t=%.1f$" % t[s]); sc(ax[1, j], fr[s], cn[s], "viridis", 0, 1, title="网络扩散标量（质量形式）$t=%.1f$" % t[s])
        om, th = omega_hat(fr, min(s, len(t)-2), dt); sc(ax[2, j], fr[s], om, "RdBu_r", -2, 2, title="只凭位置的步涡量 $\\hat\\omega$ $t=%.1f$" % t[s])
    s = 25; U = (fr[s+1]-fr[s])/dt; quiver(ax[3, 0], fr[s], U, "位置差分得到的速度 $t=%.1f$" % t[s])
    t0 = np.tile([0.0, 1.0], (N, 1)); ll, A = stretch_field(fr, h, bad, t0, 50)
    sc(ax[3, 1], fr[50], ll, "RdBu_r", -3, 3, title="拉伸场 $\\ln\\lambda$（链式张量，$x_0$ 等值线族）$t=10$")
    sc(ax[3, 2], fr[50], np.log10(np.maximum(A, 1)), "magma", 0, 2, title="各向异性 $\\log_{10}A$ $t=10$")
    sc(ax[3, 3], fr[50], -lnJ[50], "PuOr", -0.1, 0.1, title="压缩场 $-\\ln J$（账本；不可压应为 0）$t=10$")
    plt.suptitle(tr("双环流（不可压，解析速度场）：输运、扩散、涡量与掺混的可视化"), fontsize=9); plt.tight_layout(); plt.savefig(OUT+("viz_en_dg.pdf" if EN else "viz_dg.pdf")); plt.close()

def fig_lo():
    c = cases.case_lo(a=1.0, Rd=6.5, N=8000); fr, t, dt = c["frames"], c["t"], c["dt"]; N = fr.shape[1]; P0 = fr[0]
    ss = [0, 13, 27, 40]; lab = P0[:, 0] < 0; cen = c["centres"]; ring = np.abs(np.hypot(*(P0-np.array(cen[0])).T)-1.5) < 0.6*c["h"]
    lnJ, hot, bad, tau = ledger(fr); m = X.star_area(P0)/3; c0 = 0.5*(1+np.tanh(P0[:, 0]/0.3)); cn = run_net(fr, dt, c0, 0.02, m, np.ones((len(t), N)), ss)
    fig, ax = plt.subplots(4, 4, figsize=(8.6, 9.2))
    for j, s in enumerate(ss):
        labels(ax[0, j], fr[s], lab, ring, "物质标签（左右半区）与绕涡圆环 $t=%.1f$" % t[s]); sc(ax[1, j], fr[s], cn[s], "viridis", 0, 1, title="网络扩散标量 $t=%.1f$" % t[s])
        om, th = omega_hat(fr, min(s, len(t)-2), dt); sc(ax[2, j], fr[s], om, "RdBu_r", -1.5, 1.5, title="$\\hat\\omega$ $t=%.1f$" % t[s])
    s = 20; U = (fr[s+1]-fr[s])/dt; quiver(ax[3, 0], fr[s], U, "速度（位置差分）$t=%.1f$" % t[s])
    rr = np.hypot(*(P0-np.array(cen[0])).T); t0 = np.c_[-(P0[:, 1]-cen[0][1]), P0[:, 0]-cen[0][0]]/np.maximum(rr, 1e-9)[:, None]
    ll, A = stretch_field(fr, c["h"], bad, t0, 40)
    sc(ax[3, 1], fr[40], ll, "RdBu_r", -2.5, 2.5, title="拉伸场 $\\ln\\lambda$（绕左涡圆周族）$t=%.1f$" % t[40]); sc(ax[3, 2], fr[40], np.log10(np.maximum(A, 1)), "magma", 0, 2, title="各向异性 $\\log_{10}A$")
    sc(ax[3, 3], fr[40], -lnJ[40], "PuOr", -0.2, 0.2, title="压缩场 $-\\ln J$（不可压应为 0）")
    plt.suptitle(tr("Lamb–Oseen 共转涡对（不可压）：输运、扩散、涡量与掺混"), fontsize=9); plt.tight_layout(); plt.savefig(OUT+("viz_en_lo.pdf" if EN else "viz_lo.pdf")); plt.close()

def fig_sod():
    c = cases.case_sod(h=0.004); fr, t, dt = c["frames"], c["t"], c["dt"]; N = fr.shape[1]; P0 = fr[0]; x0 = c["x0"]
    ss = [0, 13, 27, 40]; lab = x0 < 0; iface = np.abs(x0-0.1) < 0.6*0.004
    lnJ, hot, bad, tau = ledger(fr); rho = c["rho"][0][None]*np.exp(-lnJ); c0 = 0.5*(1+np.tanh(x0/0.02)); cn = run_net(fr, dt, c0, 1e-3, c["mass"], rho, ss)
    fig, ax = plt.subplots(5, 4, figsize=(8.6, 7.4))
    for j, s in enumerate(ss):
        labels(ax[0, j], fr[s], lab, iface, "物质标签（驱动/被驱动气体）$t=%.3f$" % t[s], "auto", 1.2); ax[0, j].set_xlim(-0.5, 0.5)
        sc(ax[1, j], fr[s], rho[s], "magma", 0.1, 1.0, 1.2, "账本密度 $t=%.3f$" % t[s], "auto"); ax[1, j].set_xlim(-0.5, 0.5)
        sc(ax[2, j], fr[s], rho[s]/c["rho"][s]-1, "RdBu_r", -0.05, 0.05, 1.2, "账本密度相对误差", "auto"); ax[2, j].set_xlim(-0.5, 0.5)
        sc(ax[3, j], fr[s], cn[s], "viridis", 0, 1, 1.2, "网络扩散标量（质量形式）", "auto"); ax[3, j].set_xlim(-0.5, 0.5)
    cm = crossing_map(hot); sc(ax[4, 0], P0, np.where(cm >= 0, t[cm], np.nan), "viridis", t[0], t[-1], 1.2, "C 通道检出的穿越时刻（初始位置）", "auto"); ax[4, 0].set_xlim(-0.5, 0.5)
    s = 20; om, th = omega_hat(fr, s, dt); sc(ax[4, 1], fr[s], th, "PuOr", -40, 40, 1.2, "步膨胀率 $\\hat\\theta$ $t=%.3f$" % t[s], "auto"); ax[4, 1].set_xlim(-0.5, 0.5)
    sc(ax[4, 2], fr[s], om, "RdBu_r", -40, 40, 1.2, "伪涡量 $\\hat\\omega$（跨激波三角形）", "auto"); ax[4, 2].set_xlim(-0.5, 0.5)
    U = (fr[s+1]-fr[s])/dt; quiver(ax[4, 3], fr[s], U, "速度（位置差分）$t=%.3f$" % t[s], "auto", 400); ax[4, 3].set_xlim(-0.5, 0.5)
    plt.suptitle(tr("Sod 激波管（可压，精确轨迹）：输运、密度、扩散与激波检测"), fontsize=9); plt.tight_layout(); plt.savefig(OUT+("viz_en_sod.pdf" if EN else "viz_sod.pdf")); plt.close()

def fig_blast():
    c = cases.case_blast(h=0.022); fr, t, dt = c["frames"], c["t"], c["dt"]; N = fr.shape[1]; P0 = fr[0]; rs = c["rs"]
    ss = [0, 20, 40, 60]; lab = rs <= 0.2; iface = np.abs(rs-0.5) < 0.6*0.022
    lnJ, hot, bad, tau = ledger(fr); rho0 = np.where(rs <= 0.2, 6.0, 1.0); rho = rho0[None]*np.exp(-lnJ); m = rho0*X.star_area(P0)/3
    c0 = 0.5*(1+np.tanh((rs-0.5)/0.04)); cn = run_net(fr, dt, c0, 2e-4, m, rho, ss)
    fig, ax = plt.subplots(4, 4, figsize=(8.6, 9.0))
    for j, s in enumerate(ss):
        labels(ax[0, j], fr[s], lab, iface, "物质标签（内/外气体）与圆环 $t=%.0f$" % t[s]); sc(ax[1, j], fr[s], rho[s], "magma", 0, 6, title="账本密度 $t=%.0f$" % t[s])
        sc(ax[2, j], fr[s], cn[s], "viridis", 0, 1, title="网络扩散标量 $t=%.0f$" % t[s])
    cm = crossing_map(hot); sc(ax[3, 0], P0, np.where(cm >= 0, t[cm], np.nan), "viridis", title="穿越时刻（初始位置）")
    th = np.arctan2(P0[:, 1], P0[:, 0]); t0 = np.c_[-np.sin(th), np.cos(th)]; ll, A = stretch_field(fr, 0.022, bad, t0, 60)
    sc(ax[3, 1], fr[60], ll, "RdBu_r", -1.5, 1.5, title="拉伸场 $\\ln\\lambda$（圆周族）$t=25$"); sc(ax[3, 2], fr[60], -lnJ[60], "PuOr", -2.5, 2.5, title="压缩场 $-\\ln J$ $t=25$")
    s = 30; U = (fr[s+1]-fr[s])/dt; quiver(ax[3, 3], fr[s], U, "速度（位置差分）$t=%.0f$" % t[s])
    plt.suptitle(tr("柱面爆炸波（可压，精确轨迹）"), fontsize=9); plt.tight_layout(); plt.savefig(OUT+("viz_en_blast.pdf" if EN else "viz_blast.pdf")); plt.close()

def fig_sv():
    D = pickle.load(open("sv_dns.pkl", "rb")); fr, t = D["fr_in"], D["t"]; dt = t[1]-t[0]; Ev = D["Ev"]; N = fr.shape[1]; P0 = fr[0]
    ss = [0, 10, 20, 30]; cen = np.array([D["xv"], D["yv"]]); lab = np.hypot(*(P0-cen).T) < D["Rv"]; ring = np.abs(np.hypot(*(P0-cen).T)-2*D["Rv"]) < 0.6*0.005
    lnJ, hot, bad, tau = ledger(fr); rho = Ev["rho"][0][None]*np.exp(-lnJ)
    fig, ax = plt.subplots(5, 4, figsize=(8.6, 10.5))
    for j, s in enumerate(ss):
        labels(ax[0, j], fr[s], lab, ring, "涡核物质与圆环 $t=%.2f$" % t[s]); sc(ax[1, j], fr[s], rho[s], "magma", 0.5, 2.0, title="账本密度 $t=%.2f$" % t[s])
        sc(ax[2, j], fr[s], rho[s]/Ev["rho"][s]-1, "RdBu_r", -0.05, 0.05, title="账本密度相对误差（对 DNS）")
        om, th = omega_hat(fr, min(s, len(t)-2), dt); sc(ax[3, j], fr[s], om, "RdBu_r", -15, 15, title="$\\hat\\omega$（位置）$t=%.2f$" % t[s])
        sc(ax[4, j], fr[s], Ev["om"][s], "RdBu_r", -15, 15, title="DNS 涡量 $t=%.2f$" % t[s])
    plt.suptitle(tr("激波–等熵涡（可压，DNS）：涡核物质输运、密度与涡量"), fontsize=9); plt.tight_layout(); plt.savefig(OUT+("viz_en_sv.pdf" if EN else "viz_sv.pdf")); plt.close()

def fig_sb():
    D = pickle.load(open("sb_dns_dx0.005.pkl", "rb")); fr, t = D["frames"], D["t"]; dt = D["dt"]; E = D["E"]; N = fr.shape[1]; P0 = fr[0]
    Dd = pickle.load(open("sb_diff_dns.pkl", "rb"))
    ss = [0, 30, 50, 70]; phi = np.hypot(P0[:, 0]-D["xb"], P0[:, 1])-D["R"]; lab = phi < 0; iface = np.abs(phi) < 0.6*0.005
    lnJ, hot, bad, tau = ledger(fr); rho = E["rho"][0][None]*np.exp(-lnJ); m = E["rho"][0]*X.star_area(P0)/3
    cn = run_net(Dd["frames"], Dd["dt"], Dd["Y"][0], Dd["kappa"], Dd["rho"][0]*X.star_area(Dd["frames"][0])/3, Dd["rho"][0][None]*np.exp(-ledger(Dd["frames"])[0]), ss)
    fig, ax = plt.subplots(6, 4, figsize=(8.6, 11.5))
    for j, s in enumerate(ss):
        labels(ax[0, j], fr[s], lab, iface, "重气体示踪点与界面 $t=%.1f$" % t[s]); sc(ax[1, j], fr[s], rho[s], "magma", 0.5, 7, title="账本密度 $t=%.1f$" % t[s])
        sc(ax[2, j], fr[s], E["rho"][s], "magma", 0.5, 7, title="DNS 密度 $t=%.1f$" % t[s]); sc(ax[3, j], Dd["frames"][s], cn[s], "viridis", 0, 1, title="网络扩散标量（质量形式）$t=%.1f$" % t[s])
        sc(ax[4, j], Dd["frames"][s], Dd["Y"][s], "viridis", 0, 1, title="DNS 扩散标量 $t=%.1f$" % t[s])
    th = np.arctan2(P0[:, 1], P0[:, 0]-D["xb"]); t0 = np.c_[-np.sin(th), np.cos(th)]; ll, A = stretch_field(fr, 0.005, bad, t0, 70)
    sc(ax[5, 0], fr[70], ll, "RdBu_r", -2, 2, title="拉伸场 $\\ln\\lambda$（$\\phi_0$ 等值线族）$t=1.4$"); sc(ax[5, 1], fr[70], -lnJ[70], "PuOr", -0.8, 0.8, title="压缩场 $-\\ln J$ $t=1.4$")
    sc(ax[5, 2], fr[70], np.log10(np.maximum(A, 1)), "magma", 0, 2, title="各向异性 $\\log_{10}A$ $t=1.4$"); om, th_ = omega_hat(fr, 69, dt); sc(ax[5, 3], fr[69], om, "RdBu_r", -25, 25, title="$\\hat\\omega$（位置）$t=1.38$")
    plt.suptitle(tr("激波–重气泡（可压，DNS）：物质输运、密度、扩散与掺混"), fontsize=9); plt.tight_layout(); plt.savefig(OUT+("viz_en_sb.pdf" if EN else "viz_sb.pdf")); plt.close()

def fig_sbv():
    D = pickle.load(open("sbv_dns_dx0.0035.pkl", "rb")); fr, t = D["frames"], D["t"]; E = D["E"]; P0 = fr[0]
    phi = np.hypot(P0[:, 0]-D["xb"], P0[:, 1])-D["R"]; lab = phi < 0; iface = np.abs(phi) < 0.6*0.005
    lnJ, hot, bad, tau = ledger(fr); rho = E["rho"][0][None]*np.exp(-lnJ); ss = [30, 50, 70]
    th = np.arctan2(P0[:, 1], P0[:, 0]-D["xb"]); t0 = np.c_[-np.sin(th), np.cos(th)]; ll, A = stretch_field(fr, 0.005, bad, t0, 70)
    fig, ax = plt.subplots(3, 3, figsize=(8.0, 6.3))
    for j, s in enumerate(ss):
        labels(ax[0, j], fr[s], lab, iface, "粘性：重气体与界面 $t=%.1f$" % t[s]); sc(ax[1, j], fr[s], rho[s], "magma", 0.5, 7, title="账本密度 $t=%.1f$" % t[s])
    sc(ax[2, 0], fr[70], rho[70]/E["rho"][70]-1, "RdBu_r", -0.1, 0.1, title="密度相对误差（对粘性 DNS）$t=1.4$"); sc(ax[2, 1], fr[70], ll, "RdBu_r", -2, 2, title="拉伸场 $\\ln\\lambda$ $t=1.4$"); sc(ax[2, 2], fr[70], -lnJ[70], "PuOr", -0.8, 0.8, title="压缩场 $-\\ln J$ $t=1.4$")
    plt.suptitle(tr("粘性激波–重气泡（$Re\\approx1300$，$\\Delta x=0.0035$）"), fontsize=9); plt.tight_layout(); plt.savefig(OUT+("viz_en_sbv.pdf" if EN else "viz_sbv.pdf")); plt.close()

def fig_rm():
    D = pickle.load(open("rm_dns_dx0.005.pkl", "rb")); fr, t = D["frames"], D["t"]; dt = D["dt"]; E = D["E"]; N = fr.shape[1]; P0 = fr[0]; a0 = D["a0"]
    xi = 0.5+a0*np.cos(2*np.pi*P0[:, 1]); phi = (P0[:, 0]-xi)/np.sqrt(1+(2*np.pi*a0*np.sin(2*np.pi*P0[:, 1]))**2); lab = phi > 0; iface = np.abs(phi) < 0.6*0.004
    lnJ, hot, bad, tau = ledger(fr); rho = E["rho"][0][None]*np.exp(-lnJ); ss = [0, 15, 30, 50]
    yy = P0[:, 1]; tv = np.c_[-2*np.pi*a0*np.sin(2*np.pi*yy), np.ones(N)]; tv /= np.linalg.norm(tv, axis=1)[:, None]; ll, A = stretch_field(fr, 0.004, bad, tv, 50)
    fig, ax = plt.subplots(4, 4, figsize=(8.6, 9.2))
    for j, s in enumerate(ss):
        labels(ax[0, j], fr[s], lab, iface, "重气体与界面示踪点 $t=%.1f$" % t[s], "auto"); sc(ax[1, j], fr[s], rho[s], "magma", 1, 7, title="账本密度 $t=%.1f$" % t[s], aspect="auto")
        om, th_ = omega_hat(fr, min(s, len(t)-2), dt); sc(ax[2, j], fr[s], om, "RdBu_r", -8, 8, title="$\\hat\\omega$（位置）$t=%.1f$" % t[s], aspect="auto")
    sc(ax[3, 0], fr[50], rho[50]/E["rho"][50]-1, "RdBu_r", -0.1, 0.1, title="密度相对误差（对 DNS）$t=2$", aspect="auto"); sc(ax[3, 1], fr[50], ll, "RdBu_r", -0.6, 0.6, title="拉伸场 $\\ln\\lambda$ $t=2$", aspect="auto")
    sc(ax[3, 2], fr[50], -lnJ[50], "PuOr", -1.2, 1.2, title="压缩场 $-\\ln J$ $t=2$", aspect="auto"); s = 10; U = (fr[s+1]-fr[s])/dt; quiver(ax[3, 3], fr[s], U, "速度（位置差分）$t=%.1f$" % t[s], "auto")
    plt.suptitle(tr("单模 Richtmyer–Meshkov（可压，DNS）"), fontsize=9); plt.tight_layout(); plt.savefig(OUT+("viz_en_rm.pdf" if EN else "viz_rm.pdf")); plt.close()

if __name__ == "__main__":
    import os; os.makedirs(OUT, exist_ok=True)
    for nm in sys.argv[1:]:
        t0 = time.time(); globals()["fig_"+nm](); print(nm, "done %.0fs" % (time.time()-t0), flush=True)
