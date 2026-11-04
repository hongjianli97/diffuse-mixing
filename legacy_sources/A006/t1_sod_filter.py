"""T1: compression filter on exact Sod kinematics (kappa = 0, labels frozen -> true mixing state unchanged).
Each candidate network indicator is evaluated on the same ROI before (t0) and after (t1) the shock has swept it."""
import numpy as np, json, sys
sys.path.insert(0, '/home/claude/w'); sys.path.insert(0, '/home/claude/w/LHD_v3.2/lhd')
import cases as C, netmix as NM
from scipy.spatial import cKDTree
c = C.case_sod(h=0.004, Ly=0.12, S=41)
fr, h = c['frames'], c['h']; P0 = fr[0]; x0 = c['x0']; N = len(P0)
rho0 = np.where(x0 > 0, 0.125, 1.0)
net = NM.Net(fr, c['dt'], rho0)
eta = c['info']['rsR'] / 0.125
roi = (P0[:, 0] > 0.12) & (P0[:, 0] < 0.33) & (P0[:, 1] > 0.02) & (P0[:, 1] < 0.10)
labs = {}
s45 = (np.floor((P0[:, 0] + P0[:, 1]) / (np.sqrt(2) * 4 * h)) % 2 == 1)
labs['stripes45'] = (s45 & roi).astype(float)
labs['circle'] = (np.hypot(P0[:, 0] - 0.225, P0[:, 1] - 0.06) < 0.035).astype(float)
th = np.linspace(0, 2 * np.pi, 20001)[:-1]
pred_L = {'stripes45': np.sqrt(0.5 + 0.5 / eta ** 2), 'circle': np.mean(np.sqrt(np.cos(th) ** 2 / eta ** 2 + np.sin(th) ** 2))}
# smooth profile for the resolved Dirichlet energy (period 16h along (1,1)/sqrt2)
csm = np.where(roi, 0.5 + 0.5 * np.sin(2 * np.pi * (P0[:, 0] + P0[:, 1]) / np.sqrt(2) / (16 * h)), 0.5)
pred_E = (eta ** 2 + 1) / (2 * eta)          # Gamma = const closure: J |F^-T n|^2 averaged, F=diag(1/eta,1)
s0, s1 = 0, c['frames'].shape[0] - 1
def assort(s, lab, weighted):
    E, w = net.E[s], net.w[s]; keep = roi[E[:, 0]] & roi[E[:, 1]]
    E, w = E[keep], (w[keep] if weighted else np.ones(keep.sum()))
    a, b = lab[E[:, 0]], lab[E[:, 1]]; W = 2 * w.sum()
    e11 = 2 * (w * (a * b)).sum() / W; e00 = 2 * (w * ((1 - a) * (1 - b))).sum() / W
    a1 = (w * (a + b)).sum() / W; a0 = 1 - a1
    return (e11 + e00 - a1 ** 2 - a0 ** 2) / (1 - a1 ** 2 - a0 ** 2)
def eps_graph(s, r):
    P = fr[s]; tr = cKDTree(P); pairs = tr.query_pairs(r, output_type='ndarray')
    deg = np.bincount(pairs.ravel(), minlength=N).astype(float)
    # clustering: triangles via sparse adjacency
    import scipy.sparse as sp
    A = sp.csr_matrix((np.ones(2 * len(pairs)), (np.r_[pairs[:, 0], pairs[:, 1]], np.r_[pairs[:, 1], pairs[:, 0]])), shape=(N, N))
    tri = np.asarray((A @ A).multiply(A).sum(1)).ravel() / 2
    cl = np.where(deg > 1, 2 * tri / np.maximum(deg * (deg - 1), 1), np.nan)
    return deg, cl
res = {'eta': eta}
for name, lab in labs.items():
    r = {}
    r['var_mass'] = [float(NM.seg_intensity(net.m, lab, lab))] * 2
    r['assort_comb'] = [assort(s0, lab, False), assort(s1, lab, False)]
    r['assort_cot'] = [assort(s0, lab, True), assort(s1, lab, True)]
    cu0, cu1 = net.cut(s0, lab), net.cut(s1, lab)
    r['ncut_edges'] = [cu0['ncut'], cu1['ncut']]
    r['L_net'] = [cu0['Lnet'], cu1['Lnet']]; r['L_pred_ratio'] = float(pred_L[name])
    res[name] = r
r = {}
r['E_smooth_cot'] = [net.dirichlet(s0, csm), net.dirichlet(s1, csm)]; r['E_pred_ratio'] = float(pred_E)
# combinatorial Laplacian energy of the same smooth field
def comb_E(s, cc):
    E = net.E[s]; return float(((cc[E[:, 0]] - cc[E[:, 1]]) ** 2).sum())
r['E_smooth_comb'] = [comb_E(s0, csm), comb_E(s1, csm)]
d0, c0_ = eps_graph(s0, 1.5 * h); d1, c1_ = eps_graph(s1, 1.5 * h)
r['eps_degree'] = [float(d0[roi].mean()), float(d1[roi].mean())]
r['eps_clustering'] = [float(np.nanmean(c0_[roi])), float(np.nanmean(c1_[roi]))]
fc = net.far_contacts(s0, s1, 3 * h)
r['raw_contacts_per_node_window'] = float(fc['raw'][roi].mean() - np.bincount(net.E[s0].ravel(), minlength=N)[roi].mean())
r['far_contacts_per_node'] = float(fc['far'][roi].mean())
# Delaunay flips (turnover) in ROI
k0 = set(map(tuple, net.E[s0][roi[net.E[s0][:, 0]] & roi[net.E[s0][:, 1]]]))
k1 = set(map(tuple, net.E[s1][roi[net.E[s1][:, 0]] & roi[net.E[s1][:, 1]]]))
r['delaunay_edges_changed_frac'] = len(k0 ^ k1) / len(k0 | k1)
res['field'] = r
json.dump(res, open('/home/claude/w/out/res_t1_sod.json', 'w'), indent=1)
print(json.dumps(res, indent=1))
