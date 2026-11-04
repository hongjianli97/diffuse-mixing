"""One predeclared joint refinement path on the same physical scalar contact.

Constant kernel width removes variable-width ambiguity. Separate fixed-h
nonuniform midpoint quadrature controls retain the old 436 material cells.
"""
from pathlib import Path
import argparse, hashlib, json, platform
import numpy as np
import scipy
from scipy.linalg import eigh, eigh_tridiagonal
from scipy.sparse import diags, save_npz
from scipy.sparse.linalg import expm_multiply
from contact import Contact
from contact_step_reference import StepReference
from fixed_width_kernel import build_constant_exchange
from spatial_study import normalized_error


def verify(root, directory):
    entries = json.loads((directory/'sha256.json').read_text())
    for rel, digest in entries.items():
        if hashlib.sha256((root/rel).read_bytes()).hexdigest() != digest:
            raise RuntimeError('source evidence changed: '+rel)


def subdivide(faces, factor):
    """Subdivide each given interval exactly; no state or mass repair."""
    return np.r_[np.concatenate([np.linspace(a, b, factor+1)[:-1]
                  for a, b in zip(faces[:-1], faces[1:])]), faces[-1]]


def evolve(L, m, c0, t, full_eigen_check=False):
    Q = -diags(1/m) @ L
    c = expm_multiply(t*Q, c0, traceA=float(t*Q.diagonal().sum()))
    # Similarity-transformed independent calculation checks scaling and sign;
    # dense eigen evolution is added on the first two physical grids only.
    sm = np.sqrt(m); S = -diags(1/sm) @ L @ diags(1/sm)
    other = expm_multiply(t*S, sm*c0, traceA=float(t*S.diagonal().sum()))/sm
    check = float(np.max(abs(c-other)))
    if full_eigen_check:
        vals, U = eigh(-S.toarray())
        ce = (U@(np.exp(-t*vals)*(U.T@(sm*c0))))/sm
        check = max(check, float(np.max(abs(c-ce))))
    row_defect = float(np.max(abs(np.asarray(Q.sum(axis=1)).ravel())))
    mass_defect = float(np.max(abs(np.asarray(m@Q).ravel()))/m.max())
    mean = m@c0/m.sum()
    diag = {'independent_evolution_max_abs_difference': check,
            'dense_eigen_check_run': full_eigen_check,
            'relative_scalar_mass_drift': abs(float(m@(c-c0)))/abs(float(m@c0)),
            'generator_row_sum_residual': row_defect,
            'generator_relative_mass_residual': mass_defect,
            'minimum_concentration': float(c.min()), 'maximum_concentration': float(c.max()),
            'initial_half_variance': float(.5*m@((c0-mean)**2)),
            'final_half_variance': float(.5*m@((c-mean)**2))}
    if check > 1e-10 or diag['relative_scalar_mass_drift'] > 1e-10:
        raise RuntimeError('evolution or mass check failed')
    if c.min() < c0.min()-1e-10 or c.max() > c0.max()+1e-10:
        raise RuntimeError('raw concentration bounds failed')
    if diag['final_half_variance'] > diag['initial_half_variance']+1e-12:
        raise RuntimeError('energy check failed')
    return c, diag


def main(source, output):
    root = Path(__file__).resolve().parents[2]
    source = Path(source).resolve(); output = Path(output).resolve()
    verify(root, source)
    prior = source/'consistency-control/config.json'
    oldcfg = json.loads(prior.read_text())
    physical = json.loads((source/'config.json').read_text())['physical_configuration']
    f = Contact(rho_left=physical['rho_left'], rho_right=physical['rho_right'],
                D=physical['D'], half_length=physical['half_length'], speed=physical['speed'])
    ref = StepReference(f, left=physical['initial_left'], right=physical['initial_right'])
    geometry = root/'runs/rebuild-stage3-local-layer'
    verify(root, geometry)
    output.mkdir(parents=True, exist_ok=False)
    cfg = {'study': 'same physical contact, one inherited joint width/quadrature path',
           'source': str(source), 'physical_configuration': physical,
           'time': .1, 'cells': oldcfg['joint_path_cells'],
           'joint_rule': oldcfg['joint_path_rule'],
           'new_mesh': 'uniform interface-aligned material cells on [-1,1]; each backend uses identical m and exact c0',
           'old_436_mesh_not_replaced': True,
           'width': 'one global h=.25*sqrt(dx), same existing 1D normalized kernel; no correction',
           'nonuniform_control': {'source_mesh': str(geometry/'mesh-q8.npz'),
                                  'subdivision_factors': [1, 2, 4], 'fixed_h': .03125,
                                  'purpose': 'fixed continuous operator, refine only nonuniform midpoint quadrature'},
           'time_evolution': 'semidiscrete sparse exponential; no BE calibration claimed',
           'physical_acceptance': False, 'fluid_orc_enabled': False, 'third_stage_complete': False}
    (output/'config.json').write_text(json.dumps(cfg, indent=2)+'\n')
    # Inspect the exact same old nonuniform geometry, then subdivide its
    # material cells without moving an original face. No PDE solves here.
    with np.load(geometry/'mesh-q8.npz') as z:
        base = z['faces']; base_m = z['child_mass']; base_c0 = z['child_c0']
    controls = []
    for factor in cfg['nonuniform_control']['subdivision_factors']:
        faces = subdivide(base, factor); V = np.diff(faces); x = (faces[:-1]+faces[1:])/2
        rho = np.where(x < 0, f.rho_left, f.rho_right)
        # Inherited fixed parent masses split geometrically, not normalized.
        index = np.repeat(np.arange(len(base_m)), factor)
        m = base_m[index]*V/np.diff(base)[index]; c0 = base_c0[index]
        if np.max(abs(np.bincount(index, weights=m)-base_m)) > 1e-13:
            raise RuntimeError('nonuniform control changed parent mass')
        h = cfg['nonuniform_control']['fixed_h']
        L, arrays, d = build_constant_exchange(x, m, V, rho, f.D, h, (-1., 1.))
        controls.append({'subdivision_factor': factor, 'maximum_width_over_h': float(V.max()/h), **d})
        np.savez_compressed(output/f'nonuniform-split{factor}.npz', faces=faces, mass=m, c0=c0,
                            parent_index=index, h=np.array(h), **arrays)
        print('nonuniform quadrature', factor, 'M1 max', d['interior_first_moment_max_abs'], flush=True)

    rows = []; t = cfg['time']; invref, invtail, images = ref.right_inventory(0., t)
    for n in cfg['cells']:
        faces = np.linspace(-f.half_length, f.half_length, n+1)
        V = np.diff(faces); x = (faces[:-1]+faces[1:])/2
        m = f.mass(faces); c0 = ref.average(faces, 0.)
        rho = np.where(x < 0, f.rho_left, f.rho_right)
        dx = 2*f.half_length/n; h = .25*np.sqrt(dx)
        L, arrays, d = build_constant_exchange(x, m, V, rho, f.D, h, (-1., 1.))
        c, time_diag = evolve(L, m, c0, t, full_eigen_check=n <= 512)
        g = f.conductance(faces, 'ordinary_volume_center')
        ld = np.r_[g, 0.]+np.r_[0., g]
        LFV = diags([-g, ld, -g], [-1, 0, 1], shape=(n, n)).tocsr()
        fv, fvdiag = evolve(LFV, m, c0, t)
        vals, U = eigh_tridiagonal(ld/m, -g/np.sqrt(m[:-1]*m[1:]))
        independent_fv = (U@(np.exp(-t*vals)*(U.T@(np.sqrt(m)*c0))))/np.sqrt(m)
        fvcheck = float(np.max(abs(fv-independent_fv)))
        if fvcheck > 1e-10:
            raise RuntimeError('independent FV eigen check failed')
        K = ref.terms_for(t); exact = ref.average(faces, t, K); exact2 = ref.average(faces, t, 2*K)
        right = x > 0
        inv = float(m[right]@c[right]); invfv = float(m[right]@fv[right])
        row = {'cells': n, 'time': t, 'dx': dx, 'h': h, 'h_over_dx': h/dx,
               'kernel_full_error': normalized_error(m, c0, c-exact),
               'FV_full_error': normalized_error(m, c0, fv-exact),
               'kernel_right_inventory': inv, 'FV_right_inventory': invfv,
               'reference_right_inventory': invref,
               'kernel_inventory_signed_relative_error': (inv-invref)/invref,
               'FV_inventory_signed_relative_error': (invfv-invref)/invref,
               'reference_concentration_tail_bound': ref.tail_bound(t, K),
               'reference_K_to_2K_error': normalized_error(m, c0, exact-exact2),
               'reference_inventory_tail_bound': invtail,
               'FV_independent_eigen_max_difference': fvcheck,
               'operator_diagnostics': d, 'kernel_evolution_checks': time_diag, 'FV_evolution_checks': fvdiag}
        rows.append(row)
        np.savez_compressed(output/f'physical-N{n}.npz', material_faces=faces, mass=m, c0=c0, rho=rho,
                            volume=V, positions=x, h=np.array(h), reference=exact,
                            reference_2K=exact2, kernel_c=c, FV_c=fv, FV_g=g, **arrays)
        save_npz(output/f'operator-N{n}.npz', L)
        print('physical', n, 'kernel error', row['kernel_full_error'], 'FV', row['FV_full_error'],
              'inventory', row['kernel_inventory_signed_relative_error'], flush=True)
    result = {'nonuniform_fixed_h_controls': controls, 'physical_runs': rows,
              'physical_acceptance': False,
              'limits': ['one late time only', 'global constant h, not variable-h validation',
                         'different uniform meshes from old 436-cell experiment; same-input matching is per current grid',
                         'oracle density only', 'not historical 2D kernel recovery']}
    (output/'results.json').write_text(json.dumps(result, indent=2)+'\n')
    (output/'environment.json').write_text(json.dumps({'python': platform.python_version(),
                        'numpy': np.__version__, 'scipy': scipy.__version__, 'BLAS_threads_command': 4}, indent=2)+'\n')


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('source'); p.add_argument('--output', required=True)
    a = p.parse_args(); main(a.source, a.output)
