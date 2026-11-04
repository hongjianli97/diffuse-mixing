"""One fixed-h quadrature probe for the finest joint-path contact result.

Not another physical/width scan. Refine only material quadrature at inherited h,
then aggregate mass averages to the exact previous 2048 material intervals.
"""
from pathlib import Path
import argparse, hashlib, json
import numpy as np
from fixed_width_kernel import build_constant_exchange
from kernel_joint_study import evolve, subdivide
from contact import Contact
from contact_step_reference import StepReference
from spatial_study import normalized_error


def main(source, output):
    source = Path(source).resolve(); output = Path(output).resolve()
    src = json.loads((source/'config.json').read_text())
    previous = json.loads((source/'results.json').read_text())['physical_runs'][-1]
    n0 = previous['cells']
    inputs = [source/'config.json', source/'results.json', source/f'physical-N{n0}.npz']
    output.mkdir(parents=True, exist_ok=False)
    cfg = {'purpose': 'one fixed-h quadrature-only probe; diagnose nonmonotone inventory along joint path',
           'source': str(source), 'parent_cells': n0, 'subdivision_factor': 4,
           'h': previous['h'], 'time': src['time'],
           'input_sha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs},
           'no_extrapolated_solution': True, 'no_coefficient_correction': True,
           'not_a_rigorous_error_bound': True, 'physical_acceptance': False, 'fluid_orc_enabled': False}
    (output/'config.json').write_text(json.dumps(cfg, indent=2)+'\n')
    with np.load(source/f'physical-N{n0}.npz') as z:
        parent = z['material_faces']; pm = z['mass']; pc0 = z['c0']
        prior = z['kernel_c']; pref = z['reference']
    faces = subdivide(parent, cfg['subdivision_factor']); V = np.diff(faces)
    x = (faces[:-1]+faces[1:])/2; factor = cfg['subdivision_factor']
    index = np.repeat(np.arange(n0), factor)
    m = pm[index]*V/np.diff(parent)[index]; c0 = pc0[index]
    if np.max(abs(np.bincount(index, weights=m)-pm)) > 1e-13:
        raise RuntimeError('fixed parent masses changed')
    p = src['physical_configuration']
    f = Contact(rho_left=p['rho_left'], rho_right=p['rho_right'], D=p['D'],
                half_length=p['half_length'], speed=p['speed'])
    ref = StepReference(f, left=p['initial_left'], right=p['initial_right'])
    rho = np.where(x < 0, f.rho_left, f.rho_right)
    L, arrays, diag = build_constant_exchange(x, m, V, rho, f.D, cfg['h'], (-1., 1.))
    c, evolution = evolve(L, m, c0, cfg['time'])
    agg = np.bincount(index, weights=m*c)/pm
    exact = ref.average(faces, cfg['time'])
    agg_ref = np.bincount(index, weights=m*exact)/pm
    refcheck = float(np.max(abs(agg_ref-pref)))
    if refcheck > 1e-12:
        raise RuntimeError('mass-average reference aggregation changed')
    invref, _, _ = ref.right_inventory(0., cfg['time'])
    inventory = float(m[x > 0]@c[x > 0])
    inventory_change = (inventory-previous['kernel_right_inventory'])/invref
    result = {'cells': len(m), 'parent_cells': n0, 'h': cfg['h'],
              'h_over_dx': float(cfg['h']/V.max()),
              'fine_full_error': normalized_error(m, c0, c-exact),
              'aggregated_full_error': normalized_error(pm, pc0, agg-pref),
              'change_from_parent_kernel': normalized_error(pm, pc0, agg-prior),
              'reference_aggregation_max_abs_residual': refcheck,
              'right_inventory': inventory,
              'inventory_signed_relative_error': (inventory-invref)/invref,
              'inventory_relative_change_from_parent': inventory_change,
              'operator_diagnostics': diag, 'evolution_checks': evolution,
              'physical_acceptance': False,
              'interpretation': 'one observed fixed-h grid difference, not a rigorous quadrature bound or local PDE convergence proof'}
    (output/'results.json').write_text(json.dumps(result, indent=2)+'\n')
    np.savez_compressed(output/'fixed-h-probe.npz', material_faces=faces, mass=m, c0=c0, kernel_c=c,
                        reference=exact, parent_mass=pm, parent_c0=pc0, parent_kernel=prior,
                        aggregated_kernel=agg, parent_reference=pref, parent_index=index,
                        positions=x, h=np.array(cfg['h']), **arrays)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('source'); p.add_argument('--output', required=True)
    a = p.parse_args(); main(a.source, a.output)
