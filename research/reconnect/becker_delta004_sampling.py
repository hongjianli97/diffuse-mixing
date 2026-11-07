"""Sampling rule specialized to K8/16/32, without modifying old COUNTS."""
from fractions import Fraction as F
import ast, inspect
import numpy as np
from becker_point_core import T, sample_frames

def sample_frames_delta004(mu_edges, flow, rows=8):
    e = np.asarray(mu_edges)
    centres = (e[:-1] + e[1:]) / 2
    cols = len(centres)
    y = (np.arange(rows) + 0.5) / rows
    cases = {}
    for K in (8, 16, 32):
        frames = []
        for i in range(2 * K + 1):
            t = T * i / (2 * K)
            xx = flow.x_mu(centres, float(t))
            rho = flow.rho_mu(centres, float(t))
            bounds = flow.x_mu(e[[0, -1]], float(t))
            frames.append({'time_exact': str(t), 'positions': np.stack(np.meshgrid(xx, y, indexing='xy'), axis=-1).reshape(-1, 2).tolist(), 'rho': np.tile(rho, rows).tolist(), 'x_bounds': bounds.tolist(), 'period_y': 1.0, 'D': 0.02})
        cases[str(K)] = {'K': K, 'frames': frames}
    return {'ids': list(range(cols * rows)), 'columns': cols, 'rows': rows, 'cases': cases, 'point_rule': 'fixed mass-label midpoint trajectory,not geometriccellcentre', 'mass_c0_reference_internal_material_faces_or_F_C_supplied_to_builder': False}


def verify_sampling_rule():
    old=ast.parse(inspect.getsource(sample_frames)).body[0]
    new=ast.parse(inspect.getsource(sample_frames_delta004)).body[0]
    old.name=new.name
    for node in ast.walk(old):
        if isinstance(node,ast.For) and isinstance(node.iter,ast.Name) and node.iter.id=='COUNTS':
            node.iter=ast.parse('(8,16,32)',mode='eval').body
    assert ast.dump(old)==ast.dump(new) and T==F(1,10)
    return {'same_AST_except_name_and_K_tuple':True,'new_K':[8,16,32],'old_COUNTS_unchanged':True}
