"""Certify a stored fixed-grid budget; no additional optimization or scan."""
from pathlib import Path
from fractions import Fraction
import argparse,json
import numpy as np
from scipy.sparse import csr_matrix
from moment_defect_budget import budget_problem
from budget_certificates import separator_budget_lower,posterior_budget_upper,record


def main(source):
    root=Path(__file__).resolve().parents[2];source=Path(source).resolve();out=source/'budget-certificates.json'
    if out.exists():raise RuntimeError('refusing to overwrite budget certificates')
    cfg=json.loads((source/'config.json').read_text());old=root/'runs'/Path(cfg['source']).name
    previous=json.loads((old/'results.json').read_text());dy=previous['exact_embedded_separator']['sparse_dual']
    dual={int(i):Fraction(int(v['numerator']),int(v['denominator'])) for i,v in dy.items()}
    with np.load(source/'moment-budget.npz') as z:
        A=csr_matrix((z['A_data'],z['A_indices'],z['A_indptr']),shape=tuple(z['A_shape']));b=z['moment_rhs_mass_scaled']
        problem=budget_problem(A,b,z['mass'],z['interior_cells'],cfg['physical_parameters']['D'],.5)
        lower=separator_budget_lower(A,b,problem['second_rows'],problem['scale'],dual)
        if 'raw_conductances' not in z:raise RuntimeError('no numerical budget witness available for certification')
        upper,arrays=posterior_budget_upper(problem,z['raw_conductances'])
    result={'lower':lower,'upper':upper,'physical_acceptance':False,'fluid_orc_enabled':False,
        'scope':'optimality bracket for the stored fixed-grid budget only; not continuum accuracy or boundary/face validation',
        'additional_optimization_or_parameter_search_run':False,'raw_conductances_modified':False}
    if lower['verified'] and upper['verified']:
        lo=Fraction(int(lower['exact_lower_bound']['numerator']),int(lower['exact_lower_bound']['denominator']))
        hi=Fraction(int(upper['exact_upper_bound']['numerator']),int(upper['exact_upper_bound']['denominator']))
        if hi<lo:raise RuntimeError('inconsistent certified bounds')
        result['exact_optimality_interval_width']=record(hi-lo)
    np.savez_compressed(source/'budget-proof-basis.npz',**arrays)
    out.write_text(json.dumps(result,indent=2)+'\n')
    small={'lower_verified':lower['verified'],'lower':lower.get('exact_lower_bound',{}).get('float'),
        'upper_verified':upper['verified'],'upper':upper.get('exact_upper_bound',{}).get('float'),
        'inverse_residual':upper.get('exact_inverse_residual_norm',{}).get('float'),
        'correction_bound':upper.get('correction_bound_kg_per_s',{}).get('float'),
        'basis_minimum':upper.get('minimum_basis_weight_kg_per_s',{}).get('float'),
        'interval_width':result.get('exact_optimality_interval_width',{}).get('float')}
    print(json.dumps(small,indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('source');a=p.parse_args();main(a.source)
