"""Independent stdlib source/old-array/algebra audit; no flow evaluation or solver import."""
from pathlib import Path
from fractions import Fraction as F
import argparse, ast, hashlib, json, struct, time, zipfile

ROOT = Path(__file__).resolve().parents[2]


def read(path):
    return json.loads(Path(path).read_text())


def digest(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f,'sha256').hexdigest()


def npz_array(path,key):
    with zipfile.ZipFile(path) as z:
        b = z.read(key+'.npy')
    assert b[:6] == b'\x93NUMPY' and b[6] in (1,2)
    size = 2 if b[6] == 1 else 4
    count = int.from_bytes(b[8:8+size],'little')
    header = ast.literal_eval(b[8+size:8+size+count].decode('latin1'))
    assert header['descr'] == '<f8' and not header['fortran_order'] and len(header['shape']) == 1
    values = struct.unpack('<'+'d'*header['shape'][0],b[8+size+count:])
    return values


def monomial(*factors):
    coefficient = F(1)
    powers = {}
    for c,p in factors:
        coefficient *= c
        for k,v in p.items():
            powers[k] = powers.get(k,0)+v
    return coefficient,{k:v for k,v in powers.items() if v}


def main(directory,write=False):
    started = time.monotonic()
    out = Path(directory).resolve()
    p = read(ROOT/'research/reconnect/becker_delta_definition_protocol.json')
    d = read(out/'definition.json')
    parent = ROOT/'runs/reconnect-becker-common-k32'
    assert digest(parent/'sha256.json') == p['parent_seal_sha256']
    seal = read(parent/'sha256.json')
    for name,h in p['sources'].items():
        assert seal[name] == h == digest(ROOT/name) == digest(out/'source'/name),name
    for name,h in read(out/'implementation-lock.json').items():
        assert digest(ROOT/name) == h,name
    assert digest(ROOT/p['contract']) == d['contract_sha256']
    for row in read(out/'source-excerpts.json'):
        text = (out/'source'/row['path']).read_text()
        assert row['sha256'] == p['sources'][row['path']]
        if row['name'] in ('p[-1]==flow.total','H[K]=old','分别细化 h、Δt、物理层厚度δ','每个空间实验至少三档 h'):
            assert text.splitlines()[row['line']-1] == row['text']
        else:
            fn = next(n for n in ast.walk(ast.parse(text)) if isinstance(n,ast.FunctionDef) and n.name == row['name'])
            assert fn.lineno == row['line'] and ast.get_source_segment(text,fn) == row['text']
    src = out/'source/runs/reconnect-becker-spatial'
    col = src/'n32/column.npz'
    edges,m,c0 = (npz_array(col,key) for key in ('mass_edges','mass','c0'))
    parent_edges = npz_array(src/'n128/column.npz','mass_edges')
    assert len(m) == 32 and edges[0] == 0 and parent_edges[::4] == edges
    assert tuple(b-a for a,b in zip(edges[:-1],edges[1:])) == m
    material = out/'source/runs/reconnect-becker-common-k32/n32/material.npz'
    assert npz_array(material,'mass') == tuple(v/8 for v in m)*8
    assert npz_array(material,'c0') == c0*8
    inv = read(out/'input-inventory.json')
    assert inv['fixed_M_star'] == edges[-1] and inv['fixed_M_star_hex'] == edges[-1].hex()
    assert inv['fixed_M_star_exact'] == str(F(edges[-1]))
    assert inv['candidate_mass_extrema'] == [min(m),max(m)] and inv['candidate_c0_extrema'] == [min(c0),max(c0)]
    mesh = read(src/'reference/mesh-contract.json')
    refs = []
    for level,k in ((0,8192),(1,8192),(1,16384)):
        q = src/f'reference/input-l{level}-k{k}.npz'
        e,mm,cc = (npz_array(q,key) for key in ('mass_edges','mass','c0'))
        starts = [0]
        for n in mesh['counts']:
            starts.append(starts[-1]+n*2**level)
        assert tuple(e[i] for i in starts) == parent_edges and e[0] == 0 and e[-1] == edges[-1]
        assert tuple(b-a for a,b in zip(e[:-1],e[1:])) == mm
        assert len(mm) == (16486 if level == 0 else 32972)
        refs.append((e,mm,cc))
    assert refs[1][0][::2] == refs[0][0] and refs[1] == refs[2]
    # Free-symbol coefficient checks; no selected delta is evaluated on a flow.
    assert monomial((-1,{}),(1,{'B':1,'u':1,'j':-1}),(-1,{'B':-1})) == (F(1),{'u':1,'j':-1})
    assert monomial((-1,{}),(1,{'B':1,'u':1,'j':-1}),(1,{'j':1,'B':-1})) == (F(-1),{'u':1})
    assert monomial((1,{'delta':1,'j':1,'u1_plus_u2':-1,'ln9':-1}),
                    (1,{'u1_plus_u2':1,'ln9':1,'j':-1})) == (F(1),{'delta':1})
    # d_mu(rho^2 D d_mu c) has units 1/time for mu=mass/unit area.
    mu_dim,rho_dim,D_dim = (1,-2,0),(1,-3,0),(0,2,-1)
    assert tuple(2*rho_dim[i]+D_dim[i]-2*mu_dim[i] for i in range(3)) == (0,0,-1)
    assert F(p['proposed_delta'])/F(p['old_delta']) == F(1,2) and d['new_to_old_delta_ratio_exact'] == '1/2'
    assert d['fixed'] == p['fixed'] and d['recompute'] == p['recompute'] and d['forbidden_reuse'] == p['forbidden_reuse']
    old = read(out/'source/runs/reconnect-becker-common-k32/results.json')
    assert d['old_case_budgets_preserved'] == old['by_grid']['32']
    assert d['old_shifted_H_outcomes'] == {'32':True,'64':False,'128':False}
    case = d['proposed_case']
    assert case == p['proposed_case'] and case['candidate_columns'] == [32] and case['K'] == [8,16,32]
    assert case['max_new_graphs'] == sum(case['K']) == 56
    assert case['max_new_actual_paths'] == case['max_new_independent_paths'] == 3
    assert case['max_new_reference_solves'] == 3 and not case['already_executed']
    for key in ('new_delta_data_generated','new_candidate_h_control_completed','delta_zero_limit_claimed','fluid_orc_enabled','kappa_computed'):
        assert d[key] is False
    for key in ('new_graphs','new_candidate_paths','new_reference_PDE_solves','new_density_sources','new_software_checks'):
        assert d[key] == 0
    assert d['recorded_checks_total'] == 502 and d['stage3_physical_acceptance'] == 'incomplete'
    ex = read(out/'execution.json')
    assert ex['elapsed_seconds'] < 120 and ex['peak_RSS_MiB'] < 1024 and not ex['flow_geometry_or_PDE_functions_called']
    findings = {'passed':True,'source_files':len(p['sources']),
        'old_material_and_reference_arrays_independently_parsed':True,
        'coordinate_chain_rule_and_thickness_monomials_verified':True,'material_PDE_dimensions_verified':True,
        'source_excerpts_and_two_case_specific_traps_verified':True,
        'old_positive_budgets_and_H_outcomes_unchanged':True,'new_delta_flow_samples_or_solves':0,
        'planned_counts_verified':True,'new_software_checks':0,'recorded_checks_total':502,
        'scope':'source/old-input/planning audit and algebraic coefficient checks; no new flow validation or automated full theorem proof'}
    if write:
        with (out/'audit.json').open('x') as f:
            json.dump({'passed':True,'findings':findings,'elapsed_seconds':time.monotonic()-started},f,ensure_ascii=False,indent=2)
            f.write('\n')
    else:
        assert read(out/'audit.json')['findings'] == findings
    print(json.dumps(findings,ensure_ascii=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory',default=ROOT/'runs/reconnect-becker-delta-definition')
    parser.add_argument('--write',action='store_true')
    args = parser.parse_args()
    main(args.directory,args.write)
