"""Seven new adapter checks; no historical test suite or saved-flow case."""
from fractions import Fraction as F
from inspect import signature
from pathlib import Path
from unittest.mock import patch
import argparse
import json
import unittest

from reconstructed_path_rows import build_reconstructed_path_rows as build, apply_reconstructed_rows as apply
from periodic_density_path import inverse_density_path


def fixture():
    return dict(positions=[(F(1, 8), F(1, 8)), (F(3, 8), F(1, 8)),
                           (F(1, 4), F(1, 8)), (F(7, 8), F(1, 8)), (F(3, 4), F(3, 4))],
                ids=[10, 20, 3, 40, 50], mass=[F(1, 10)]*5,
                rho=[1, 4, 2, 3, 8], D=.02, h=F(1, 7), periods=[1, 1], target_ids=[10, 20])


class ReconstructedRowChecks(unittest.TestCase):
    def test_all_support_and_localization_match_full_exact_primitive(self):
        data = fixture(); result = build(**data)
        pairs = set()
        for target in data['target_ids']:
            i = data['ids'].index(target)
            for j, identity in enumerate(data['ids']):
                dx = [min(abs(a-b), 1-abs(a-b)) for a,b in zip(data['positions'][i],data['positions'][j])]
                if identity != target and sum(v*v for v in dx) <= (2*data['h'])**2:
                    pairs.add(tuple(sorted((target, identity))))
        self.assertEqual(pairs, {(e.left_id,e.right_id) for e in result.edges})
        for e in result.edges:
            expected = inverse_density_path(data['positions'],data['ids'],data['rho'],data['periods'],e.left_id,e.right_id)
            self.assertEqual(e.path, expected)

    def test_interior_site_changes_path_not_endpoint_only(self):
        result = build(**fixture())
        edge = next(e for e in result.edges if (e.left_id,e.right_id)==(10,20))
        self.assertEqual(edge.path.mean_inverse_density,F(9,16))
        self.assertNotEqual(edge.path.mean_inverse_density,(F(1)+F(1,4))/2)
        self.assertIn(3, {v.source_id for v in edge.path.intervals})

    def test_shared_edge_and_state_application_conserve_exchange_exactly(self):
        result = build(**fixture()); values = {identity:F(identity,100) for identity in result.ids}
        rates = apply(result, values)
        shared = [(i,e) for i,e in enumerate(result.edges) if (e.left_id,e.right_id)==(10,20)]
        self.assertEqual(len(shared),1)
        index,e = shared[0]
        self.assertTrue(all(any(k==index for _,k in row.neighbors) for row in result.rows))
        self.assertEqual(e.conductance*(values[20]-values[10])+e.conductance*(values[10]-values[20]),0)
        self.assertEqual(apply(result,{i:F(7,9) for i in result.ids}),{10:0,20:0})
        self.assertNotEqual(sum(row.mass*rates[row.target_id] for row in result.rows),0)
        self.assertNotIn('c0',signature(build).parameters)
        self.assertNotIn('reference_rate',signature(build).parameters)

    def test_permutation_and_periodic_translation_preserve_shared_weights(self):
        data = fixture(); a = build(**data)
        order = list(range(len(data['ids'])))[::-1]
        changed = {**data,**{k:[data[k][i] for i in order] for k in ['positions','ids','mass','rho']}}
        b = build(**changed)
        translated = {**data,'positions':[((x+F(1,2))%1,(y+F(1,2))%1) for x,y in data['positions']]}
        c = build(**translated)
        self.assertEqual([(e.left_id,e.right_id,e.conductance) for e in a.edges],
                         [(e.left_id,e.right_id,e.conductance) for e in b.edges])
        self.assertEqual([e.conductance for e in a.edges],[e.conductance for e in c.edges])

    def test_global_invalid_input_not_hidden_by_localization(self):
        for value in [-1,0,float('nan'),float('inf')]:
            data=fixture(); data['rho'][-1]=value
            with self.assertRaises((ValueError,TypeError)):
                build(**data)
        data=fixture(); data['mass'][-1]=0
        with self.assertRaises(ValueError): build(**data)
        data=fixture(); data['ids'][-1]=data['ids'][0]
        with self.assertRaises(ValueError): build(**data)

    def test_zero_D_and_support_boundary_have_zero_raw_exchange(self):
        data=fixture(); data['D']=0
        with patch('reconstructed_path_rows.localized_inverse_density_path',side_effect=AssertionError('D=0 must bypass path')):
            result=build(**data)
        self.assertTrue(all(e.path is None and e.conductance==0 for e in result.edges))
        data=dict(positions=[(0,0),(F(1,4),0)],ids=[1,2],mass=[1,1],rho=[1,1],D=.02,h=F(1,8),periods=[1,1],target_ids=[1])
        edge=build(**data).edges[0]
        self.assertEqual(edge.distance_squared,F(1,16))
        self.assertEqual(edge.kernel_B,0)
        self.assertEqual(edge.conductance,0)

    def test_fraction_fallback_and_constant_density_constitutive_rule(self):
        data=fixture(); data['rho']=[3]*5
        result=build(**data)
        for e in result.edges:
            i,j=data['ids'].index(e.left_id),data['ids'].index(e.right_id)
            expected=2*result.quadrature_area[i]*result.quadrature_area[j]*F(.02)*3*F(e.kernel_B)
            self.assertEqual(e.path.mean_inverse_density,F(1,3))
            self.assertEqual(e.conductance,expected)
        data['positions'][-1]=(F(3,4)+F(1,2**80),F(3,4))
        fallback=build(**data)
        self.assertFalse(fallback.exact_integer_broad_phase)
        self.assertEqual([e.conductance for e in result.edges],[e.conductance for e in fallback.edges])


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--json',required=True)
    args=parser.parse_args()
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(ReconstructedRowChecks))
    with Path(args.json).open('x') as f:
        json.dump({'successful':result.wasSuccessful(),'checks_run':result.testsRun,
                   'failures':len(result.failures),'errors':len(result.errors),'skipped':len(result.skipped)},f,indent=2)
        f.write('\n')
    raise SystemExit(not result.wasSuccessful())
