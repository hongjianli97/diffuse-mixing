"""Seven necessary cache/integer-action checks; no saved-flow or old suite."""
from fractions import Fraction as F
from pathlib import Path
from unittest.mock import patch
import argparse
import inspect
import json
import unittest
import numpy as np

from cached_global_exchange import _build_product_cache as build,apply_cached_exchange as apply
from cached_global_exchange import coefficient_for_indices,floating_action,build_original_global_cache
from periodic_density_path import inverse_density_path
from positive_kernel_2d import slope_over_radius
import math


def fixture(N=4,h=.2):
    positions=[(F(2*c+1,2*N),F(2*r+1,2*N)) for r in range(N) for c in range(N)]
    rho=[F(1) if c<N//2 else F(4) for r in range(N) for c in range(N)]
    return dict(positions=positions,ids=list(range(N*N)),mass=[v/(N*N) for v in rho],rho=rho,D=.02,h=h,periods=[1,1])


def all_edges(data):
    n=len(data['ids'])
    for i in range(n):
        for j in range(i+1,n):
            delta=[b-a for a,b in zip(data['positions'][i],data['positions'][j])]
            delta=[v-1 if v>F(1,2) else v+1 if v<-F(1,2) else v for v in delta]
            r2=sum(v*v for v in delta)
            if r2<=(2*F(data['h']))**2:yield i,j,r2


class CachedExchangeChecks(unittest.TestCase):
    def test_all_small_grid_coefficients_equal_full_original_primitive(self):
        data=fixture();cache=build(**data)
        for i,j,r2 in all_edges(data):
            p=inverse_density_path(data['positions'],data['ids'],data['rho'],data['periods'],data['ids'][i],data['ids'][j])
            Bh=float(slope_over_radius(math.sqrt(float(r2)),data['h']))
            expected=2*F(1,16)**2*F(.02)/p.mean_inverse_density*F(Bh)
            self.assertEqual(coefficient_for_indices(cache,i,j),expected)
            self.assertEqual(coefficient_for_indices(cache,i,j),coefficient_for_indices(cache,j,i))

    def test_integer_action_equals_direct_fraction_inventory_and_energy(self):
        data=fixture();cache=build(**data)
        states={i:F(i*i-7,13) for i in cache.ids};result=apply(cache,states)
        flux=[F(0)]*16;energy=F(0);edges=0
        for i,j,r2 in all_edges(data):
            g=coefficient_for_indices(cache,i,j);difference=states[j]-states[i]
            flux[i]+=g*difference;flux[j]-=g*difference;energy+=g*difference*difference;edges+=1
        self.assertEqual(result.rates,tuple(v/m for v,m in zip(flux,cache.mass)))
        self.assertEqual(sum(m*r for m,r in zip(cache.mass,result.rates)),0)
        self.assertEqual(result.energy_pair_exact,energy)
        self.assertEqual(result.energy_row_exact,-energy)
        self.assertEqual(result.edge_count,edges)
        np.testing.assert_allclose(floating_action(cache,[states[i] for i in cache.ids]),list(map(float,result.rates)),rtol=1e-14,atol=1e-14)

    def test_internal_column_and_oblique_integral_value_reuse(self):
        data=fixture(8,.14);cache=build(**data)
        for i,j in [(2,4),(2,12),(2,60),(7,9),(7,57),(3,20)]:
            p=inverse_density_path(data['positions'],data['ids'],data['rho'],data['periods'],i,j)
            r2=sum(min(abs(a-b),1-abs(a-b))**2 for a,b in zip(data['positions'][i],data['positions'][j]))
            expected=2*F(1,64)**2*F(.02)/p.mean_inverse_density*F(float(slope_over_radius(math.sqrt(float(r2)),.14)))
            self.assertEqual(coefficient_for_indices(cache,i,j),expected)

    def test_fixed_ID_reordering_changes_no_integral_but_receipts_are_local(self):
        data=fixture();a=build(**data);data['ids']=list(reversed(data['ids']));b=build(**data)
        self.assertEqual(a.conductances,b.conductances)
        self.assertNotEqual(a.representative_records,b.representative_records)
        self.assertTrue(all(r['ownership_applies_only_to_representative'] for r in b.representative_records))
        for record in b.representative_records:
            i,j=record['representative_input_indices']
            self.assertEqual(record['path']['start_id'],data['ids'][i])
            self.assertEqual(record['path']['end_id'],data['ids'][j])

    def test_global_invalid_or_unsupported_input_rejected_before_primitive(self):
        changes=[('rho',-1),('mass',0),('rho',float('nan'))]
        for key,value in changes:
            data=fixture();data[key][-1]=value
            with patch('cached_global_exchange.localized_inverse_density_path',side_effect=AssertionError('must validate first')):
                with self.assertRaises(ValueError):build(**data)
        data=fixture();data['positions'][-1]=(F(1,3),F(1,3))
        with self.assertRaises(ValueError):build(**data)
        data=fixture();data['rho'][-1]=3
        with self.assertRaises(ValueError):build(**data)
        with self.assertRaises(ValueError):build_original_global_cache(**fixture())

    def test_constant_zero_D_and_exact_linearity_without_weight_selection(self):
        data=fixture();cache=build(**data);before=cache.conductances
        a={i:F(i,7) for i in cache.ids};b={i:F((i%3)-1,11) for i in cache.ids}
        ra,rb=apply(cache,a),apply(cache,b);combined=apply(cache,{i:2*a[i]-3*b[i] for i in cache.ids})
        self.assertEqual(combined.rates,tuple(2*x-3*y for x,y in zip(ra.rates,rb.rates)))
        self.assertTrue(all(r==0 for r in apply(cache,{i:F(3,8) for i in cache.ids}).rates))
        self.assertEqual(before,cache.conductances)
        data['D']=0
        with patch('cached_global_exchange.localized_inverse_density_path',side_effect=AssertionError('zero D path bypass')):zero=build(**data)
        self.assertTrue(all(r==0 for r in apply(zero,a).rates))
        self.assertNotIn('c0',inspect.signature(build).parameters)
        with self.assertRaises(ValueError):apply(cache,{0:F(1)})

    def test_input_and_coefficients_unchanged_when_state_order_changes(self):
        data=fixture();cache=build(**data)
        states={i:F(i+1,19) for i in cache.ids};reversed_state=dict(reversed(list(states.items())))
        self.assertEqual(apply(cache,states),apply(cache,reversed_state))
        self.assertEqual(cache.ids,tuple(data['ids']))
        self.assertEqual(cache.mass,tuple(data['mass']))
        self.assertEqual(cache.rho,tuple(data['rho']))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--json',required=True);args=parser.parse_args()
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(CachedExchangeChecks))
    with Path(args.json).open('x') as f:
        json.dump({'successful':result.wasSuccessful(),'checks_run':result.testsRun,'failures':len(result.failures),
            'errors':len(result.errors),'skipped':len(result.skipped)},f,indent=2);f.write('\n')
    raise SystemExit(not result.wasSuccessful())
