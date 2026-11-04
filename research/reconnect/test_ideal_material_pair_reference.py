"""Six necessary reference-integration checks; no original saved case here."""
from pathlib import Path
from unittest.mock import patch
import argparse
import json
import math
import unittest
import numpy as np

from ideal_material_pair_reference import tensor_cell_rule,cell_support_offsets,integrate_ideal_material_rows
from positive_kernel_2d import slope_over_radius


def fixture(N=8):
    x=np.array([((c+.5)/N,(r+.5)/N) for r in range(N) for c in range(N)])
    rho=np.where(x[:,0]<.5,1.,4.)
    return dict(N=N,ids=np.arange(N*N),mass=rho/N**2,cbar=(np.arange(N*N,dtype=float)**2+1)/100,
                centers=x,D=.02,h=.06,target_ids=[N//2-1])


def scalar_gamma(x,y,D):
    left=1 if x%1<.5 else 4;right=1 if y%1<.5 else 4
    if left==right:return left*D
    boundaries=[k/2 for k in range(-2,5) if min(x,y)<k/2<max(x,y)]
    if len(boundaries)!=1:raise AssertionError('independent short-path fixture must cross one contact')
    alpha=abs(boundaries[0]-x)/abs(y-x)
    return 1/(alpha/(left*D)+(1-alpha)/(right*D))


class IdealMaterialPairChecks(unittest.TestCase):
    def test_raw_tensor_area_and_quadratic_moments(self):
        N=8;rule=tensor_cell_rule(N,4);p=rule['local_node_offsets'];w=rule['weights_2D']
        self.assertTrue(np.all(w>0))
        self.assertAlmostEqual(math.fsum(map(float,w)),1/N**2,places=16)
        self.assertAlmostEqual(float(w@p[:,0]),0,places=18)
        self.assertAlmostEqual(float(w@(p[:,0]**2)),1/(12*N**4),places=18)
        self.assertAlmostEqual(float(w@(p[:,0]*p[:,1])),0,places=18)

    def test_full_small_row_equals_independent_scalar_four_dimensional_sum(self):
        data=fixture();rule,rows=integrate_ideal_material_rows(**data,order=2,target_block=3)
        row=rows[0];N=data['N'];i=row['target_index'];r,c=divmod(i,N);local=rule['local_node_offsets'];weights=rule['weights_2D']
        terms=[]
        for dx,dy in cell_support_offsets(N,data['h']):
            j=((r+dy)%N)*N+(c+dx)%N
            for p,wp in zip(local,weights):
                x=data['centers'][i]+p
                for q,wq in zip(local,weights):
                    y=data['centers'][i]+np.array([dx,dy])/N+q
                    radius=math.hypot(*(y-x));B=float(slope_over_radius(radius,data['h']))
                    terms.append(float(wp*wq)*scalar_gamma(float(x[0]),float(y[0]),data['D'])*B*(data['cbar'][j]-data['cbar'][i]))
        expected=2*math.fsum(terms)/data['mass'][i]
        self.assertAlmostEqual(row['rate'],expected,places=12)

    def test_material_support_includes_cells_outside_point_support_and_periodic_lifts(self):
        offsets=cell_support_offsets(128,.04)
        self.assertEqual(len(offsets),420)
        self.assertIn((11,1),offsets)
        self.assertGreater(math.hypot(11/128,1/128),.08)
        data=fixture();_,rows=integrate_ideal_material_rows(**data,order=2)
        row=rows[0];i=np.flatnonzero(np.all(row['neighbor_offsets']==[0,-1],axis=1))[0]
        self.assertEqual(row['neighbor_indices'][i],7*8+3)
        self.assertLess(row['neighbor_lifted_centers'][i,1],0)

    def test_shared_pair_mass_contribution_reverses_without_pair_inventory_repair(self):
        data=fixture();data['target_ids']=[3,4]
        _,rows=integrate_ideal_material_rows(**data,order=3)
        left,right=rows
        a=left['neighbor_signed_mass_rate_contributions'][np.flatnonzero(left['neighbor_ids']==4)[0]]
        b=right['neighbor_signed_mass_rate_contributions'][np.flatnonzero(right['neighbor_ids']==3)[0]]
        np.testing.assert_allclose(a,-b,rtol=1e-13,atol=1e-15)
        self.assertNotEqual(sum(row['raw_mass_rate_numerator'] for row in rows),0)

    def test_blocking_and_constant_state_without_weight_or_state_normalization(self):
        data=fixture();_,a=integrate_ideal_material_rows(**data,order=3,target_block=1)
        _,b=integrate_ideal_material_rows(**data,order=3,target_block=8)
        np.testing.assert_allclose(a[0]['target_node_inner_integrals'],b[0]['target_node_inner_integrals'],rtol=1e-14,atol=1e-15)
        before=data['cbar'].copy();data['cbar']=np.full(64,3/8)
        _,constant=integrate_ideal_material_rows(**data,order=3)
        self.assertEqual(constant[0]['rate'],0)
        np.testing.assert_array_equal(data['cbar'],np.full(64,3/8))
        self.assertFalse(np.array_equal(before,data['cbar'])) # Unit fixture only, never the original c0.

    def test_invalid_global_inputs_and_zero_D_bypass(self):
        for key,value in [('mass',0),('cbar',float('nan'))]:
            data=fixture();data[key][-1]=value
            with patch('ideal_material_pair_reference.path_gamma',side_effect=AssertionError('must validate first')):
                with self.assertRaises(ValueError):integrate_ideal_material_rows(**data,order=2)
        data=fixture();data['centers'][-1,0]+=1e-6
        with self.assertRaises(ValueError):integrate_ideal_material_rows(**data,order=2)
        data=fixture();data['D']=0
        with patch('ideal_material_pair_reference.path_gamma',side_effect=AssertionError('D=0 must bypass')):
            _,rows=integrate_ideal_material_rows(**data,order=2)
        self.assertEqual(rows[0]['rate'],0)
        self.assertEqual(rows[0]['kernel_node_pairs_evaluated'],0)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--json',required=True);args=parser.parse_args()
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(IdealMaterialPairChecks))
    with Path(args.json).open('x') as f:
        json.dump({'successful':result.wasSuccessful(),'checks_run':result.testsRun,'failures':len(result.failures),
            'errors':len(result.errors),'skipped':len(result.skipped)},f,indent=2);f.write('\n')
    raise SystemExit(not result.wasSuccessful())
