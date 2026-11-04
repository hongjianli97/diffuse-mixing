"""Independent exact-integral and information-loss checks for material unions."""
from fractions import Fraction as F
import argparse,json,unittest
import numpy as np
from material_union import group_sum,block_index,union_labels,oracle_density,define_union,observe_path


class Checks(unittest.TestCase):
    def test_exact_union_mass_and_component_integral(self):
        side,ratio=2,2;m=(np.arange(16)+1)/32;c=np.arange(16)/32;index=block_index(side,ratio)
        mp,cp,ref=define_union(m,c,np.array([c,.5*c]),side,ratio)
        for j in range(4):
            exact_m=sum((F(float(v)) for v,i in zip(m,index) if i==j),F(0))
            exact_mc=sum((F(float(a))*F(float(b)) for a,b,i in zip(m,c,index) if i==j),F(0))
            self.assertEqual(mp[j],float(exact_m));self.assertEqual(cp[j],float(exact_mc/exact_m))
        np.testing.assert_array_equal(ref[0],cp);np.testing.assert_array_equal(ref[1],cp/2)
        with self.assertRaises(ValueError):group_sum(np.ones(15),2,2)

    def test_direct_factor4_preserves_original_denominator(self):
        m=np.full(16,1/16);mp=np.array([np.nextafter(1.,np.inf)])
        p=observe_path(np.ones((1,16)),m,mp,1,4)
        self.assertEqual(p[0,0],1/mp[0]);self.assertLess(p[0,0],1.)
        c=np.arange(64)/128;mf=np.full(64,1/64)
        intermediate=np.full(16,1/16);intermediate[0]=np.nextafter(intermediate[0],np.inf)
        target=group_sum(intermediate,2,2)
        first=observe_path(np.array([c]),mf,intermediate,4,2)
        chained=observe_path(first,intermediate,target,2,2)
        direct=observe_path(np.array([c]),mf,target,2,4)
        np.testing.assert_allclose(chained,direct,atol=2e-16,rtol=0)

    def test_seed_centers_and_point_density_distinct_from_cell_average(self):
        axis=(np.arange(8)+.5)/8;a=np.stack(np.meshgrid(axis,axis,indexing='xy'),axis=-1).reshape(-1,2)
        seed=union_labels(a);target=(np.arange(4)+.5)/4
        np.testing.assert_array_equal(seed,np.stack(np.meshgrid(target,target,indexing='xy'),axis=-1).reshape(-1,2))
        point=oracle_density(seed);average=1+.25*np.sinc(1/4)*np.cos(2*np.pi*seed[:,0])
        self.assertGreater(float(np.max(abs(point-average))),.01)
        with self.assertRaises(ValueError):union_labels(a[::-1])

    def test_coarse_observation_can_hide_native_error(self):
        row,col=np.indices((8,8));error=.25*((-1.)**(row+col)).ravel()
        m=np.full(64,1/64);mp=np.full(4,1/4);reference=np.full(64,.5);state=reference+error
        projected=observe_path(np.array([state]),m,mp,2,4)[0]
        np.testing.assert_array_equal(projected,np.full(4,.5))
        self.assertEqual(float(m@(error*error)),1/16)
        self.assertGreater(float(m@((state-.5)**2)),float(mp@((projected-.5)**2)))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--json',required=True);a=p.parse_args()
    r=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Checks))
    with open(a.json,'x') as f:json.dump({'successful':r.wasSuccessful(),'checks_run':r.testsRun,'failures':len(r.failures),'errors':len(r.errors),'skipped':len(r.skipped)},f,indent=2);f.write('\n')
    raise SystemExit(0 if r.wasSuccessful() else 1)
