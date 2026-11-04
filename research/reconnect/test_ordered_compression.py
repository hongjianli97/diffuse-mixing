"""Small algebra fixtures only; physical case is run once with time control."""
import unittest
import numpy as np
from scipy.linalg import expm
from scipy.sparse import csr_matrix
from ordered_compression import interval_grid,frozen_exponential_step,dissipation


class OrderedCompressionChecks(unittest.TestCase):
    def fixture(self):
        return np.array([1.,2.,3.]),csr_matrix([[1.,-1.,0.],[-1.,3.,-2.],[0.,-2.,2.]]),np.array([.1,.4,.8])

    def test_unequal_mass_action_matches_small_dense_exponential(self):
        m,L,c=self.fixture();end,half,check=frozen_exponential_step(m,L,c,.1)
        exact=expm(-.1*L.toarray()/m[:,None])@c
        np.testing.assert_allclose(end,exact,atol=3e-16,rtol=0.)
        self.assertLess(np.max(abs(check['two_half_minus_path'])),4e-16)

    def test_raw_bounds_mass_and_energy_dissipation(self):
        m,L,c=self.fixture();end,half,_=frozen_exponential_step(m,L,c,.1)
        self.assertGreaterEqual(end.min(),c.min());self.assertLessEqual(end.max(),c.max())
        self.assertLess(abs(m@(end-c)),2e-15)
        self.assertLess(m@(end*end),m@(c*c));self.assertGreater(dissipation(L,end),0.)

    def test_zero_diffusion_and_constant_state_controls(self):
        m,L,c=self.fixture();zero,_,_=frozen_exponential_step(m,0*L,c,.1)
        np.testing.assert_array_equal(zero,c)
        constant,_,_=frozen_exponential_step(m,L,np.full(3,.4),.1)
        np.testing.assert_allclose(constant,.4,atol=2e-16,rtol=0.)

    def test_noncommuting_graphs_have_chronological_left_action(self):
        m,L,c=self.fixture()
        A=csr_matrix([[1.,-1.,0.],[-1.,1.,0.],[0.,0.,0.]])
        B=csr_matrix([[0.,0.,0.],[0.,2.,-2.],[0.,-2.,2.]])
        first,_,_=frozen_exponential_step(m,A,c,.1);ordered,_,_=frozen_exponential_step(m,B,first,.1)
        first,_,_=frozen_exponential_step(m,B,c,.1);reversed_state,_,_=frozen_exponential_step(m,A,first,.1)
        exact=expm(-.1*B.toarray()/m[:,None])@(expm(-.1*A.toarray()/m[:,None])@c)
        np.testing.assert_allclose(ordered,exact,atol=3e-16,rtol=0.)
        self.assertGreater(np.max(abs(ordered-reversed_state)),.002)

    def test_identity_permutation_preserves_actual_scalar_action(self):
        m,L,c=self.fixture();p=np.array([2,0,1]);a,_,_=frozen_exponential_step(m,L,c,.1)
        b,_,_=frozen_exponential_step(m[p],L[p][:,p],c[p],.1)
        np.testing.assert_allclose(b,a[p],atol=2e-16,rtol=0.)

    def test_four_eight_interval_midpoint_sampling_is_fixed(self):
        all_midpoints=[]
        for K in [4,8]:
            boundaries,midpoints,samples=interval_grid(K,.05)
            self.assertEqual(boundaries[K//2],.025);self.assertEqual(boundaries[-1],.05)
            np.testing.assert_array_equal(midpoints,samples[1::2]);self.assertTrue(np.all(np.diff(samples)>0))
            all_midpoints.extend(midpoints)
        self.assertEqual(len(set(all_midpoints)),12)

    def test_constant_generator_temporal_partition_gives_same_endpoint(self):
        m,L,c=self.fixture();ends=[]
        for K in [4,8]:
            state=c.copy()
            for _ in range(K):state,_,_=frozen_exponential_step(m,L,state,.05/K)
            ends.append(state)
        np.testing.assert_allclose(ends[0],ends[1],atol=1e-15,rtol=0.)

    def test_invalid_inputs_and_unexplained_time_grid_are_rejected(self):
        m,L,c=self.fixture()
        for mass,A,dt in [([1.,0.,3.],L,.1),(m,L.toarray(),.1),(m,L,0.)]:
            with self.assertRaises(ValueError):frozen_exponential_step(mass,A,c,dt)
        with self.assertRaises(ValueError):interval_grid(3,.05)


if __name__=='__main__':unittest.main(verbosity=2)
