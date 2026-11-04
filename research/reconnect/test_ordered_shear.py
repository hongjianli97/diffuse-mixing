"""Small integration controls for unchanged shear producer/kernel/propagator."""
import unittest
import numpy as np
from shear_probe import make_shear_samples,shear_gauss_reference
from positive_kernel_2d import build_periodic_exchange
from ordered_compression import interval_grid,frozen_exponential_step


class OrderedShearChecks(unittest.TestCase):
    def fixture(self):
        N=12;axis=(np.arange(N)+.5)/N
        labels=np.stack(np.meshgrid(axis,axis,indexing='xy'),axis=-1).reshape(-1,2)
        rho=1+.25*np.cos(2*np.pi*labels[:,0])
        m=(1+.25*np.sinc(1/N)*np.cos(2*np.pi*labels[:,0]))/N**2
        c0=shear_gauss_reference(N,.25,0.,.02,labels,m,8)['state']
        return labels,np.arange(N*N),m,c0,rho

    def evolve(self,inputs,D=.02,offset=None):
        labels,ids,m,c0,rho=inputs;_,midpoints,times=interval_grid(4,.05)
        data,_,_=make_shear_samples(labels,ids,m,c0,rho,times)
        if offset is not None:
            from trajectory_chain import TrajectorySamples,SI_UNITS
            data=TrajectorySamples(times,(data.positions+offset)%1.,ids,m,c0,SI_UNITS,rho=data.rho)
        c=c0.copy();states=[c]
        for j in range(len(midpoints)):
            L,_,_=build_periodic_exchange(data,2*j+1,D,.12,[1.,1.])
            c,_,_=frozen_exponential_step(m,L,c,.05/4);states.append(c)
        return np.array(states)

    def test_moving_point_cloud_zero_D_preserves_original_mass_averages(self):
        inputs=self.fixture();labels,ids,m,c0,rho=inputs
        point=.5+.2*np.cos(2*np.pi*labels[:,0])*np.cos(2*np.pi*labels[:,1])
        self.assertGreater(np.max(abs(c0-point)),1e-3)
        np.testing.assert_array_equal(self.evolve(inputs,D=0.),np.broadcast_to(c0,(5,len(c0))))

    def test_chronological_shear_is_invariant_under_identity_permutation(self):
        inputs=self.fixture();p=np.random.default_rng(410).permutation(len(inputs[2]))
        permuted=tuple(v[p] for v in inputs)
        np.testing.assert_allclose(self.evolve(permuted),self.evolve(inputs)[:,p],atol=3e-14,rtol=0.)

    def test_rigid_periodic_translation_preserves_shear_scalar_history(self):
        inputs=self.fixture()
        np.testing.assert_allclose(self.evolve(inputs,offset=np.array([.313,.217])),self.evolve(inputs),atol=3e-14,rtol=0.)

    def test_raw_shear_preserves_component_mass_bounds_and_dissipates(self):
        inputs=self.fixture();m=inputs[2];c0=inputs[3];states=self.evolve(inputs)
        self.assertLess(np.max(abs((states-c0)@m)),3e-14)
        self.assertGreaterEqual(states.min(),c0.min()-3e-14)
        self.assertLessEqual(states.max(),c0.max()+3e-14)
        mean=m@c0/m.sum();energy=((states-mean)**2)@m
        self.assertTrue(np.all(np.diff(energy)<0.))


if __name__=='__main__':unittest.main(verbosity=2)
