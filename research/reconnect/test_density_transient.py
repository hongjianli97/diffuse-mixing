"""Independent PDE, projection and exponential-action controls for one case."""
import unittest
from unittest.mock import patch
import numpy as np
from scipy.linalg import expm
from scipy.sparse import csr_matrix
from trajectory_chain import laplacian
from sparse_exchange_audit import sparse_laplacian
from spectral_density_reference import periodic_second_derivative,spectral_system,reference_path,project_component_amplitude
from sparse_density_evolution import evolve_saved
from smooth_density_probe import analytic_reference,point_fields


class DensityTransientChecks(unittest.TestCase):
    def test_raw_periodic_D2_has_correct_fourier_modes_and_exact_symmetry(self):
        M=32;x=np.arange(M)/M;A=periodic_second_derivative(M)
        np.testing.assert_array_equal(A,A.T)
        np.testing.assert_allclose(A@np.ones(M),0.,atol=3e-12)
        for mode in [1,2,3]:
            f=np.cos(2*np.pi*mode*x);np.testing.assert_allclose(A@f,-(2*np.pi*mode)**2*f,atol=2e-11,rtol=2e-13)

    def test_selfadjoint_transform_matches_full_physical_drift_PDE(self):
        s=spectral_system(64,.25,.02);_,_,lap,drift,full=point_fields(s['x'],np.zeros(64),.25,.02)
        point_rate=-(s['H']@s['v0'])/s['root_rho']
        np.testing.assert_allclose(point_rate,full,atol=5e-12,rtol=0.)
        np.testing.assert_array_equal(s['H'],s['H'].T)
        self.assertGreater(np.max(abs(full-lap)),.01)

    def test_uniform_density_reference_matches_exact_diffusing_mode(self):
        r=analytic_reference(16,0.,.02);times=np.array([0.,.05,.1]);states,amps,audit=reference_path(32,16,0.,.02,times,r['mass'])
        expected=.5+np.exp(-8*np.pi**2*.02*times[:,None])*(r['state']-.5)
        np.testing.assert_allclose(states,expected,atol=3e-14,rtol=0.)

    def test_fourier_cell_projection_recovers_exact_rho_weighted_initial_state(self):
        N=16;M=64;eps=.25;x=np.arange(M)/M;rho=1+eps*np.cos(2*np.pi*x);q=rho*.2*np.cos(2*np.pi*x)
        reference=analytic_reference(N,eps,.02);state,amp,audit=project_component_amplitude(q,N,reference['mass'])
        np.testing.assert_allclose(state,reference['state'],atol=4e-16,rtol=0.)
        self.assertLess(audit['maximum_imaginary_cell_integral'],1e-16)

    def test_variable_density_weighted_mode_inventory_obeys_exact_decay(self):
        r=analytic_reference(16,.25,.02);states,amp,audit=reference_path(64,16,.25,.02,[0.,.05,.1],r['mass'])
        for control in audit['projection_controls']:self.assertLess(control['x_mode_inventory_absolute_difference'],2e-14)
        for state in states:self.assertLess(abs(float(r['mass']@state)-.5),1e-15)

    def test_sparse_actual_c0_path_matches_dense_exponential_on_small_graph(self):
        m=np.array([1.,2.,3.]);c=np.array([0.,.2,1.]);L=laplacian(3,np.array([[0,1],[1,2]]),[.8,.4]);times=np.array([0.,.05,.1])
        actual,arrays,audit=evolve_saved(m,csr_matrix(L),c,times)
        expected=np.array([expm(-t*L/m[:,None])@c for t in times]);np.testing.assert_allclose(actual,expected,atol=2e-15,rtol=0.)
        self.assertLess(audit['full_vs_two_half_max_difference'],2e-15);self.assertLess(audit['raw_mass_symmetric_vs_Q_endpoint_max_difference'],2e-15)
        self.assertFalse(audit['full_propagation_matrix_constructed'])

    def test_zero_diffusion_and_constant_states_are_preserved_without_repair(self):
        m=np.arange(1.,9.);times=np.array([0.,.05,.1]);p=np.column_stack((np.arange(7),np.arange(1,8)));L=sparse_laplacian(8,p,np.ones(7))
        constant=np.full(8,.37);states,_,_=evolve_saved(m,L,constant,times);np.testing.assert_allclose(states,.37,atol=4e-16)
        c=np.linspace(.1,.9,8);zero,_,_=evolve_saved(m,L*0,c,times);np.testing.assert_array_equal(zero,np.broadcast_to(c,(3,8)))
        r=analytic_reference(16,.25,0.);reference,_,_=reference_path(32,16,.25,0.,times,r['mass']);np.testing.assert_allclose(reference,np.broadcast_to(r['state'],reference.shape),atol=5e-16,rtol=0.)

    def test_sparse_evolution_does_not_request_a_dense_propagator(self):
        n=32;m=np.linspace(1.,2.,n);p=np.column_stack((np.arange(n-1),np.arange(1,n)));L=sparse_laplacian(n,p,np.ones(n-1))
        with patch.object(csr_matrix,'toarray',side_effect=AssertionError('dense conversion forbidden')):
            states,_,audit=evolve_saved(m,L,np.linspace(.1,.9,n),[0.,.05,.1])
        self.assertEqual(states.shape,(3,n));self.assertFalse(audit['full_propagation_matrix_constructed'])


if __name__=='__main__':unittest.main(verbosity=2)
