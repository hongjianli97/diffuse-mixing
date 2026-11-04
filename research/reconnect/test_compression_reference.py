"""Continuum modes, time order, full PDE and unchanged-mass observations."""
import unittest
import numpy as np
from compression_reference import material_columns,material_laplacian,project_to_saved_cells,reference_path
from smooth_density_probe import analytic_reference
from compression_probe import analytic_material_rate


class CompressionReferenceChecks(unittest.TestCase):
    def test_exact_mass_average_initial_projection_on_saved_cells(self):
        original=analytic_reference(8,.25,.02)
        for M in [32,64]:
            a,m,B=material_columns(M,.25)
            state,child=project_to_saved_cells(B,m,0.,.02,8,original['mass'])
            np.testing.assert_allclose(state,original['state'],atol=4e-16,rtol=0.)
            np.testing.assert_allclose(child,original['mass'][:8]*8,atol=3e-17,rtol=0.)

    def test_shared_reference_faces_conserve_inventory_and_constants(self):
        a,m,B=material_columns(32,.25);L=material_laplacian(32,.25,.125,.02)
        np.testing.assert_allclose(L@np.ones(32),0.,atol=2e-16,rtol=0.)
        np.testing.assert_array_equal((L-L.T).data,[])
        self.assertLess(abs(np.sum(L@B)),8e-16)

    def test_uniform_static_mode_has_known_CN_amplification(self):
        M=32;steps=32;T=.05;D=.02;ref=analytic_reference(8,0.,D)
        arrays,audit=reference_path(M,steps,8,ref['mass'],0.,D,0.,T)
        discrete_decay=4*D*M*M*np.sin(np.pi/M)**2
        factor=((1-discrete_decay*T/(2*steps))/(1+discrete_decay*T/(2*steps)))**steps
        np.testing.assert_allclose(arrays['B_snapshots'][-1],arrays['initial_B']*factor,atol=4e-15,rtol=0.)

    def test_full_material_PDE_initial_rate_converges_at_second_order(self):
        errors=[]
        for N in [32,64]:
            ref=analytic_reference(N,.25,.02);a,m,B=material_columns(N,.25)
            Bdot=-(material_laplacian(N,.25,.125,.02)@B)/m
            expected,_=analytic_material_rate(N,.25,.125,.02,ref['mass'],ref['state'],ref['labels'])
            Y=np.sinc(1/N)*np.cos(2*np.pi*a)
            rate=.2*np.outer(Y,Bdot-.02*(2*np.pi)**2*B).ravel()
            errors.append(np.sqrt(ref['mass']@((rate-expected)**2)))
        self.assertGreater(errors[0]/errors[1],3.9);self.assertLess(errors[0]/errors[1],4.1)

    def test_nonautonomous_time_control_has_second_order_and_raw_energy_balance(self):
        m=analytic_reference(8,.25,.02)['mass'];states=[]
        for steps in [8,16,32]:
            arrays,audit=reference_path(32,steps,8,m)
            states.append(arrays['projected_states'][-1])
            self.assertLess(audit['B_weighted_inventory_max_change'],2e-15)
            self.assertLess(audit['CN_energy_balance_max_abs_residual'],3e-16)
            self.assertLess(audit['B_energy_max_increase'],0.)
        ratio=np.linalg.norm(states[0]-states[1])/np.linalg.norm(states[1]-states[2])
        self.assertGreater(ratio,3.9);self.assertLess(ratio,4.1)

    def test_zero_D_preserves_initial_scalar_during_prescribed_compression(self):
        r=analytic_reference(8,.25,0.);a,d=reference_path(32,8,8,r['mass'],D=0.)
        np.testing.assert_array_equal(a['projected_states'][0],a['projected_states'][-1])
        np.testing.assert_allclose(a['projected_states'][0],r['state'],atol=4e-16,rtol=0.)

    def test_projection_reuses_denominator_without_parent_mass_repair(self):
        r=analytic_reference(8,.25,.02);a,m,B=material_columns(32,.25)
        original,_=project_to_saved_cells(B,m,.05,.02,8,r['mass'])
        changed,_=project_to_saved_cells(B,m,.05,.02,8,r['mass']*1.01)
        np.testing.assert_allclose(changed,original/1.01,atol=2e-16,rtol=0.)

    def test_reference_rejects_nonphysical_map_and_unmatched_observations(self):
        for alpha,D in [(1.,.02),(.25,-.1)]:
            with self.assertRaises(ValueError):material_laplacian(32,.25,alpha,D)
        a,m,B=material_columns(32,.25)
        with self.assertRaises(ValueError):project_to_saved_cells(B,m,.05,.02,7,np.ones(49))
        with self.assertRaises(ValueError):reference_path(32,3,8,np.ones(64))


if __name__=='__main__':unittest.main(verbosity=2)
