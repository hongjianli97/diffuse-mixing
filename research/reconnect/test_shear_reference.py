"""Full-PDE, conservative energy and mass-average spectral reference controls."""
import unittest
import numpy as np
from shear_reference import ShearReferenceOperator,project_mass_average,reference_path
from shear_probe import shear_point_fields
from smooth_density_probe import analytic_reference


class ShearReferenceChecks(unittest.TestCase):
    def test_full_conservative_rhs_matches_analytic_initial_PDE_at_multiple_times(self):
        op=ShearReferenceOperator(15)
        for t in [0.,.025,.05]:
            expected=shear_point_fields(op.a,op.b,.25,.25*t/.05,.02)[2]['rate']
            actual=op.rhs(t,op.initial_scalar().ravel()).reshape(15,15)
            np.testing.assert_allclose(actual,expected,atol=2e-13,rtol=2e-13)

    def test_weighted_inventory_adjoint_symmetry_and_energy_identity(self):
        op=ShearReferenceOperator(15);rng=np.random.default_rng(17);u=rng.normal(size=(15,15));v=rng.normal(size=(15,15))
        Qu=op.rhs(.025,u.ravel()).reshape(15,15);Qv=op.rhs(.025,v.ravel()).reshape(15,15)
        self.assertLess(abs(np.mean(op.rho*Qu)),3e-14)
        self.assertLess(abs(np.mean(op.rho*u*Qv)-np.mean(op.rho*v*Qu)),1e-12)
        self.assertLess(abs(np.mean(op.rho*u*Qu)+op.dissipation(.025,u)),1e-12)
        self.assertGreater(op.dissipation(.025,u),0.)

    def test_exact_initial_Fourier_cell_integrals_match_saved_mass_average_formula(self):
        op=ShearReferenceOperator(15)
        for N in [8,16]:
            original=analytic_reference(N,.25,.02)
            state,imag=project_mass_average(op.initial_scalar(),op.rho,N,original['mass'])
            np.testing.assert_allclose(state,original['state'],atol=1e-15,rtol=0.)
            self.assertLess(np.max(abs(imag)),3e-16)

    def test_projection_preserves_raw_component_inventory(self):
        op=ShearReferenceOperator(15);m=analytic_reference(16,.25,.02)['mass']
        field=.5+.1*np.sin(2*np.pi*op.a)*np.cos(4*np.pi*op.b)+.03*np.sin(6*np.pi*op.a)
        state,imag=project_mass_average(field,op.rho,16,m)
        self.assertLess(abs(m@state-np.mean(op.rho*field)),3e-16)

    def test_zero_D_and_constant_scalar_controls_under_changing_shear(self):
        zero,audit=reference_path(9,.005,[0.,.01,.02],D=0.,T=.02)
        np.testing.assert_array_equal(zero['point_snapshots'][0],zero['point_snapshots'][-1])
        constant,audit=reference_path(9,.005,[0.,.01,.02],T=.02,initial=np.full((9,9),.37))
        np.testing.assert_allclose(constant['point_snapshots'],.37,atol=2e-15,rtol=0.)

    def test_uniform_zero_shear_mode_matches_analytic_diffusion_solution(self):
        path,audit=reference_path(9,.05/8,[0.,.025,.05],epsilon=0.,shear_end=0.)
        c0=path['point_snapshots'][0]
        for t,c in zip(path['snapshot_time'],path['point_snapshots']):
            exact=.5+np.exp(-8*np.pi**2*.02*t)*(c0-.5)
            np.testing.assert_allclose(c,exact,atol=3e-14,rtol=0.)

    def test_invalid_reference_grid_mass_and_time_controls_rejected(self):
        for M,eps,D in [(8,.25,.02),(9,1.,.02),(9,.25,-.1)]:
            with self.assertRaises(ValueError):ShearReferenceOperator(M,eps,D)
        op=ShearReferenceOperator(9)
        with self.assertRaises(ValueError):project_mass_average(op.initial_scalar(),op.rho,8,np.zeros(64))
        with self.assertRaises(ValueError):reference_path(9,0.,[0.,.025,.05])


if __name__=='__main__':unittest.main(verbosity=2)
