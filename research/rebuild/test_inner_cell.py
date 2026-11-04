"""Continuous cell quadrature, flux matching and independent symmetries."""
import unittest
import numpy as np
from scipy.integrate import dblquad, quad
from inner_cell_problem import solve_inner, integrated_pair_weight, affine_weighted_residual
from inner_layer_study import gaussian_average, inner_average
from positive_kernel_1d import slope_over_radius


class InnerCellChecks(unittest.TestCase):
    def test_positive_cell_pair_integral_matches_independent_double_integral(self):
        d=.125
        for k in (1,7,16):
            independent=dblquad(lambda y,x:float(slope_over_radius(y-x,1.)),
                                -d/2,d/2,lambda x:k*d-d/2,lambda x:k*d+d/2,
                                epsabs=1e-13,epsrel=1e-11)[0]
            value=integrated_pair_weight(k,d)
            self.assertGreater(value,0.)
            self.assertAlmostEqual(value,independent,places=12)

    def test_continuous_affine_source_trace_mass_and_second_moment(self):
        beta=1/3
        self.assertAlmostEqual(float(affine_weighted_residual(0.,beta)),-27/16*beta,places=13)
        mass=2*quad(lambda x:float(affine_weighted_residual(x,beta)),0.,2.,epsabs=1e-13)[0]
        second=2*quad(lambda x:x*x*float(affine_weighted_residual(x,beta)),0.,2.,epsabs=1e-13)[0]
        self.assertLess(abs(mass),1e-13)
        self.assertAlmostEqual(second,12/55*beta,places=13)

    def test_unit_background_flux_and_stationary_corrector_constraints(self):
        ar,d=solve_inner(half_window=4.,spacing=1/16)
        self.assertLess(d['max_augmented_linear_residual'],1e-12)
        self.assertLess(d['zero_correction_flux_residual'],1e-12)
        self.assertLess(d['gauge_residual'],1e-12)
        self.assertGreater(d['kappa'],0.)
        self.assertGreater(d['minimum_pair_weight'],0.)
        # Continuous affine background has exactly F=1 across the interface.
        H=4/3;sL=-1/2;sR=-1.
        flux=-H*(sL+sR)*quad(lambda r:float(slope_over_radius(r,1.))*r*r,0.,2.,epsabs=1e-13)[0]
        self.assertAlmostEqual(flux,1.,places=13)

    def test_equal_densities_have_exactly_zero_layer_and_far_jump(self):
        ar,d=solve_inner(1.,1.,half_window=4.,spacing=1/16)
        np.testing.assert_array_equal(ar['corrector'],0.)
        self.assertEqual(d['kappa'],0.)

    def test_density_scaling_at_fixed_unit_flux(self):
        a,da=solve_inner(2.,1.,half_window=4.,spacing=1/16)
        b,db=solve_inner(6.,3.,half_window=4.,spacing=1/16)
        np.testing.assert_allclose(b['corrector'],a['corrector']/3,atol=1e-13,rtol=1e-11)
        self.assertAlmostEqual(db['kappa'],da['kappa']/3,places=13)

    def test_material_swap_and_reflection(self):
        a,da=solve_inner(2.,1.,half_window=4.,spacing=1/16)
        b,db=solve_inner(1.,2.,half_window=4.,spacing=1/16)
        np.testing.assert_allclose(b['corrector'],-a['corrector'][::-1],atol=1e-13,rtol=1e-11)
        self.assertAlmostEqual(db['kappa'],da['kappa'],places=13)

    def test_prediction_uses_cell_integrals_and_preserves_the_observable(self):
        faces=np.linspace(-1.,1.,129);D=.02;T=.1
        means=gaussian_average(faces,D,T)
        integral=quad(lambda x:np.exp(-x*x/(4*D*T)),-1.,1.,epsabs=1e-13)[0]
        self.assertAlmostEqual(float(np.diff(faces)@means),integral,places=13)
        ar,_=solve_inner(half_window=4.,spacing=1/16)
        target=np.linspace(-4.,4.,33)
        indices,average=inner_average(ar,target,region=2.)
        chosen=(ar['positions']>=-2.)&(ar['positions']<2.)
        expected=float(np.diff(ar['faces'])[chosen]@ar['localized_corrector'][chosen])
        self.assertAlmostEqual(float(np.diff(target)[indices]@average),expected,places=13)

    def test_invalid_or_unaligned_analysis_parameters_rejected(self):
        for args in ({'rho_left':0.},{'half_window':2.},{'spacing':.03},{'spacing':np.nan}):
            with self.assertRaises(ValueError):solve_inner(**args)


if __name__=='__main__':unittest.main(verbosity=2)
