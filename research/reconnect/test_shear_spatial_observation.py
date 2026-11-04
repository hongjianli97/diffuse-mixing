"""Nested full-PDE material rates and invariant common observations."""
import unittest
import numpy as np
from smooth_density_probe import analytic_reference
from shear_probe import shear_face_reference,shear_gauss_reference
from variable_parent_observation import observe_parent
from shear_spatial_observation import compare_parent_rates


class ShearSpatialChecks(unittest.TestCase):
    def test_full_shear_reference_commutes_with_material_nesting(self):
        coarse=analytic_reference(8,.25,.02);fine=analytic_reference(16,.25,.02)
        old=shear_face_reference(8,.25,.25,.02,coarse['labels'],coarse['mass'],16)['rate']
        new=shear_gauss_reference(16,.25,.25,.02,fine['labels'],fine['mass'],16)['rate']
        np.testing.assert_allclose(observe_parent(new,fine['mass'],coarse['mass']),old,atol=4e-15,rtol=2e-15)

    def test_signed_rate_inventory_and_negative_values_are_retained(self):
        coarse=analytic_reference(8,.25,.02);fine=analytic_reference(16,.25,.02)
        rate=3*np.sin(np.arange(256)*.037)-1.
        observed=observe_parent(rate,fine['mass'],coarse['mass'])
        self.assertLess(observed.min(),0.);self.assertGreater(observed.max(),0.)
        self.assertLess(abs(coarse['mass']@observed-fine['mass']@rate),3e-16)

    def test_common_error_comparison_is_invariant_to_physical_mass_scale(self):
        coarse=analytic_reference(8,.25,.02);fine=analytic_reference(16,.25,.02)
        ref=coarse['rate'];old=ref+.01*np.cos(np.arange(64));rate=fine['rate']+.003*np.sin(np.arange(256))
        observed,a=compare_parent_rates(coarse['mass'],ref,old,fine['mass'],rate)
        scaled,b=compare_parent_rates(3*coarse['mass'],ref,old,3*fine['mass'],rate)
        np.testing.assert_allclose(observed,scaled,atol=2e-16,rtol=0.)
        for key in ['coarse_error_common_cells','fine_error_common_cells','absolute_error_reduction']:
            np.testing.assert_allclose(a[key],b[key],atol=3e-16,rtol=0.)


if __name__=='__main__':unittest.main(verbosity=2)
