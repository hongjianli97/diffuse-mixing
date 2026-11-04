"""Independent geometry and analytic raw moments for the one-row control."""
import unittest
import numpy as np
from ideal_row_response import direct_material_phase_means,aggregate_row


class IdealRowChecks(unittest.TestCase):
    def test_direct_projected_quadratic_phase_means_at_signed_displacements(self):
        ell=1/128;center=63/256
        for s in [-3*ell,-1.3*ell,-.4*ell,0.,.2*ell,1.7*ell,2*ell]:
            mean_jump,mean_delta=direct_material_phase_means(s,ell,center)
            k=np.floor(s/ell);theta=s/ell-k
            self.assertLess(abs(mean_jump-s),3e-16)
            expected=center*s+(s*s+ell**2*theta*(1-theta))/2
            self.assertLess(abs(mean_delta-expected),3e-16)

    def test_split_row_recovers_unchanged_analytic_kernel_moments(self):
        a,r=aggregate_row(1/128,.04,.02,1.,1/128**2,63/256,32,32)
        self.assertEqual(r['normal_panels'],22)
        self.assertLess(abs(r['raw_second_moment']-1),2e-11)
        self.assertLess(abs(r['raw_zero_moment_relative_difference_from_exact']),2e-11)
        self.assertTrue(np.all(a['normal_weights']>0))

    def test_direct_aggregate_material_row_agrees_with_variance_identity(self):
        a,r=aggregate_row(1/128,.04,.02,1.,1/128**2,63/256,32,32)
        self.assertLess(abs(r['raw_direct_minus_raw_variance_per_s']),5e-14)
        self.assertLess(abs(r['raw_variance_mass_rate_numerator_kg_per_s']/r['original_mass_denominator_kg']-r['raw_variance_aggregate_rate_per_s']),1e-17)
        self.assertGreater(r['raw_direct_aggregate_rate_per_s'],.02)
        self.assertGreater(r['eta'],6890625/3023656976384)
        self.assertLess(r['eta'],5625/131072)

    def test_invalid_original_mass_or_geometric_scale_is_rejected(self):
        with self.assertRaises(ValueError):aggregate_row(1/128,.04,.02,1.,0.,63/256,32,32)
        with self.assertRaises(ValueError):aggregate_row(0.,.04,.02,1.,1/128**2,63/256,32,32)
        with self.assertRaises(ValueError):direct_material_phase_means(float('nan'),1/128,63/256)


if __name__=='__main__':unittest.main(verbosity=2)
