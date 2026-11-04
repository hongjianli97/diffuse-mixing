"""Independent coefficient, flux and direct-2D checks for an oracle only."""
import unittest
import numpy as np
from oracle_path_reference import path_gamma,continuous_point_rate,material_means
from contact_continuous import gauss,normal_intervals
from contact_frozen_probe import point_fields,contact_inputs,AMPLITUDE
from positive_kernel_2d import slope_over_radius


def resistance_primitive(x,D=.02):
    x=np.asarray(x);period=np.floor(x);wrapped=x-period
    return period*(.5/D+.5/(4*D))+point_fields(wrapped,0.,D)['xi']*(.5/D+.5/(4*D))


class OraclePathChecks(unittest.TestCase):
    def test_raw_positive_path_coefficient_reciprocity_and_same_phase_values(self):
        cases=[(.49,.04),(.53,-.04),(.02,-.04),(.98,.04),(.5,-.01),(.7,.01)]
        for x,s in cases:
            K=float(path_gamma(x,s));reverse=float(path_gamma((x+s)%1.,-s))
            self.assertGreaterEqual(K,.02);self.assertLessEqual(K,.08)
            self.assertLess(abs(K-reverse),3e-16)
        self.assertLess(abs(float(path_gamma(.49,.04))-8/175),3e-16)
        self.assertEqual(float(path_gamma(.1,.02)),.02)
        self.assertEqual(float(path_gamma(.7,0.)),.08)

    def test_split_coefficient_matches_independent_integral_primitive(self):
        for x,s in [(.497,.02),(.03,-.05),(.98,.03),(.75,.02),(.5,-.03)]:
            exact=s/float(resistance_primitive(x+s)-resistance_primitive(x))
            self.assertLess(abs(float(path_gamma(x,s))-exact),2e-15)

    def test_local_constant_normal_flux_pair_identity_without_coefficient_fit(self):
        flux=.001
        for x,s in [(.497,.02),(.503,-.03),(.48,.01),(.52,.01)]:
            dc=flux*float(resistance_primitive(x+s)-resistance_primitive(x))
            self.assertLess(abs(float(path_gamma(x,s))*dc-flux*s),3e-17)

    def test_transverse_mode_reduction_matches_direct_two_dimensional_integral(self):
        h=.04;D=.02;nodes,w=gauss(64)
        for x in [.003,.497,.503,.75]:
            y=.13;center=point_fields(x,y,D);total=0.
            for left,right in normal_intervals(x,h):
                s=(left+right)/2+(right-left)/2*nodes;ws=(right-left)/2*w
                width=np.sqrt((2*h-abs(s))*(2*h+abs(s)))
                for sign in [-1.,1.]:
                    z=sign*width[:,None]*(nodes+1)/2
                    other=point_fields(((x+s)%1.)[:,None],y+z,D)
                    B=slope_over_radius(np.sqrt(s[:,None]**2+z*z),h)
                    integrand=path_gamma(x,s,D)[:,None]*B*(other['c']-center['c'])
                    total+=float(ws@np.sum(integrand*(width[:,None]*w/2),axis=1))
            direct=2*total/float(center['rho'])
            self.assertLess(abs(direct-continuous_point_rate(x,y,h,D,64,64)),3e-12)

    def test_original_mass_denominator_inventory_and_positive_physical_dissipation(self):
        d,a=contact_inputs(8);q={'cell_x_order':4,'normal_displacement_order':16,'transverse_order':16}
        first=material_means(8,a['mass'],a['material_label_centers'],.04,.02,q)
        changed=a['mass'].copy();changed[0]*=1.01
        second=material_means(8,changed,a['material_label_centers'],.04,.02,q)
        np.testing.assert_array_equal(first['component_mass_rate_numerator'],second['component_mass_rate_numerator'])
        np.testing.assert_array_equal(second['rate'],first['component_mass_rate_numerator']/changed)
        self.assertLess(abs(float(first['component_mass_rate_numerator'].sum())),3e-17)
        phi=point_fields(first['outer_x'],0.)['phase']
        chi=-AMPLITUDE**2*float(first['outer_weight']@(np.cos(phi)*first['outer_normal_response']))
        self.assertGreater(chi,0.)

    def test_invalid_oracle_paths_are_rejected_without_repair(self):
        for x,s,D in [(0.,.5,.02),(np.nan,.01,.02),(0.,.01,0.),(0.,np.inf,.02)]:
            with self.assertRaises(ValueError):path_gamma(x,s,D)


if __name__=='__main__':unittest.main(verbosity=2)
