"""Independent disk/Fourier controls for the continuous kernel reference."""
import unittest
import numpy as np
from scipy.special import j0
from contact_continuous import gauss,transverse_moments,normal_intervals,continuous_point_rate,material_means
from contact_frozen_probe import contact_inputs,point_fields,WAVE
from positive_kernel_2d import slope_over_radius


class ContactContinuousChecks(unittest.TestCase):
    def normal_grid(self,order,h):
        z,w=gauss(order)
        return np.r_[-h+h*z,h+h*z],np.r_[h*w,h*w]

    def test_raw_kernel_second_moment_is_one_without_normalization(self):
        s,w=self.normal_grid(64,.04);b0,bk=transverse_moments(s,.04,64)
        self.assertLess(abs(float(w@(s*s*b0))-1.),2e-11)
        self.assertTrue(np.all(b0>=0));self.assertTrue(np.all(bk<=b0))

    def test_uniform_fourier_eigenvalue_matches_independent_polar_Bessel_integral(self):
        h=.04;D=.02;s,w=self.normal_grid(64,h);b0,bk=transverse_moments(s,h,64)
        reduced=2*D*float(w@(np.cos(WAVE*s)*bk-b0))
        z,v=gauss(128);r=h*(z+1)
        polar=4*np.pi*D*float((h*v)@(slope_over_radius(r,h)*r*(j0(np.sqrt(2)*WAVE*r)-1)))
        self.assertLess(abs(reduced-polar),2e-11)

    def test_contact_fourier_reduction_matches_direct_two_dimensional_integral(self):
        h=.04;D=.02;nodes,w=gauss(64)
        for x in [.003,.497,.503,.75]:
            y=.13;center=point_fields(x,y,D);total=0.
            for left,right in normal_intervals(x,h):
                s=(left+right)/2+(right-left)/2*nodes;ws=(right-left)/2*w
                width=np.sqrt((2*h-abs(s))*(2*h+abs(s)))
                for sign in [-1.,1.]:
                    z=sign*width[:,None]*(nodes+1)/2
                    other=point_fields(((x+s)%1.)[:,None],y+z,D)
                    gamma=float(center['rho'])*D;other_gamma=other['rho']*D
                    H=2*gamma*other_gamma/(gamma+other_gamma)
                    B=slope_over_radius(np.sqrt(s[:,None]**2+z*z),h)
                    integrand=H*B*(other['c']-center['c'])
                    total+=float(ws@np.sum(integrand*(width[:,None]*w/2),axis=1))
            direct=2*total/float(center['rho'])
            self.assertLess(abs(direct-continuous_point_rate(x,y,h,D,64,64)),3e-12)

    def test_observation_uses_original_mass_without_rescaling_component_integrals(self):
        data,a=contact_inputs(8);q={'cell_x_order':4,'normal_displacement_order':16,'transverse_order':16}
        first=material_means(8,a['mass'],a['material_label_centers'],.04,.02,q)
        changed=a['mass'].copy();changed[0]*=1.01
        second=material_means(8,changed,a['material_label_centers'],.04,.02,q)
        np.testing.assert_array_equal(first['component_mass_rate_numerator'],second['component_mass_rate_numerator'])
        np.testing.assert_array_equal(second['rate'],first['component_mass_rate_numerator']/changed)
        self.assertLess(abs(float(first['component_mass_rate_numerator'].sum())),3e-17)

    def test_invalid_support_or_observation_inputs_are_rejected(self):
        for s,h,order in [([.09],.04,32),([0.],0.,32),([0.],.04,1)]:
            with self.assertRaises(ValueError):transverse_moments(s,h,order)
        with self.assertRaises(ValueError):normal_intervals(-.01,.04)


if __name__=='__main__':unittest.main(verbosity=2)
