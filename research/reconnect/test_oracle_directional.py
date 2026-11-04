"""Direct physical integrands and independent signed-Gram algebra checks."""
import unittest
import numpy as np
from oracle_directional import directional_response,observe_integrals,gram_budget
from oracle_path_reference import path_gamma,normal_response
from contact_continuous import gauss,normal_intervals
from contact_frozen_probe import point_fields,contact_inputs,AMPLITUDE
from positive_kernel_2d import slope_over_radius


class OracleDirectionalChecks(unittest.TestCase):
    def test_normal_first_components_match_two_independent_direct_integrands(self):
        h=.04;D=.02;nodes,w=gauss(64)
        for x in [.003,.497,.503,.75]:
            y=.13;center=point_fields(x,y,D)['c'];direct=np.zeros(2)
            for left,right in normal_intervals(x,h):
                s=(left+right)/2+(right-left)/2*nodes;ws=(right-left)/2*w
                width=np.sqrt((2*h-abs(s))*(2*h+abs(s)))
                frozen=point_fields((x+s)%1.,y,D)['c'][:,None]
                for sign in [-1.,1.]:
                    z=sign*width[:,None]*(nodes+1)/2;other=point_fields(((x+s)%1.)[:,None],y+z,D)['c']
                    B=slope_over_radius(np.sqrt(s[:,None]**2+z*z),h);K=path_gamma(x,s,D)[:,None]
                    for axis,dc in [(0,frozen-center),(1,other-frozen)]:
                        direct[axis]+=float(ws@np.sum(K*B*dc*(width[:,None]*w/2),axis=1))
            direct*=2/float(point_fields(x,y,D)['rho'])
            reduced=2*AMPLITUDE*np.cos(2*np.pi*y)/float(point_fields(x,y,D)['rho'])*np.array(directional_response(x,h,D,64,64))
            np.testing.assert_allclose(reduced,direct,atol=3e-12,rtol=0.)

    def test_component_sum_recovers_unchanged_total_response(self):
        for x in [.003,.497,.503,.75]:
            a,b=directional_response(x,.04,.02,64,64)
            self.assertLess(abs(a+b-normal_response(x,.04,.02,64,64)),5e-13)

    def test_material_observation_keeps_saved_mass_denominator_and_inventory(self):
        d,a=contact_inputs(8);axis=(np.arange(8)+.5)/8;y=np.sinc(1/8)*np.cos(2*np.pi*axis)/8
        I=np.array([np.sin(2*np.pi*axis),np.cos(4*np.pi*axis)])
        rates,numerator=observe_integrals(a['mass'],y,I);changed=a['mass'].copy();changed[0]*=1.01
        after,again=observe_integrals(changed,y,I)
        np.testing.assert_array_equal(again,numerator);np.testing.assert_array_equal(after,numerator/changed[None,:])
        for component in numerator:self.assertLess(abs(float(component.sum())),3e-17)

    def test_signed_Gram_identity_matches_exact_nonorthogonal_toy_values(self):
        m=np.array([1.,3.]);ref=np.array([2.,-1.]);errors=np.array([[1.,2.],[-2.,-1.],[.25,-.5]])
        r=gram_budget(m,errors,errors.sum(axis=0),ref,np.array([True,False]))
        expected=np.array([[13/7,-8/7,-11/28],[-8/7,1.,1/7],[-11/28,1/7,13/112]])
        np.testing.assert_allclose(r['all']['gram_over_global_reference_norm_squared'],expected,atol=3e-16,rtol=0.)
        self.assertLess(abs(r['all']['saved_total_error_norm']**2-3/16),3e-16)
        self.assertLess(abs(r['all']['signed_twice_cross_terms']['normal_transverse']+16/7),3e-16)
        self.assertLess(abs(r['all']['squared_norm_identity_residual']),3e-15)
        self.assertLess(abs(r['near_contact']['saved_total_error_norm']**2-9/112),3e-16)

    def test_invalid_mass_or_zero_reference_norm_is_rejected(self):
        with self.assertRaises(ValueError):observe_integrals(np.zeros(4),np.ones(2),np.ones((2,2)))
        with self.assertRaises(ValueError):gram_budget(np.ones(2),np.ones((3,2)),np.ones(2),np.zeros(2),np.ones(2,dtype=bool))


if __name__=='__main__':unittest.main(verbosity=2)
