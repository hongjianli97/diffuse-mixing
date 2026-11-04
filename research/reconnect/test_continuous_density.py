"""Independent polar, Cartesian, full 2D and uniform-symbol reference checks."""
import unittest
import numpy as np
from scipy.special import roots_legendre
from continuous_density_kernel import PolarRule,input_extension,x_coefficient,reference_rates,split_budget
from positive_kernel_2d import slope_over_radius
from kernel2d_study import continuum_decay
from smooth_density_probe import analytic_reference


class ContinuousDensityChecks(unittest.TestCase):
    def test_uniform_limit_matches_independent_Bessel_symbol_and_mass_average(self):
        N=8;h=.06;D=.02;rule=PolarRule(24,48,8);r=reference_rates(N,0.,D,h,rule)
        ref=analytic_reference(N,0.,D);decay=continuum_decay(h,D,2*np.pi*np.sqrt(2),64,'W')
        expected=-decay*(ref['state']-.5)
        np.testing.assert_allclose(r['physical_rate'],expected,atol=8e-15)
        np.testing.assert_allclose(r['input_closure_rate'],expected,atol=8e-15)

    def test_fixed_input_extension_reproduces_analytic_saved_mass_averages(self):
        N=8;eps=.25;ref=analytic_reference(N,eps,.02);a=ref['labels'];rho,average,amplitude,weight=input_extension(a[:,0],N,eps)
        np.testing.assert_allclose(.5+amplitude*np.cos(2*np.pi*a[:,1]),ref['state'],atol=2e-16)
        np.testing.assert_allclose(rho,ref['rho_point'],atol=0.);np.testing.assert_allclose(average,ref['rho_average'],atol=2e-16)
        np.testing.assert_allclose(weight,ref['mass']/ref['rho_point']*N*N,atol=3e-16)

    def test_reduced_action_matches_full_two_dimensional_polar_integral(self):
        x=.137;y=.211;N=64;eps=.25;D=.02;h=.06;rule=PolarRule(32,64,8)
        z,w=roots_legendre(rule.radial);r=h*(z+1);theta=2*np.pi*np.arange(rule.angular)/rule.angular
        u=r[:,None]*np.cos(theta);v=r[:,None]*np.sin(theta);rho0=1+eps*np.cos(2*np.pi*x);rho1=1+eps*np.cos(2*np.pi*(x+u))
        H=2*D*rho0*rho1/(rho0+rho1);weights=(h*w*r*slope_over_radius(r,h))[:,None]*(2*np.pi/rule.angular)
        for closure in ['physical','saved_inputs']:
            if closure=='physical':
                c0=.5+.2*np.cos(2*np.pi*x)*np.cos(2*np.pi*y);c1=.5+.2*np.cos(2*np.pi*(x+u))*np.cos(2*np.pi*(y+v));weight=1.
            else:
                _,_,f0,_=input_extension(np.array(x),N,eps);_,_,f1,weight=input_extension(x+u,N,eps)
                c0=.5+f0*np.cos(2*np.pi*y);c1=.5+f1*np.cos(2*np.pi*(y+v))
            full=2/rho0*np.sum(weights*weight*H*(c1-c0))
            reduced=x_coefficient(np.array([x]),N,eps,D,h,rule,closure)[0]*np.cos(2*np.pi*y)
            self.assertAlmostEqual(full,reduced,places=13)

    def test_polar_reference_matches_independent_cartesian_disk_quadrature(self):
        x=.137;N=64;eps=.25;D=.02;h=.06;R=2*h;z,w=roots_legendre(192)
        u=R*z;limit=np.sqrt(R*R-u*u);v=limit[:,None]*z[None,:]
        rho0=1+eps*np.cos(2*np.pi*x);rho1=1+eps*np.cos(2*np.pi*(x+u))
        H=2*D*rho0*rho1/(rho0+rho1)
        delta=.2*np.cos(2*np.pi*(x+u[:,None]))*np.cos(2*np.pi*v)-.2*np.cos(2*np.pi*x)
        radius=np.sqrt(u[:,None]**2+v*v);weights=(R*w*limit)[:,None]*w[None,:]
        cartesian=2/rho0*np.sum(weights*H[:,None]*slope_over_radius(radius,h)*delta)
        polar=x_coefficient(np.array([x]),N,eps,D,h,PolarRule(64,128,16),'physical')[0]
        self.assertAlmostEqual(cartesian,polar,places=10)

    def test_periodic_reflection_D_linearity_and_zero_limit(self):
        x=np.array([.03,.137,.271,.431]);rule=PolarRule(24,48,8)
        for closure in ['physical','saved_inputs']:
            base=x_coefficient(x,64,.25,.02,.06,rule,closure)
            np.testing.assert_allclose(x_coefficient(x+1,64,.25,.02,.06,rule,closure),base,atol=2e-14)
            np.testing.assert_allclose(x_coefficient(-x,64,.25,.02,.06,rule,closure),base,atol=2e-14)
            np.testing.assert_allclose(x_coefficient(x,64,.25,.04,.06,rule,closure),2*base,atol=0.,rtol=0.)
            np.testing.assert_array_equal(x_coefficient(x,64,.25,0.,.06,rule,closure),0.)

    def test_signed_budget_retains_cancellation_and_Gram_cross_terms(self):
        m=np.array([1.,2.,3.]);ref=np.array([.2,-.1,.3]);physical=ref+np.array([.1,.2,-.1])
        inputs=physical+np.array([-.08,-.15,.09]);discrete=inputs+np.array([.001,-.002,.003])
        components,b=split_budget(m,ref,discrete,physical,inputs);total=discrete-ref
        np.testing.assert_allclose(components.sum(axis=0),total,atol=1e-16)
        self.assertAlmostEqual(np.sum(b['normalized_Gram']),b['total_relative_error']**2)
        self.assertAlmostEqual(sum(b['signed_projection_onto_total_error']),1.)
        self.assertLess(b['signed_projection_onto_total_error'][1],0.)
        self.assertLess(b['total_relative_error'],sum([b['finite_width_relative_norm'],b['averaged_input_point_density_closure_relative_norm'],b['node_quadrature_relative_norm']]))

    def test_two_predeclared_rules_agree_for_variable_density_reference(self):
        low=reference_rates(8,.25,.02,.06,PolarRule(32,64,8));high=reference_rates(8,.25,.02,.06,PolarRule(64,128,16))
        m=analytic_reference(8,.25,.02)['mass']
        for key in ['physical_rate','input_closure_rate']:
            np.testing.assert_allclose(low[key],high[key],atol=2e-13)
            self.assertLess(abs(m@high[key]),2e-15)


if __name__=='__main__':unittest.main(verbosity=2)
