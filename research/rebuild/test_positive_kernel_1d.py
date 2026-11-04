"""Physical normalization, graph algebra and Neumann kernel consistency checks."""
import unittest
import numpy as np
from scipy.integrate import quad
from scipy.linalg import expm
from positive_kernel_1d import kernel_value,slope_over_radius,build_exchange


def uniform(n=64):
    faces=np.linspace(-1.,1.,n+1);x=(faces[:-1]+faces[1:])/2;V=np.diff(faces)
    return faces,x,V


class PositiveKernelChecks(unittest.TestCase):
    def test_continuous_1D_normalization_and_Brookshaw_second_moment(self):
        for h in (.03,.17):
            mass=2*quad(lambda r:float(kernel_value(r,h)),0,2*h,epsabs=1e-13)[0]
            moment=2*quad(lambda r:float(slope_over_radius(r,h))*r*r,0,2*h,epsabs=1e-13)[0]
            self.assertAlmostEqual(mass,1.,places=13);self.assertAlmostEqual(moment,1.,places=13)

    def test_finite_origin_support_and_independent_derivative(self):
        h=.2
        self.assertAlmostEqual(float(slope_over_radius(0.,h)),(63/16)/h**3,places=12)
        np.testing.assert_array_equal(slope_over_radius(np.array([2*h,3*h]),h),0.)
        for r in (.03,.15,.3):
            eps=1e-7
            independent=-(kernel_value(r+eps,h)-kernel_value(r-eps,h))/(2*eps*r)
            self.assertAlmostEqual(float(independent/slope_over_radius(r,h)),1.,places=8)

    def test_shared_graph_constant_mass_sign_energy_and_propagator(self):
        faces,x,V=uniform(24);rho=np.where(x<0,2.,1.);m=rho*V;h=2*V
        a,d=build_exchange(x,m,V,rho,.02,h,(-1.,1.));G=a['G'];L=a['L'];Q=-L/m[:,None]
        np.testing.assert_array_equal(G,G.T);np.testing.assert_array_equal(np.diag(G),0.)
        self.assertGreaterEqual(G.min(),0.);self.assertEqual(d['components'],1)
        self.assertGreater(d['cross_contact_edges'],0)
        np.testing.assert_allclose(Q.sum(axis=1),0.,atol=1e-12)
        np.testing.assert_allclose(m@Q,0.,atol=1e-13)
        c=.5+.3*np.sin(np.arange(len(x)));v=c-m@c/m.sum()
        self.assertLessEqual(float(v@(m*(Q@v))),1e-13)
        P=expm(.1*Q);np.testing.assert_allclose(P.sum(axis=1),1.,atol=2e-13)
        np.testing.assert_allclose(m@P,m,atol=2e-13);self.assertGreaterEqual(P.min(),0.)

    def test_symmetric_variable_support_pair_is_present_and_permutation_equivariant(self):
        x=np.array([-.1,.1]);V=np.array([.1,.1]);rho=np.ones(2);h=np.array([.02,.2])
        a,d=build_exchange(x,V,V,rho,.02,h,(-1.,1.))
        # Only the larger individual h sees its neighbor, but their symmetric
        # support contains the pair. No one-sided neighbor enumeration is used.
        self.assertGreater(abs(x[1]-x[0]),2*h[0]);self.assertLess(abs(x[1]-x[0]),h.sum())
        self.assertGreater(a['G'][0,1],0.)
        b,e=build_exchange(x[::-1],V[::-1],V[::-1],rho[::-1],.02,h[::-1],(-1.,1.))
        np.testing.assert_allclose(a['G'],b['G'][::-1,::-1],rtol=2e-14)

    def test_even_reflection_recovers_uniform_discrete_cosine_eigenvalue(self):
        faces,x,V=uniform(128);h=2*V;D=.02;k=np.pi/2
        a,d=build_exchange(x,V,V,np.ones_like(x),D,h,(-1.,1.))
        c=np.cos(k*(x+1))*np.sinc(V/4)
        offsets=np.arange(1,5)*V[0]
        rate=4*D*V[0]*np.sum(slope_over_radius(offsets,h[0])*(1-np.cos(k*offsets)))
        np.testing.assert_allclose(-(a['L']@c)/V,-rate*c,atol=2e-12,rtol=0.)
        self.assertGreater(d['reflected_exchange_sum_left'],0.)
        self.assertGreater(d['reflected_exchange_sum_right'],0.)

    def test_mass_density_scaling_D_linearity_and_D0(self):
        faces,x,V=uniform(32);rho=np.where(x<0,2.,1.);m=rho*V;h=2*V
        a,d=build_exchange(x,m,V,rho,.02,h,(-1.,1.))
        b,e=build_exchange(x,13*m,V,13*rho,.02,h,(-1.,1.))
        np.testing.assert_allclose(a['L']/m[:,None],b['L']/(13*m[:,None]),rtol=2e-13,atol=1e-12)
        c,f=build_exchange(x,m,V,rho,.04,h,(-1.,1.));np.testing.assert_allclose(c['G'],2*a['G'],rtol=2e-13)
        z,f=build_exchange(x,m,V,rho,0.,h,(-1.,1.));np.testing.assert_array_equal(z['L'],0.)

    def test_fixed_ratio_lattice_limit_is_exactly_the_reported_biased_value(self):
        from fractions import Fraction
        moment=Fraction(0)
        for j in (1,2,3):
            q=Fraction(j,2)
            B=Fraction(63,32)/8*(2+5*q)*(1-q/2)**5
            moment+=2*B*j*j
        self.assertEqual(moment,Fraction(130725,131072))
        for dx in (.1,.03,1.):
            r=np.arange(1,5)*dx
            numerical=2*dx*np.sum(slope_over_radius(r,2*dx)*r*r)
            self.assertAlmostEqual(numerical,float(moment),places=13)
        self.assertNotEqual(moment,Fraction(1))

    def test_invalid_or_inconsistent_inputs_are_not_silently_repaired(self):
        faces,x,V=uniform(32);rho=np.ones_like(x);h=2*V
        with self.assertRaises(ValueError):build_exchange(x,1.1*V,V,rho,.02,h,(-1.,1.))
        with self.assertRaises(ValueError):build_exchange(x,V,V,rho,.02,np.ones_like(h)*2,(-1.,1.))
        with self.assertRaises(ValueError):build_exchange(x,V,V,rho,-.02,h,(-1.,1.))
        with self.assertRaises(ValueError):build_exchange(x,V,V,rho,np.ones_like(x),h,(-1.,1.))
        with self.assertRaises(ValueError):slope_over_radius(np.nan,.1)
        with self.assertRaises(ValueError):kernel_value(0.,0.)


if __name__=='__main__':unittest.main(verbosity=2)
