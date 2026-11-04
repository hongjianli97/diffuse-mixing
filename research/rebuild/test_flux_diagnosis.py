"""Independent manufactured checks for flux identities and reference sampling."""
import unittest
import numpy as np
from scipy.integrate import quad
from becker import Becker,initial
from experiment import conductance
from flux_diagnosis import sampled_reference_flux,tendency,split


class FluxChecks(unittest.TestCase):
    def test_average_flux_identity_for_transient_nonlinear_field(self):
        # Variable k and nonlinear c; J is not constant, so this exercises the
        # general weighted identity rather than the earlier steady test.
        class Flow:
            def rho_mu(self,mu,t):return 1/np.sqrt(1+mu)
        edges=np.array([0.,.13,.41,1.,1.7]);m=np.diff(edges)
        primitive=lambda x:.3*x+.2*x*x+.1*x**4/4
        cbar=np.diff(primitive(edges))/m
        k=lambda x:1/(1+x)
        # primitive differentiates to c=.3+.4x+.1x^3, so c_mu=.4+.3x^2.
        J=lambda x:-k(x)*(.4+.3*x*x)
        g=conductance(Flow(),edges,0.,1.,'mass_average_resistance')
        predicted=-g*np.diff(cbar)
        for i in range(len(g)):
            a,b,d=edges[i:i+3]
            left=lambda x:(x-a)/(b-a)/k(x)
            right=lambda x:(d-x)/(d-b)/k(x)
            denominator=quad(left,a,b)[0]+quad(right,b,d)[0]
            numerator=quad(lambda x:left(x)*J(x),a,b)[0]+quad(lambda x:right(x)*J(x),b,d)[0]
            self.assertAlmostEqual(predicted[i],numerator/denominator,places=13)

    def test_exact_initial_becker_flux_identity(self):
        f=Becker();e=f.mu(np.linspace(-1,1,33));m=np.diff(e);cbar=initial(e)
        g=conductance(f,e,0.,.02,'mass_average_resistance')
        predicted=-g*np.diff(cbar)
        # J/k=-c_mu, so independent analytic concentration derivatives can
        # be integrated directly against the triangular cell-average weights.
        deriv=lambda x:.25*np.pi/f.total*np.sin(np.pi*x/f.total)
        for i in (0,14,18,22,30):
            a,b,d=e[i:i+3]
            numerator=quad(lambda x:(x-a)/m[i]*deriv(x),a,b,epsabs=1e-13)[0]
            numerator+=quad(lambda x:(d-x)/m[i+1]*deriv(x),b,d,epsabs=1e-13)[0]
            self.assertAlmostEqual(predicted[i],g[i]*numerator,places=13)

    def test_reference_flux_sampling_against_analytic_heat_field(self):
        class Uniform:
            def rho_mu(self,mu,t):return np.ones_like(mu)
        # For cell averages of cos(pi*mu/L), centered derivative of averages
        # has sinc(pi*h/(2L))**2 amplitude: second-order, not exact.
        L=2.;target=np.array([.25,.75,1.,1.5]);errors=[]
        for n in (128,256,512):
            e=np.linspace(0,L,n+1);h=L/n
            c=np.diff(np.sin(np.pi*e/L))*L/(np.pi*h)
            j=sampled_reference_flux(Uniform(),e,c,0.,.02,target)
            exact=.02*np.pi/L*np.sin(np.pi*target/L)
            errors.append(np.linalg.norm(j-exact))
        np.testing.assert_allclose(np.array(errors[:-1])/errors[1:],4.,rtol=.001)

    def test_flux_divergence_orientation_and_mass_balance(self):
        m=np.array([.1,.2,.7]);flux=np.array([1.,2.])
        r=tendency(flux,m)
        np.testing.assert_allclose(r,[-10.,-5.,2/.7])
        self.assertAlmostEqual(float(m@r),0.,places=14)
        with self.assertRaises(ValueError):tendency([1.],m)

    def test_bias_scaling_and_state_splits_against_direct_flux(self):
        f=Becker();e=f.mu(np.linspace(-1,1,33));c0=initial(e)
        jr=f.rho_mu(e[1:-1],0.)**2*.02*.25*np.pi/f.total*np.sin(np.pi*e[1:-1]/f.total)
        perturbed={method:c0+.001*np.sin(np.arange(len(c0))) for method in
                   ('ordinary_volume_center','ordinary_mass_center','mass_average_resistance')}
        rows,arrays=split(f,e,c0,jr,0.,.02,perturbed)
        for row in rows:
            self.assertLess(row['closure_split_max_residual'],1e-15)
            self.assertLess(row['state_split_max_residual'],1e-15)
            method=row['method'];g=arrays[method+'_g']
            direct=-g*np.diff(perturbed[method])-jr
            np.testing.assert_allclose(arrays[method+'_total_terminal_flux_error'],direct,atol=1e-16)


if __name__=='__main__':unittest.main(verbosity=2)
