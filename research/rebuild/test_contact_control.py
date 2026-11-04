"""Independent checks of the fixed compatible two-mode control."""
import unittest
import numpy as np
from scipy.integrate import quad
from contact_control import TwoModeContact
from contact import mesh


class ContactControlChecks(unittest.TestCase):
    def test_two_mode_average_matches_quadrature_and_has_nonzero_interface_time_derivative(self):
        f=TwoModeContact();faces=mesh(64,'cut');m=f.mass(faces);t=.1
        actual=f.cell_average(faces,t);expected=[]
        for a,b in zip(faces[:-1],faces[1:]):
            points=[a,0.,b] if a<0<b else [a,b];mass=integral=0.
            for l,r in zip(points[:-1],points[1:]):
                rho=f.rho_left if (l+r)/2<0 else f.rho_right
                mass+=rho*(r-l);integral+=quad(lambda x:rho*f.scalar(x,t),l,r,epsabs=1e-13)[0]
            expected.append(integral/mass)
        np.testing.assert_allclose(actual,expected,rtol=3e-14)
        self.assertAlmostEqual(float(m@actual),.5*f.total_mass,places=13)
        eps=1e-5;ct=(f.scalar(0.,t+eps)-f.scalar(0.,t-eps))/(2*eps)
        exact=f.even_amplitude*f.D*(np.pi/f.half_length)**2*np.exp(-f.D*(np.pi/f.half_length)**2*t)
        self.assertAlmostEqual(float(ct),float(exact),places=10)
        self.assertGreater(abs(ct),.01)

    def test_two_mode_one_sided_flux_transmission_and_bulk_pde(self):
        f=TwoModeContact();t=.1;h=1e-4;eps=1e-5;c=f.scalar(0.,t)
        dl=(3*c-4*f.scalar(-h,t)+f.scalar(-2*h,t))/(2*h)
        dr=(-3*c+4*f.scalar(h,t)-f.scalar(2*h,t))/(2*h)
        self.assertAlmostEqual(float(-f.rho_left*f.D*dl),float(-f.rho_right*f.D*dr),delta=2e-10)
        for a in (-.4,.4):
            ct=(f.scalar(a,t+eps)-f.scalar(a,t-eps))/(2*eps)
            cxx=(f.scalar(a+h,t)-2*f.scalar(a,t)+f.scalar(a-h,t))/h**2
            self.assertAlmostEqual(float(ct),float(f.D*cxx),delta=3e-9)


if __name__=='__main__':unittest.main(verbosity=2)
