"""Independent checks of contact transmission, exact averages and FV geometry."""
import unittest
import numpy as np
from scipy.integrate import quad
from contact import Contact,METHODS,mesh,semidiscrete,backward_euler


class ContactChecks(unittest.TestCase):
    def test_exact_mode_obeys_bulk_pde_and_one_sided_interface_flux(self):
        f=Contact();t=.07;h=1e-4;eps=1e-5
        for a in (-.6,-.2,.2,.7):
            ct=(f.scalar(a,t+eps)-f.scalar(a,t-eps))/(2*eps)
            cxx=(f.scalar(a+h,t)-2*f.scalar(a,t)+f.scalar(a-h,t))/h**2
            self.assertAlmostEqual(float(ct),float(f.D*cxx),delta=2e-9)
        c0=f.scalar(0.,t)
        dl=(3*c0-4*f.scalar(-h,t)+f.scalar(-2*h,t))/(2*h)
        dr=(-3*c0+4*f.scalar(h,t)-f.scalar(2*h,t))/(2*h)
        self.assertAlmostEqual(float(-f.rho_left*f.D*dl),float(-f.rho_right*f.D*dr),places=10)
        self.assertAlmostEqual(float(-f.rho_left*f.D*dl),float(f.flux(0.,t)),delta=2e-10)
        for a in (-1.,1.):self.assertAlmostEqual(float(f.flux(a,t)),0.,places=14)

    def test_mass_and_scalar_averages_against_split_physical_quadrature(self):
        f=Contact();faces=np.array([-1.,-.4,-.07,.023,.4,1.]);t=.1
        expected=[];masses=[]
        for a,b in zip(faces[:-1],faces[1:]):
            pieces=([a,0.,b] if a<0<b else [a,b]);mass=scalar=0.
            for l,r in zip(pieces[:-1],pieces[1:]):
                rho=f.rho_left if (l+r)/2<0 else f.rho_right
                mass+=rho*(r-l);scalar+=quad(lambda x:rho*f.scalar(x,t),l,r,epsabs=1e-13)[0]
            masses.append(mass);expected.append(scalar/mass)
        np.testing.assert_allclose(f.mass(faces),masses,rtol=2e-14)
        np.testing.assert_allclose(f.cell_average(faces,t),expected,rtol=2e-14)
        self.assertAlmostEqual(float(f.mass(faces)@f.cell_average(faces,t)),.5*f.total_mass,places=13)

    def test_aligned_interface_is_harmonic_resistance_not_arithmetic_density(self):
        f=Contact();faces=mesh(32,'aligned');h=2/32;idx=15
        expected=1/(h/(2*f.rho_left*f.D)+h/(2*f.rho_right*f.D))
        arithmetic=.5*(f.rho_left+f.rho_right)*f.D/h
        self.assertGreater(abs(expected-arithmetic),.01)
        for method in METHODS:
            self.assertAlmostEqual(f.conductance(faces,method)[idx],expected,places=13)

    def test_mass_average_local_constant_flux_across_cut_contact(self):
        f=Contact();faces=mesh(32,'cut');m=f.mass(faces)
        # c=A+B psi is a local nonzero-flux manufactured field. This does not
        # impose nonzero steady flux on the closed experimental domain.
        psi_bar=[]
        for a,b,mi in zip(faces[:-1],faces[1:],m):
            points=[a,0.,b] if a<0<b else [a,b];v=0.
            for l,r in zip(points[:-1],points[1:]):
                rho=f.rho_left if (l+r)/2<0 else f.rho_right
                v+=quad(lambda x:rho*f.resistance_potential(x),l,r)[0]
            psi_bar.append(v/mi)
        g=f.conductance(faces,'mass_average_resistance')
        np.testing.assert_allclose(g*np.diff(psi_bar),1.,rtol=2e-13)

    def test_geometric_resistance_by_lab_frame_integration(self):
        f=Contact();faces=mesh(32,'cut');centers=(faces[:-1]+faces[1:])/2
        g=f.conductance(faces,'ordinary_volume_center');t=.1
        for i in (14,15,16,17):
            left,right=centers[i]+f.speed*t,centers[i+1]+f.speed*t;front=f.speed*t
            points=[left,front,right] if left<front<right else [left,right]
            resistance=0.
            for a,b in zip(points[:-1],points[1:]):
                rho=f.rho_left if (a+b)/2<front else f.rho_right
                resistance+=quad(lambda x:1/(rho*f.D),a,b)[0]
            self.assertAlmostEqual(g[i]*resistance,1.,places=12)
        changed=Contact(speed=-.7)
        for method in METHODS:np.testing.assert_array_equal(f.conductance(faces,method),changed.conductance(faces,method))

    def test_aligned_semidiscrete_eigenmode_matches_independent_eigenvalue(self):
        f=Contact();faces=mesh(32,'aligned');m=f.mass(faces);c0=f.cell_average(faces,0.)
        rate=4*f.D/(2/32)**2*np.sin(np.pi/(2*32))**2
        exact=.5+(c0-.5)*np.exp(-rate*.1)
        for method in METHODS:
            c,P,d=semidiscrete(c0,m,f.conductance(faces,method),.1)
            np.testing.assert_allclose(c,exact,rtol=2e-13)

    def test_zero_diffusion_and_constant_preservation_in_cut_cells(self):
        f=Contact(D=0.);faces=mesh(32,'cut');m=f.mass(faces);c0=f.cell_average(faces,0.)
        for method in METHODS:
            g=f.conductance(faces,method);np.testing.assert_array_equal(g,0.)
            c,d=backward_euler(c0,m,g,.1,32);np.testing.assert_allclose(c,c0,atol=1e-14)
        f=Contact()
        for method in METHODS:
            for value in (0.,.37,1.):
                c,d=backward_euler(np.full(32,value),m,f.conductance(faces,method),.1,32)
                np.testing.assert_allclose(c,value,atol=1e-14)

    def test_mass_symmetrized_eigendecomposition_matches_dense_exponential(self):
        from scipy.linalg import eigh
        from contact import laplacian
        f=Contact();faces=mesh(64,'cut');m=f.mass(faces);c0=f.cell_average(faces,0.);root=np.sqrt(m)
        for method in METHODS:
            g=f.conductance(faces,method);L=laplacian(g)
            vals,vec=eigh(L/root[:,None]/root[None,:])
            independent=(vec@(np.exp(-.1*vals)*(vec.T@(root*c0))))/root
            actual,P,d=semidiscrete(c0,m,g,.1)
            np.testing.assert_allclose(actual,independent,atol=2e-13,rtol=0.)

    def test_reject_nonphysical_parameters_and_invalid_faces(self):
        for kwargs in ({'rho_left':0.},{'D':-.01},{'rho_left':10.},{'speed':float('nan')}):
            with self.assertRaises(ValueError):Contact(**kwargs)
        with self.assertRaises(ValueError):Contact().conductance([0.,0.,1.],METHODS[0])
        with self.assertRaises(ValueError):mesh(31,'cut')


if __name__=='__main__':unittest.main(verbosity=2)
