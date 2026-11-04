"""Scientific checks for the reconstructed experiment, independent of old claims."""
import unittest
import numpy as np
from scipy.integrate import quad,solve_ivp
from becker import Becker,initial
from reference import solve,project
from experiment import conductance


class RebuildChecks(unittest.TestCase):
    def test_width_and_conserved_fluid_fluxes(self):
        f=Becker();self.assertAlmostEqual(float(f.xi(np.log(9))-f.xi(-np.log(9))),.08,places=14)
        targets=[f.j,f.rho1*f.u1**2+f.p1,f.j*f.H]
        for values,target in zip(f.fluxes(np.linspace(-40,40,501)),targets):
            np.testing.assert_allclose(values,target,rtol=2e-14)

    def test_profile_momentum_ode_by_numerical_derivative(self):
        f=Becker();xi=np.linspace(-.1,.1,31);h=1e-6
        z=f.z_at_xi(xi);u=f.u(z)
        numerical=(f.u(f.z_at_xi(xi+h))-f.u(f.z_at_xi(xi-h)))/(2*h)
        expected=f.j*(f.gamma+1)/(2*f.gamma)*(u-f.u1)*(u-f.u2)/(4*f.eta/3*u)
        np.testing.assert_allclose(numerical,expected,rtol=1e-7,atol=1e-8)

    def test_material_trajectories_against_ode_and_mass_quadrature(self):
        f=Becker()
        for a,b in ((-.1,.05),(.0,.0625),(.4,.5)):
            mu=f.mu(np.array([a,b]));m=mu[1]-mu[0]
            def velocity(t,x):return f.u1-f.u(f.z_at_xi(f.u1*t-x))
            sol=solve_ivp(velocity,[0,.1],[a,b],rtol=1e-11,atol=1e-12)
            np.testing.assert_allclose(sol.y[:,-1],f.x_mu(mu,.1),rtol=1e-10,atol=1e-11)
            left,right=f.x_mu(mu,.1)
            integrated=quad(lambda x:f.j/f.u(f.z_at_xi(f.u1*.1-x)),left,right,epsabs=1e-12)[0]
            self.assertAlmostEqual(integrated/m,1.,places=11)

    def test_initial_averages_by_independent_quadrature(self):
        f=Becker();e=f.mu(np.linspace(-1,1,33));c=initial(e)
        exact=np.array([quad(lambda mu:.5+.25*np.cos(np.pi*mu/f.total),a,b)[0]/(b-a) for a,b in zip(e[:-1],e[1:])])
        np.testing.assert_allclose(c,exact,rtol=1e-13)
        self.assertAlmostEqual(float(np.diff(e)@c),f.total*.5,places=13)

    def test_mass_resistance_against_physical_coordinate_integral(self):
        f=Becker();e=f.mu(np.linspace(-1,1,33));mu=(e[:-1]+e[1:])/2;t=.07;D=.02
        g=conductance(f,e,t,D,'ordinary_mass_center')
        for i in (10,14,17,20):
            a,b=f.x_mu(mu[[i,i+1]],t)
            res=quad(lambda x:f.u(f.z_at_xi(f.u1*t-x))/(f.j*D),a,b,epsabs=1e-12)[0]
            self.assertAlmostEqual(g[i]*res,1.,places=11)

    def test_volume_center_matches_existing_physical_space_helper(self):
        from material_fv import center_resistance
        f=Becker();edges=f.mu(np.linspace(-1,1,33));t=.05
        rho=lambda x,time:f.j/f.u(f.z_at_xi(f.u1*time-x))
        expected,_=center_resistance(f.x_mu(edges,t),rho,.02,t)
        actual=conductance(f,edges,t,.02,'ordinary_volume_center')
        np.testing.assert_allclose(actual,expected,rtol=2e-12)

    def test_average_resistance_constant_flux_on_nonuniform_mass_cells(self):
        class Analytic:
            def rho_mu(self,mu,t):return 1/np.sqrt(1+mu)
        e=np.array([0.,.1,.6,1.,2.]);a=e[:-1];b=e[1:]
        psi_mean=(a+b)/2+(a*a+a*b+b*b)/6
        g=conductance(Analytic(),e,0.,1.,'mass_average_resistance')
        np.testing.assert_allclose(g*np.diff(psi_mean),1.,rtol=1e-13)

    def test_reference_against_analytic_heat_mode(self):
        class Uniform:
            total=2.
            def rho_mu(self,mu,t):return np.ones_like(mu)
        f=Uniform();e,c0,c,d=solve(f,128,256)
        exact=.5+(c0-.5)*np.exp(-.02*(np.pi/2)**2*.1)
        self.assertLess(np.max(abs(c-exact)),1e-7)
        e,i,c,d=solve(f,64,32,D=0)
        np.testing.assert_allclose(c,i,atol=1e-14)

    def test_reference_time_order_against_semidiscrete_mode(self):
        class Uniform:
            total=2.
            def rho_mu(self,mu,t):return np.ones_like(mu)
        n=32;lam=4*.02*np.sin(np.pi/(2*n))**2/(2/n)**2
        errors=[]
        for steps in (4,8,16):
            e,c0,c,d=solve(Uniform(),n,steps,T=1.)
            exact=.5+(c0-.5)*np.exp(-lam)
            errors.append(np.linalg.norm(c-exact))
        np.testing.assert_allclose(np.array(errors[:-1])/errors[1:],4.,rtol=.002)

    def test_projection_preserves_mass_and_constant(self):
        e=np.array([0.,.2,.6,1.]);c=np.array([.2,.5,.9]);target=np.array([0.,.1,.4,.9,1.])
        p=project(e,c,target)
        self.assertAlmostEqual(float(np.diff(target)@p),float(np.diff(e)@c),places=14)
        np.testing.assert_allclose(project(e,np.full(3,.37),target),.37,atol=1e-14)


if __name__=='__main__':unittest.main(verbosity=2)
