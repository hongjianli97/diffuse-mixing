"""True material averages, dual-boundary transport and FV isolation checks."""
import unittest
import numpy as np
from scipy.integrate import quad
from scipy.linalg import solve
from becker import Becker
from density_geometry_study import geometry_diagnostics,mass_coordinate_at_x,velocity_at_x,point_trace_nonidentifiability
from experiment import step
from fv_density_geometry import density_from_mass_faces,midpoint_dual_faces,cellwise_conductance


class DensityGeometryChecks(unittest.TestCase):
    def setUp(self):
        self.f=Becker();self.labels=np.linspace(-1.,1.,33)
        self.edges=self.f.mu(self.labels);self.m=np.diff(self.edges)

    def test_true_face_mean_density_mass_and_absolute_mass_scaling(self):
        faces=np.array([0.,.1,.4,1.]);m=np.array([.2,.6,1.8])
        rho=density_from_mass_faces(m,faces)
        np.testing.assert_allclose(rho*np.diff(faces),m,atol=1e-15)
        rho13=density_from_mass_faces(13*m,faces)
        np.testing.assert_allclose(rho13,13*rho,rtol=1e-14)
        g=cellwise_conductance(faces,rho,.02);g13=cellwise_conductance(faces,rho13,.02)
        np.testing.assert_allclose(g13,13*g,rtol=1e-14)

    def test_series_resistance_and_double_volume_sensitivity(self):
        faces=np.array([0.,.2,.7,1.]);m=np.array([.4,.5,.9]);V=np.diff(faces);D=.02
        rho=density_from_mass_faces(m,faces);g=cellwise_conductance(faces,rho,D)
        half=V*V/(2*m*D)
        np.testing.assert_allclose(g*(half[:-1]+half[1:]),1.,rtol=1e-14)
        proxyfaces=faces*1.07;proxy_rho=density_from_mass_faces(m,proxyfaces)
        density_only=cellwise_conductance(faces,proxy_rho,D)
        both=cellwise_conductance(proxyfaces,proxy_rho,D)
        np.testing.assert_allclose(density_only,g/1.07,rtol=1e-14)
        np.testing.assert_allclose(both,g/1.07**2,rtol=1e-14)

    def test_midpoint_dual_matches_material_cells_for_affine_motion(self):
        a=np.linspace(-1.,1.,17);nodes=(a[:-1]+a[1:])/2
        for scale,shift in ((1.,0.),(.4,.2),(1.3,-.1)):
            true=scale*a+shift;dual=midpoint_dual_faces(scale*nodes+shift,true[[0,-1]])
            np.testing.assert_allclose(dual,true,atol=3e-16)

    def test_material_mass_and_average_density_against_independent_oracle_integral(self):
        true,points,dual,rhot,rhod,_,_=geometry_diagnostics(self.f,self.labels,self.edges,self.m,np.array([.1]))
        for i in (13,15,17):
            mass=quad(lambda x:float(self.f.j/self.f.u(self.f.z_at_xi(self.f.u1*.1-x))),
                      true[0,i],true[0,i+1],epsabs=1e-12)[0]
            self.assertAlmostEqual(mass/self.m[i],1.,places=11)
            self.assertAlmostEqual(rhot[0,i],mass/(true[0,i+1]-true[0,i]),places=10)
        self.assertGreater(float(np.max(abs(dual-true))),1e-5)

    def test_dual_cell_mass_rate_matches_reynolds_transport_independently(self):
        t=.05;eps=1e-6
        true,points,dual,_,_,_,_=geometry_diagnostics(self.f,self.labels,self.edges,self.m,np.array([t-eps,t,t+eps]))
        actual=np.diff(mass_coordinate_at_x(self.f,dual,np.array([t-eps,t,t+eps])[:,None]),axis=1)
        derivative=(actual[2]-actual[0])/(2*eps)
        v=velocity_at_x(self.f,points[1],t);vb=velocity_at_x(self.f,true[1,[0,-1]],t)
        w=np.r_[vb[0],(v[:-1]+v[1:])/2,vb[1]]
        rho=self.f.j/self.f.u(self.f.z_at_xi(self.f.u1*t-dual[1]))
        flux=rho*(velocity_at_x(self.f,dual[1],t)-w)
        np.testing.assert_allclose(derivative,flux[:-1]-flux[1:],atol=2e-9,rtol=2e-6)
        self.assertGreater(float(np.max(abs(derivative))),1e-4)

    def test_positive_fixed_mass_step_matches_independent_dense_solve_and_D0(self):
        faces=np.array([0.,.2,.6,1.]);m=np.array([.4,.5,.6]);c0=np.array([.2,.8,.4]);dt=.03
        g=cellwise_conductance(faces,density_from_mass_faces(m,faces),.02)
        L=np.diag(np.r_[g,0.]+np.r_[0.,g])-np.diag(g,1)-np.diag(g,-1)
        actual=step(c0,m,g,dt);expected=solve(np.diag(m)+dt*L,m*c0,assume_a='pos')
        np.testing.assert_allclose(actual,expected,atol=2e-15)
        self.assertAlmostEqual(float(m@actual),float(m@c0),places=14)
        self.assertGreaterEqual(actual.min(),c0.min());self.assertLessEqual(actual.max(),c0.max())
        zero=cellwise_conductance(faces,density_from_mass_faces(m,faces),0.)
        np.testing.assert_allclose(step(c0,m,zero,dt),c0,atol=1e-15)

    def test_same_point_traces_do_not_identify_true_material_volumes(self):
        d,a=point_trace_nonidentifiability(self.f,self.labels,.1)
        self.assertGreater(d['minimum_map_derivative_certified_lower_bound'],0.)
        self.assertLess(d['final_node_position_max_difference'],1e-14)
        self.assertGreater(d['final_material_volume_max_relative_difference'],.01)
        np.testing.assert_array_equal(a['base_final_faces'][[0,-1]],a['alternative_final_faces'][[0,-1]])

    def test_invalid_geometry_density_or_identity_order_is_rejected(self):
        with self.assertRaises(ValueError):density_from_mass_faces([1.,1.],[0.,.2,.1])
        with self.assertRaises(ValueError):density_from_mass_faces([1.,0.],[0.,.2,1.])
        with self.assertRaises(ValueError):midpoint_dual_faces([.3,.2],[0.,1.])
        with self.assertRaises(ValueError):midpoint_dual_faces([.2,.4],[.3,1.])
        with self.assertRaises(ValueError):cellwise_conductance([0.,.2,1.],[1.,-1.],.02)
        with self.assertRaises(ValueError):cellwise_conductance([0.,.2,1.],[1.,1.],-.02)


if __name__=='__main__':unittest.main(verbosity=2)
