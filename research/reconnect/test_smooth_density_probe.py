"""Independent mass-average, PDE divergence and point-kernel input checks."""
import unittest
import numpy as np
from scipy.special import roots_legendre
from trajectory_chain import TrajectorySamples,SI_UNITS
from smooth_density_probe import analytic_reference,make_samples,gauss_reference,point_fields
from kernel2d_study import fixture
from positive_kernel_2d import build_periodic_exchange
from sparse_exchange_audit import audit_laplacian


class SmoothDensityChecks(unittest.TestCase):
    def test_analytic_mass_and_state_match_independent_tensor_quadrature(self):
        r=analytic_reference(8,.25,.02);q=gauss_reference(8,.25,.02,r['labels'],16)
        np.testing.assert_allclose(r['mass'],q['mass'],atol=2e-17,rtol=0.)
        np.testing.assert_allclose(r['state'],q['state'],atol=5e-16,rtol=0.)
        for key in ['rate','laplacian_rate','drift_rate']:np.testing.assert_allclose(r[key],q[key],atol=8e-16,rtol=0.)

    def test_point_PDE_matches_independent_complex_step_flux_divergence(self):
        x=np.array([.04,.13,.37,.68,.91]);y=np.array([.08,.24,.41,.63,.87]);eps=.25;D=.02;k=2*np.pi;step=1e-25
        rho,c,lap,drift,total=point_fields(x,y,eps,D)
        def flux(x,y,axis):
            gradient=-.2*k*(np.sin(k*x)*np.cos(k*y) if axis==0 else np.cos(k*x)*np.sin(k*y))
            return D*(1+eps*np.cos(k*x))*gradient
        divergence=np.imag(flux(x+1j*step,y,0)+flux(x,y+1j*step,1))/step
        np.testing.assert_allclose(total,divergence/rho,atol=3e-16,rtol=2e-15)

    def test_analytic_mass_average_rate_matches_true_face_flux_balance(self):
        N=8;eps=.25;D=.02;k=2*np.pi;r=analytic_reference(N,eps,D);z,w=roots_legendre(16);offset=z/(2*N);weight=w/(2*N)
        for index in [0,9,21,38,63]:
            x,y=r['labels'][index];xl=x-1/(2*N);xr=x+1/(2*N);yl=y-1/(2*N);yr=y+1/(2*N)
            def Fxx(xedge):return D*(1+eps*np.cos(k*xedge))*(-.2*k*np.sin(k*xedge)*np.cos(k*(y+offset)))
            def Fyy(yedge):return D*(1+eps*np.cos(k*(x+offset)))*(-.2*k*np.cos(k*(x+offset))*np.sin(k*yedge))
            integrated=np.dot(weight,Fxx(xr)-Fxx(xl))+np.dot(weight,Fyy(yr)-Fyy(yl))
            self.assertAlmostEqual(integrated/r['mass'][index],r['rate'][index],places=13)

    def test_uniform_density_limit_recovers_saved_control_formulas(self):
        d,r=make_samples(8,0.,.02);old,_,_=fixture(8)
        np.testing.assert_array_equal(d.mass,old.mass);np.testing.assert_allclose(d.c0,old.c0,atol=2e-16)
        np.testing.assert_allclose(r['rate'],-8*np.pi**2*.02*(old.c0-.5),atol=4e-16);np.testing.assert_array_equal(r['drift_rate'],0.)
        L,a,_=build_periodic_exchange(d,1,.02,.19,[1.,1.]);Lo,ao,_=build_periodic_exchange(old,1,.02,.19,[1.,1.])
        np.testing.assert_array_equal(L.data,Lo.data);np.testing.assert_array_equal(a['conductances'],ao['conductances'])

    def test_density_drift_matters_despite_global_mass_checks(self):
        r=analytic_reference(8,.25,.02)
        np.testing.assert_allclose(r['rate'],r['laplacian_rate']+r['drift_rate'],atol=2e-16)
        self.assertGreater(np.sqrt(r['mass']@(r['drift_rate']**2)),.01)
        self.assertLess(abs(r['mass']@r['rate']),1e-15);self.assertLess(abs(r['mass']@r['laplacian_rate']),1e-15)
        self.assertAlmostEqual(r['mass'].sum(),1.);self.assertAlmostEqual(r['mass']@r['state'],.5)

    def test_variable_density_candidate_preserves_positive_graph_without_probe_fitting(self):
        d,r=make_samples(8,.25,.02);L,a,_=build_periodic_exchange(d,1,.02,.19,[1.,1.]);self.assertTrue(audit_laplacian(d.mass,L)['graph_algebra_eligible'])
        changed=TrajectorySamples(d.time,d.positions,d.ids,d.mass,1-d.c0,SI_UNITS,rho=d.rho);Lc,ac,_=build_periodic_exchange(changed,1,.02,.19,[1.,1.])
        np.testing.assert_array_equal(a['conductances'],ac['conductances']);np.testing.assert_array_equal(L.data,Lc.data)
        self.assertTrue(np.all(a['conductances']>0));self.assertLess(float(d.c0@(L@d.c0)*-1),0.)

    def test_point_density_quadrature_areas_are_retained_without_normalizing(self):
        d,r=make_samples(8,.25,.02);saved=d.mass.copy();L,a,_=build_periodic_exchange(d,1,.02,.19,[1.,1.])
        np.testing.assert_array_equal(a['quadrature_area'],d.mass/r['rho_point']);np.testing.assert_array_equal(d.mass,saved)
        self.assertGreater(a['quadrature_area'].sum(),1.);self.assertGreater(np.max(abs(a['quadrature_area']/r['true_area']-1)),1e-3)
        self.assertGreater(np.max(abs(r['rho_average']-r['rho_point'])),1e-3)

    def test_material_state_and_oracle_density_remain_attached_to_ids(self):
        d,r=make_samples(8,.25,.02);times=d.time;velocity=np.array([.2,-.1])
        for s,t in enumerate(times):
            material=np.mod(d.positions[s]-t*velocity,1.);rho,point,*_=point_fields(material[:,0],material[:,1])
            np.testing.assert_allclose(d.rho[s],rho,atol=2e-16)
            self.assertGreater(np.max(abs(point-d.c0)),.001)
        np.testing.assert_array_equal(d.rho[0],d.rho[1]);self.assertFalse(np.array_equal(d.positions[0],d.positions[1]))


if __name__=='__main__':unittest.main(verbosity=2)
