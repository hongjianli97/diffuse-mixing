"""Independent map, full PDE, face integrals and unchanged point-backend tests."""
import unittest
import numpy as np
from scipy.special import roots_legendre
from shear_probe import shear_map,make_shear_samples,shear_point_fields,material_flux,shear_gauss_reference,shear_face_reference,shear_geometry
from smooth_density_probe import make_samples
from positive_kernel_2d import build_periodic_exchange
from sparse_exchange_audit import audit_laplacian


class ShearChecks(unittest.TestCase):
    def samples(self,s=.25):
        initial,r=make_samples(8,.25,.02)
        data,g,a=make_shear_samples(r['labels'],initial.ids,initial.mass,initial.c0,initial.rho[0],[0.,.05],s)
        return initial,r,data

    def test_unit_determinant_and_periodic_map_from_complex_derivatives(self):
        a=np.array([.13,.31,.57]);b=np.array([.19,.42,.83]);step=1e-25;s=.25
        xa,ya=shear_map(a+1j*step,b,s,.05);xb,yb=shear_map(a,b+1j*step,s,.05)
        determinant=(xa.imag/step)*(yb.imag/step)-(xb.imag/step)*(ya.imag/step)
        np.testing.assert_allclose(determinant,1.,atol=2e-16,rtol=0.)
        x,y=shear_map(a,b,s,.05);xx,yy=shear_map(a,b+1,s,.05)
        np.testing.assert_allclose(xx,x,atol=2e-16,rtol=0.)
        # y carries two translated-coordinate roundings. At unit period one
        # float64 ulp is 2.22e-16, larger than the initial absolute 2e-16 test.
        # Check the representation in ulps; no PDE/algebra tolerance changes.
        np.testing.assert_array_max_ulp(yy-y,np.ones_like(y),maxulp=1)

    def test_fixed_material_mass_c0_ids_and_advected_oracle_density(self):
        initial,r,d=self.samples()
        for key in ['mass','c0','ids']:np.testing.assert_array_equal(getattr(d,key),getattr(initial,key))
        np.testing.assert_array_equal(d.rho,initial.rho)
        self.assertGreater(np.max(abs(d.positions[1]-initial.positions[1])),.03)

    def test_full_point_PDE_matches_independent_material_flux_complex_derivative(self):
        a=np.array([.11,.29,.53,.81]);b=np.array([.17,.33,.49,.73]);step=1e-25
        rho,c,parts=shear_point_fields(a,b,.25,.25,.02)
        independent=(np.imag(material_flux(a+1j*step,b,.25,.25,.02)[0])+np.imag(material_flux(a,b+1j*step,.25,.25,.02)[1]))/step/rho
        np.testing.assert_allclose(parts['rate'],independent,atol=5e-16,rtol=2e-15)
        for key in ['mixed_derivative','shear_gradient','density_drift']:self.assertGreater(np.max(abs(parts[key])),.001)

    def test_integrated_face_flux_matches_independent_full_Eulerian_integral(self):
        initial,r,d=self.samples();q=shear_gauss_reference(8,.25,.25,.02,r['labels'],d.mass,16)
        face=shear_face_reference(8,.25,.25,.02,r['labels'],d.mass,16)
        np.testing.assert_allclose(q['rate'],face['rate'],atol=4e-15,rtol=2e-15)
        np.testing.assert_allclose(q['mass'],d.mass,atol=2e-17,rtol=0.)
        np.testing.assert_allclose(q['state'],d.c0,atol=5e-16,rtol=0.)

    def test_zero_shear_recovers_saved_translation_and_existing_PDE(self):
        initial,r,d=self.samples(0.)
        np.testing.assert_array_equal(d.positions,initial.positions)
        face=shear_face_reference(8,.25,0.,.02,r['labels'],d.mass,16)
        np.testing.assert_allclose(face['rate'],r['rate'],atol=3e-15,rtol=2e-15)

    def test_area_centroid_offset_raw_point_weights_positive_graph_and_D_zero(self):
        initial,r,d=self.samples();geometry=shear_geometry(8,r['labels'],.25,.05)
        z,w=roots_legendre(16);a=r['labels'];b=a[:,1,None]+z[None,:]/16
        x,_=shear_map(a[:,0,None],b,.25,.05)
        np.testing.assert_allclose(x@(w/2),geometry['true_area_centroid_x'],atol=2e-16,rtol=0.)
        self.assertGreater(np.max(abs(geometry['node_minus_area_centroid_x'])),1e-4)
        L,arrays,diag=build_periodic_exchange(d,1,.02,.19,[1.,1.])
        self.assertTrue(audit_laplacian(d.mass,L)['graph_algebra_eligible'])
        np.testing.assert_array_equal(arrays['quadrature_area'],d.mass/d.rho[1])
        self.assertGreater(np.max(abs(arrays['quadrature_area']/geometry['true_cell_area']-1)),.001)
        zero,az,_=build_periodic_exchange(d,1,0.,.19,[1.,1.]);np.testing.assert_array_equal(zero.data,0.);np.testing.assert_array_equal(az['conductances'],0.)

    def test_invalid_or_unexplained_shear_samples_are_rejected(self):
        initial,r=make_samples(8,.25,.02)
        for times,s,end in [([0.,.1],.25,.05),([0.,.05],np.nan,.05),([0.,.05],.25,0.)]:
            with self.assertRaises(ValueError):make_shear_samples(r['labels'],initial.ids,initial.mass,initial.c0,initial.rho[0],times,s,end)


if __name__=='__main__':unittest.main(verbosity=2)
