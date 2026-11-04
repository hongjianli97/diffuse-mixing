"""Four necessary solver checks for the one-case FV time comparison."""
from copy import deepcopy
import argparse
import json
from pathlib import Path
import sys
import unittest
import numpy as np

from cartesian_fv_modes import MATERIAL_PRIOR,build_modes,coefficients,reconstruct,propagate,direct_structured_action
from cartesian_fv_stream import stream_action


def fixture(nx=4,ny=4,homogeneous=False):
    periods=(2.,2.);dx,dy=periods[0]/nx,periods[1]/ny
    rho=np.tile(np.ones(nx) if homogeneous else np.array([1.]*(nx//2)+[4.]*(nx-nx//2)),(ny,1))
    mass=rho*dx*dy
    state=(.2+((np.arange(nx*ny).reshape(ny,nx)*7)%13)/20)
    return rho,mass,state,periods


def exact_action(rho,mass,state,periods,D):
    ny,nx=rho.shape;dx,dy=periods[0]/nx,periods[1]/ny
    positions=[((i+.5)*dx,(j+.5)*dy) for j in range(ny) for i in range(nx)]
    return np.asarray(list(map(float,stream_action(positions,list(range(nx*ny)),mass.ravel(),rho.ravel(),
        periods,D,state.ravel(),model_prior=MATERIAL_PRIOR).rates))).reshape(ny,nx)


class FVModesChecks(unittest.TestCase):
    def test_structured_generator_matches_independent_shared_face_action(self):
        rho,m,c,box=fixture(4,8);before=deepcopy((rho,m,c));D=.02
        modes=build_modes(rho,m,D,box,model_prior=MATERIAL_PRIOR)
        actual,imag=direct_structured_action(modes,c)
        np.testing.assert_allclose(actual,exact_action(rho,m,c,box,D),rtol=1e-13,atol=1e-14)
        self.assertLess(np.max(abs(imag)),1e-14)
        np.testing.assert_array_equal(modes.normal_matrix,modes.normal_matrix.T)
        self.assertTrue(all(g>0 for g in modes.normal_g_exact))
        for a,b in zip((rho,m,c),before):np.testing.assert_array_equal(a,b)

    def test_every_discrete_mode_is_retained_and_homogeneous_decay_is_exact(self):
        rho,m,c,box=fixture(4,8,True);D=.02
        modes=build_modes(rho,m,D,box,model_prior=MATERIAL_PRIOR)
        arbitrary=coefficients(modes,c);recovered,imag=reconstruct(modes,arbitrary)
        np.testing.assert_allclose(recovered,c,rtol=1e-13,atol=1e-14)
        self.assertEqual(arbitrary.shape,c.shape)
        x=np.cos(2*np.pi*np.arange(4)/4);y=np.cos(2*np.pi*3*np.arange(8)/8)
        c=.5+.2*y[:,None]*x[None,:];T=.05
        lam=4*D/(box[0]/4)**2*np.sin(np.pi/4)**2+4*D/(box[1]/8)**2*np.sin(3*np.pi/8)**2
        out,_=propagate(modes,coefficients(modes,c),T)
        np.testing.assert_allclose(out,.5+np.exp(-T*lam)*(c-.5),rtol=1e-13,atol=1e-14)

    def test_implicit_Euler_factors_match_independent_actual_step_solves(self):
        rho,m,c,box=fixture();D=.02;T=.05;steps=8;n=c.size
        modes=build_modes(rho,m,D,box,model_prior=MATERIAL_PRIOR)
        A=np.column_stack([exact_action(rho,m,np.eye(n)[j].reshape(c.shape),box,D).ravel() for j in range(n)])
        value=c.ravel().copy();system=np.eye(n)-(T/steps)*A
        for _ in range(steps):value=np.linalg.solve(system,value)
        spectral,_=propagate(modes,coefficients(modes,c),T,steps)
        np.testing.assert_allclose(spectral.ravel(),value,rtol=1e-13,atol=1e-14)
        self.assertGreaterEqual(spectral.min(),c.min()-1e-13)
        self.assertLessEqual(spectral.max(),c.max()+1e-13)
        self.assertLess(abs(np.sum(m*(spectral-c))),1e-13)

    def test_inconsistent_material_geometry_and_nonfinite_state_are_rejected(self):
        rho,m,c,box=fixture();bad=m.copy();bad[0,0]+=1e-12
        for changes in [{'mass':bad},{'D':-1},{'model_prior':'points only'},
                        {'rho':np.roll(rho,1,axis=1)+np.arange(4)[:,None]}]:
            args={'rho':rho,'mass':m,'D':.02,'periods':box,'model_prior':MATERIAL_PRIOR,**changes}
            with self.assertRaises(ValueError):build_modes(**args)
        modes=build_modes(rho,m,.02,box,model_prior=MATERIAL_PRIOR)
        with self.assertRaises(ValueError):coefficients(modes,c*np.nan)
        with self.assertRaises(ValueError):propagate(modes,coefficients(modes,c),T=.05,steps=0)
        with self.assertRaises(TypeError):build_modes(rho,m,.02,box,model_prior=MATERIAL_PRIOR,reference_PDE=None)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--json');args=parser.parse_args()
    result=unittest.TextTestRunner(verbosity=2,stream=sys.stdout).run(unittest.defaultTestLoader.loadTestsFromTestCase(FVModesChecks))
    summary={'checks_run':result.testsRun,'successful':result.wasSuccessful(),'failures':len(result.failures),
        'errors':len(result.errors),'skipped':len(result.skipped),'previous_checks_rerun':0}
    if args.json:
        with Path(args.json).open('x') as f:json.dump(summary,f,indent=2);f.write('\n')
    sys.exit(0 if result.wasSuccessful() else 1)
