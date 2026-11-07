"""Two predeclared bounded software fixtures, not Becker physical experiments."""
from pathlib import Path
import sys,unittest
import numpy as np
from aligned_mass_reference import build_mesh,observe,solve,StepRestrictionError,RawStateError
from audit_route_decision import ROOT,save
sys.path.insert(0,str(ROOT/'research/rebuild'))
from becker import initial
from reference import solve as legacy_solve

OUTPUT=None
CALLS=[]
METRICS={}


class ConstantDensity:
    total=1.
    def __init__(self):self.calls=[]
    def rho_mu(self,mu,t):self.calls.append(float(t));return np.ones_like(mu)


def variable_density(mu,t):return 1.+.25*np.asarray(mu)+.125*t


def run_case(name,mesh,c0,rho,steps=8,T=.05,D=.02,positivity='require'):
    assert len(mesh.mass)<=32 and steps<=16
    try:
        c,diag,history=solve(mesh,c0,rho,steps,T,D,positivity,True)
    except (StepRestrictionError,RawStateError) as exc:
        CALLS.append({'name':name,'kind':'prototype','outcome':type(exc).__name__,'cells':len(mesh.mass),'requested_steps':steps,**exc.diagnostics})
        if OUTPUT:
            np.savez_compressed(OUTPUT/(name+'.npz'),edges=mesh.edges,mass=mesh.mass,c0=c0,
              raw_state=getattr(exc,'raw_state',np.array([])),margin=getattr(exc,'margin',np.array([])))
        raise
    CALLS.append({'name':name,'kind':'prototype','outcome':'completed','cells':len(mesh.mass),'requested_steps':steps,**diag})
    if OUTPUT:
        np.savez_compressed(OUTPUT/(name+'.npz'),parent_edges=mesh.parent_edges,counts=np.array(mesh.counts),edges=mesh.edges,mass=mesh.mass,
          centres=mesh.centres,starts=mesh.starts,c0=c0,c=c,states=history['states'],g=np.array([row['g'] for row in history['actions']]),
          times=np.array([row['time'] for row in history['actions']]),dt=np.array(T/steps),observed_initial=observe(mesh,c0),observed_final=observe(mesh,c))
    return c,diag,history


class AlignedReferenceFixtures(unittest.TestCase):
    def test_01_uniform_limit(self):
        mesh=build_mesh([0.,1.],[16]);flow=ConstantDensity();e,c0,expected,legacy_diag=legacy_solve(flow,16,8,T=.05,D=.02)
        CALLS.append({'name':'legacy_uniform','kind':'legacy','outcome':'completed','cells':16,'requested_steps':8,'linear_solves':len(flow.calls),**legacy_diag})
        c,diag,hist=run_case('uniform',mesh,c0,ConstantDensity().rho_mu)
        np.testing.assert_array_equal(mesh.edges,e);np.testing.assert_array_equal(c,expected)
        np.testing.assert_array_equal(c0,initial(mesh.edges));self.assertEqual(diag['linear_solves'],8)
        self.assertEqual(diag['steps_without_sufficient_positivity'],0)
        self.assertEqual(flow.calls,[(i+.5)*(.05/8) for i in range(8)])
        METRICS['uniform_bitwise_equal_to_unchanged_legacy']=True
        if OUTPUT:np.savez_compressed(OUTPUT/'legacy-uniform.npz',edges=e,c0=c0,c=expected)

    def test_02_nonuniform_material_contract(self):
        p=np.array([0.,.125,.375,1.]);mesh=build_mesh(p,[2,4,2]);fine=build_mesh(p,[2,4,2],level=1)
        np.testing.assert_array_equal(mesh.edges,fine.edges[::2])
        np.testing.assert_array_equal(mesh.edges[mesh.starts],p);np.testing.assert_array_equal(fine.edges[fine.starts],p)
        c0=initial(mesh.edges);fine0=initial(fine.edges);target=initial(p);allowance=64*np.finfo(float).eps*np.sqrt(p[-1]-p[0])
        bridge=float(np.sqrt(mesh.parent_mass@((observe(mesh,c0)-target)**2)))
        finebridge=float(np.sqrt(mesh.parent_mass@((observe(fine,fine0)-target)**2)))
        self.assertLessEqual(bridge,allowance);self.assertLessEqual(finebridge,allowance)
        # Complete-child summation, independent of analytic initial values.
        values=.1+.8*np.arange(len(fine.mass))/(len(fine.mass)-1)
        mid=build_mesh(mesh.edges,np.full(len(mesh.mass),2,dtype=int))
        np.testing.assert_array_equal(mid.edges,fine.edges)
        direct=observe(fine,values);composed=observe(mesh,observe(mid,values))
        self.assertLessEqual(float(np.max(abs(direct-composed))),allowance)
        self.assertLessEqual(abs(float(mesh.parent_mass@direct)-float(fine.mass@values)),allowance)
        c,diag,hist=run_case('nonuniform',mesh,c0,variable_density)
        self.assertLessEqual(diag['max_relative_mass_drift'],1e-9);self.assertEqual(diag['steps_without_sufficient_positivity'],0)
        const,cd,ch=run_case('constant',mesh,np.full(len(mesh.mass),.5),variable_density)
        self.assertLessEqual(float(np.max(abs(const-.5))),allowance)
        zero,zd,zh=run_case('D0',mesh,c0,variable_density,D=0.)
        np.testing.assert_array_equal(zero,c0);np.testing.assert_array_equal(zh['states'],np.tile(c0,(9,1)));self.assertEqual(zd['linear_solves'],0)
        # True quadratic cell means expose the nonuniform first-order face term.
        a,b=mesh.edges[:-1],mesh.edges[1:];quadratic=(a*a+a*b+b*b)/3
        gradient=np.diff(quadratic)/np.diff(mesh.centres);expected=2*mesh.edges[1:-1]+2*(mesh.mass[1:]-mesh.mass[:-1])/3
        self.assertLessEqual(float(np.max(abs(gradient-expected))),allowance)
        jump=build_mesh(p,[1,1,1]);pulse=np.array([1.,0.,0.])
        with self.assertRaises(StepRestrictionError) as strict:run_case('strict-rejected',jump,pulse,ConstantDensity().rho_mu,steps=1,T=10.,D=1.)
        self.assertEqual(strict.exception.diagnostics['linear_solves'],0);self.assertLess(float(strict.exception.margin.min()),0.)
        with self.assertRaises(RawStateError) as monitored:run_case('monitor-rejected',jump,pulse,ConstantDensity().rho_mu,steps=1,T=10.,D=1.,positivity='monitor')
        self.assertEqual(monitored.exception.diagnostics['linear_solves'],1);self.assertLess(float(monitored.exception.raw_state.min()),-1e-10)
        self.assertFalse(monitored.exception.diagnostics['states_clipped_or_normalized'])
        METRICS.update({'nonuniform_initial_bridge':bridge,'fine_initial_bridge':finebridge,'initial_arithmetic_allowance':allowance,
          'constant_maximum_difference':float(np.max(abs(const-.5))),'nested_observation_maximum_difference':float(np.max(abs(direct-composed))),
          'quadratic_gradient_formula_maximum_difference':float(np.max(abs(gradient-expected))),
          'strict_rejected_before_linear_solve':True,'monitor_rejected_raw_minimum':float(monitored.exception.raw_state.min()),
          'D0_identity_bitwise':True,'parent_masses_not_normalized':True})
        if OUTPUT:
            np.savez_compressed(OUTPUT/'aggregation.npz',parent_edges=p,base_counts=np.array(mesh.counts),base_edges=mesh.edges,fine_edges=fine.edges,
              parent_c0=target,base_c0=c0,fine_c0=fine0,base_observed=observe(mesh,c0),fine_observed=observe(fine,fine0),
              arbitrary_fine_values=values,direct=direct,composed=composed,quadratic_gradient=gradient,quadratic_expected=expected)


if __name__=='__main__':unittest.main()
