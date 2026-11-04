"""Fixed two-mode affine-shear references, raw observables and positive error budgets."""
from fractions import Fraction as F
import math
import numpy as np
from scipy.sparse import csr_matrix
from sliding_periodic_voronoi import SlidingPeriodicDomain
from voronoi_time_action import metrics
from t1_uniform_core import h_indicator

COUNTS=(4,8,16);TIMES=(F(0),F(1,40),F(1,20));D=F(float(.02))


def pure_inputs(labels,ids):
    points=tuple(tuple(F(float(x)) for x in p) for p in labels);result={}
    for K in COUNTS:
        frames=[]
        for j in range(2*K+1):
            t=F(j,40*K);s=5*t;boundary=SlidingPeriodicDomain(s)
            frames.append({'frame':j,'time_exact':str(t),'shear_exact':str(s),'boundary':boundary.packet(),
              'world_positions_exact':[list(map(str,boundary.push(p))) for p in points],'rho':[1.]*len(ids),'D_float':float(D)})
        result[str(K)]={'K':K,'frames':frames}
    return {'ids':list(map(int,ids)),'cases':result,'material_mass_c0_reference_or_internal_faces_supplied':False}


def directional_certificate(packet,frame,N):
    s=F(frame['shear_exact']);points=[tuple(map(F,p)) for p in frame['world_positions_exact']];h=F(1,N)
    expected=(D*(1-s+s*s),D*(1-s),D*s);counts=[0,0,0];moments=[[[F(0),F(0)],[F(0),F(0)]] for _ in points]
    valid=frame['rho']==[1.]*len(points) and F(float(frame['D_float']))==D and len(points)==N*N
    for (i,j),(sx,sy),value in zip(packet['edges'],packet['image_shifts'],packet['conductances_exact']):
        dx=points[j][0]+sx+s*sy-points[i][0];dy=points[j][1]+sy-points[i][1];g=F(value)
        if dy==0 and abs(dx)==h:k=0
        elif abs(dy)==h and dx==s*dy:k=1
        elif abs(dy)==h and dx==(s-1)*dy:k=2
        else:valid=False;continue
        counts[k]+=1;valid=valid and g==expected[k] and g>0
        for node in (i,j):
            for a,x in enumerate((dx,dy)):
                for b,y in enumerate((dx,dy)):moments[node][a][b]+=g*x*y
    target=[[2*D*h*h,F(0)],[F(0),2*D*h*h]]
    moment=all(m==target for m in moments);areas=set(map(F,packet['cell_areas_exact']))=={h*h}
    return {'shear_exact':str(s),'expected_g_exact':list(map(str,expected)),'directional_face_counts':counts,
      'all_three_direction_g_and_counts_exact':valid and counts==[N*N]*3,'all_cell_second_moment_tensors_exact':moment,
      'all_cell_areas_exact':areas,'all_geometry_controls_passed':valid and counts==[N*N]*3 and moment and areas,
      'analytic_coefficients_supplied_to_builder':False}


def physical_clocks(t):
    t=F(t);base=2*t+F(25,3)*t**3
    return D*(base-5*t*t),D*(base+5*t*t)


def continuous_grid_coefficients(t):
    t=F(t);return D*(2*t-5*t*t+F(25,3)*t**3),D*F(5,2)*t*t


def frozen_coefficients(world,K,stop):
    U=V=F(0);dt=F(1,20*K)
    for step in range(stop):
        s=F(world['cases'][str(K)]['frames'][2*step+1]['shear_exact']);U+=dt*D*(2-2*s+s*s);V+=dt*D*s
    return U,V


def mode_data(labels,N):
    a=np.asarray(labels,dtype=float);weights=np.array([np.sin(2*math.pi*(a[:,0]+eta*a[:,1])) for eta in (1,-1)])
    amplitude=.1*(math.sin(math.pi/N)/(math.pi/N))**2
    return weights,amplitude*weights


def reference_bundle(data,labels,world,N=32):
    m,c0=data['mass'],data['c0'];weights,modes=mode_data(labels,N);u=math.sin(math.pi/N)**2;v=math.sin(2*math.pi/N)**2
    def state(exponents):return .5+modes[0]*math.exp(-exponents[0])+modes[1]*math.exp(-exponents[1])
    def grid(U,V):return state((4*N*N*u*float(U),4*N*N*(u*float(U)+v*float(V))))
    reference=np.array([state(tuple(4*math.pi**2*float(q) for q in physical_clocks(t))) for t in TIMES])
    continuous=np.array([grid(*continuous_grid_coefficients(t)) for t in TIMES]);frozen={};coeffs={}
    for K in COUNTS:
        values=[frozen_coefficients(world,K,k*K//2) for k in range(3)]
        frozen[K]=np.array([grid(*pair) for pair in values]);coeffs[str(K)]=[list(map(str,pair)) for pair in values]
    R=float(64*np.finfo(float).eps*math.sqrt(float(m.sum()))*max(1.,float(np.max(abs(c0)))))
    return {'weights':weights,'modes':modes,'reference':reference,'continuous_grid_symbol':continuous,'frozen_symbols':frozen,
      'meta':{'physical_clocks_exact':[list(map(str,physical_clocks(t))) for t in TIMES],
        'continuous_grid_coefficients_exact':[list(map(str,continuous_grid_coefficients(t))) for t in TIMES],
        'frozen_coefficients_exact':coeffs,'reference_evaluation_indicator_absolute':R},'reference_evaluation_indicator_absolute':R}


def modal_certificate(data,labels,frame,action,N,step):
    m=data['mass'];root=np.sqrt(m);weights,modes=mode_data(labels,N);s=F(frame['shear_exact'])
    u=math.sin(math.pi/N)**2;v=math.sin(2*math.pi/N)**2;U=float(D*(2-2*s+s*s));V=float(D*s)
    lambdas=(4*N*N*u*U,4*N*N*(u*U+v*V))
    A=csr_matrix((action['symmetric_data'],action['symmetric_indices'],action['symmetric_indptr']),shape=tuple(action['shape']))
    norm=lambda x:float(np.sqrt(m@(x*x)))
    residues=[norm(A@(root*phi)/root+lam*phi) for phi,lam in zip(modes,lambdas)];constant=norm(.5*(A@root)/root)
    limits=[1e-10*max(1.,lam*norm(phi)) for phi,lam in zip(modes,lambdas)]
    if any(value>limit for value,limit in zip(residues,limits)) or constant>1e-10:raise ArithmeticError('original per-mode/constant algebra guard failed')
    return {'step':step,'declared_frozen_eigenvalues':list(lambdas),'mode_residues_absolute':residues,'constant_residue_absolute':constant,
      'original_mode_algebra_limits':limits,'all_original_residue_guards_passed':True}


def evaluate(data,labels,world,runs,paths,residues,N=32):
    m,c0=data['mass'],data['c0'];b=reference_bundle(data,labels,world,N);norm=lambda x:float(np.sqrt(m@(x*x)))
    H=lambda c:metrics(m,c,c0)['H'];sigma=math.sqrt(metrics(m,c0,c0)['variance']);R=b['reference_evaluation_indicator_absolute']
    bridge=norm(c0-b['reference'][0]);cases={};symbols=[];observations=[]
    for K in COUNTS:
        rows=[]
        for k,t in enumerate(TIMES):
            i=k*K//2;c=paths[K]['states'][i];other=paths[K]['independent_states'][i]
            machine=0. if i==0 else runs[K]['steps'][i-1]['cumulative_machine_indicator_absolute']+runs[K]['steps'][i-1]['cumulative_tail_indicator_absolute']
            observed=norm(c-other);defect=sum(residues[K][:i])/(20*K)
            numerical={'machine_tail':machine,'observed_complete_sequence':observed,'integrated_mode_constant_residues':defect,
              'twice_reference_evaluation':2*R,'initial_two_mode_bridge':bridge}
            nb=sum(numerical.values());frozen=b['frozen_symbols'][K][k];delta=norm(c-frozen)
            symbols.append({'K':K,'time':float(t),'actual_vs_frozen_symbol_absolute':delta,'fixed_numerical_indicator_absolute':nb,
              'symbol_within_indicator':delta<=nb,'H_within_indicator':abs(H(c)-H(frozen))<=h_indicator(m,c0,nb),
              'complete_sequence_within_original_allowance':observed<=machine})
            temporal=norm(frozen-b['continuous_grid_symbol'][k]);spatial=norm(b['continuous_grid_symbol'][k]-b['reference'][k])
            parts={**numerical,'midpoint_time_error':temporal,'spatial_symbol_error':spatial};budget=sum(parts.values());error=norm(c-b['reference'][k])
            rows.append({'time':float(t),'components_absolute':parts,'total_absolute':budget,'reference_error_within_indicator':error<=budget,
              'H_reference_error_within_indicator':abs(H(c)-H(b['reference'][k]))<=h_indicator(m,c0,budget)})
            if K==16:observations.append({'time':float(t),'reference_error_relative':error/sigma,'H':H(c),'reference_H':H(b['reference'][k]),
              'raw_mass_change':float(m@(c-c0)),'spatial_symbol_error_relative':spatial/sigma,'midpoint_time_error_relative':temporal/sigma,
              'fixed_total_indicator_relative':budget/sigma})
        cases[K]=rows
    time_rows=[]
    for k,t in enumerate(TIMES):
        c4,c8,c16=(paths[K]['states'][k*K//2] for K in COUNTS)
        d48=norm(c4-c8);d816=norm(c8-c16);h48=abs(H(c4)-H(c8));h816=abs(H(c8)-H(c16))
        numeric=2*sum(sum(row[k]['components_absolute'][key] for key in ('machine_tail','observed_complete_sequence','integrated_mode_constant_residues','twice_reference_evaluation','initial_two_mode_bridge')) for row in cases.values())
        time_rows.append({'time':float(t),'state4_8_relative':d48/sigma,'state8_16_relative':d816/sigma,'H4_8':h48,'H8_16':h816,
          'fixed_state_decrease_indicator_relative':numeric/sigma,'fixed_H_decrease_indicator':h_indicator(m,c0,numeric),
          'state_time_decrease_resolved':d48-d816>numeric,'H_time_decrease_resolved':h48-h816>h_indicator(m,c0,numeric)})
    mu0=metrics(m,c0,c0)['mean'];weights=b['weights'];wnorms=[norm(w) for w in weights]
    raw=lambda c:np.array([float(m@((c-mu0)*w)) for w in weights])
    initial=raw(c0);refinitial=raw(b['reference'][0]);direction=[]
    for k,t in enumerate(TIMES):
        actual=raw(paths[16]['states'][8*k]);a8=raw(paths[8]['states'][4*k]);ref=raw(b['reference'][k])
        changes=actual-initial;refchanges=ref-refinitial
        indicators=[w*(cases[16][k]['total_absolute']+cases[16][0]['total_absolute']+2*R)+abs(x-y) for w,x,y in zip(wnorms,actual,a8)]
        contrast=float(changes[0]-changes[1]);refcontrast=float(refchanges[0]-refchanges[1]);budget=sum(indicators)
        direction.append({'time':float(t),'raw_actual_observables':actual.tolist(),'raw_initial_observables':initial.tolist(),
          'raw_reference_observables':ref.tolist(),'raw_reference_initial_observables':refinitial.tolist(),
          'actual_changes':changes.tolist(),'reference_changes':refchanges.tolist(),'positive_observable_indicators':indicators,
          'both_observable_changes_reference_consistent':all(abs(x-y)<=z for x,y,z in zip(changes,refchanges,indicators)),
          'actual_direction_contrast':contrast,'reference_direction_contrast':refcontrast,'fixed_positive_contrast_indicator':budget,
          'direction_contrast_reference_consistent':abs(contrast-refcontrast)<=budget,'direction_contrast_resolved':contrast>budget,
          'state_or_observables_rescaled_or_projected':False,'signed_error_cancellation_used_for_acceptance':False})
    solver_ok=all(x['symbol_within_indicator'] and x['H_within_indicator'] and x['complete_sequence_within_original_allowance'] for x in symbols)
    ref_ok=all(x['reference_error_within_indicator'] and x['H_reference_error_within_indicator'] for row in cases.values() for x in row)
    time_ok=all(x['state_time_decrease_resolved'] and x['H_time_decrease_resolved'] for x in time_rows if x['time']>0)
    direction_ok=all(x['both_observable_changes_reference_consistent'] and x['direction_contrast_reference_consistent'] and x['direction_contrast_resolved'] for x in direction if x['time']>0)
    passed=solver_ok and ref_ok and time_ok and direction_ok
    return {'observations':observations,'initial_mass_fluctuation_norm':sigma,'initial_two_mode_bridge_absolute':bridge,
      'frozen_symbol_controls':symbols,'reference_error_components':{str(K):row for K,row in cases.items()},'time_controls':time_rows,
      'direction_observables':direction,'all_frozen_symbol_and_sequence_controls_passed':solver_ok,'all_reference_decompositions_passed':ref_ok,
      'time_decreases_resolved':time_ok,'two_tilted_mode_direction_response_resolved':direction_ok,'registered_affine_shear_numerical_control_passed':passed,
      'comparison_decision':'T3_affine_shear_control_passed' if passed else 'T3_affine_shear_control_failed_or_unresolved',
      'single_N_case_not_spatial_convergence':True,'indicators_are_certified_total_error_bounds':False,'general_T3_admitted':False,
      'general_nonorthogonal_domains_admitted':False,'stage3_physical_acceptance':'incomplete','fluid_orc_enabled':False},b
