"""Pure planar affine motion and two distinct constitutive laws; no material scalar inputs."""
from fractions import Fraction as F
import numpy as np

PARAMETER=F(float(.02))
MODELS=('constant_D','constant_Gamma')
COUNTS=(4,8,16)


def constitutive_values(model,a):
    a=F(a)
    if model not in MODELS or not 0<a<=1:raise ValueError('declared positive compression and model required')
    rho=1/a;D=PARAMETER if model=='constant_D' else PARAMETER*a
    rf,df=float(rho),float(D);actual=F(rf)*F(df);nominal=PARAMETER/a if model=='constant_D' else PARAMETER
    return {'rho_exact':str(rho),'rho_float':rf,'D_nominal_exact':str(D),'D_float':df,
      'Gamma_nominal_exact':str(nominal),'Gamma_actual_exact':str(actual),'Gamma_difference_exact':str(actual-nominal)}


def pure_inputs(labels,ids):
    a=np.asarray(labels,dtype=float);ids=np.asarray(ids)
    if a.shape!=(len(ids),2) or not np.isfinite(a).all() or ids.dtype.kind not in 'iu' or len(set(map(int,ids)))!=len(ids):raise ValueError('complete pure labels and uniqueIDs required')
    source=tuple(tuple(F(float(v)) for v in p) for p in a);models={}
    for model in MODELS:
        models[model]={}
        for K in COUNTS:
            frames=[]
            for j in range(2*K+1):
                t=F(j,40*K);scale=1-5*t;values=constitutive_values(model,scale)
                boundary={'origin_exact':['0','0'],'periods_exact':[str(scale),'1'],
                  'orientation_rows_exact':[['1','0'],['0','1']],'basis_columns_exact':[[str(scale),'0'],['0','1']]}
                frames.append({'frame':j,'time_exact':str(t),'a_exact':str(scale),'model':model,'boundary':boundary,
                  'world_positions_exact':[[str(scale*x),str(y)] for x,y in source],'rho':[values['rho_float']]*len(ids),**values})
            models[model][str(K)]={'K':K,'frames':frames}
    return {'ids':list(map(int,ids)),'models':models,'T_exact':'1/20','parameter_exact_wire':str(PARAMETER),
      'law':'a=1-t/(4T),physicalpoints(a*labelx,labely),periods(a,1),rho=1/a',
      'material_mass_c0_reference_or_internal_faces_supplied':False}


def directional_certificate(packet,frame,ids,N):
    points=tuple(tuple(map(F,p)) for p in frame['world_positions_exact']);scale=F(frame['a_exact'])
    rho=frame['rho_float'];D=frame['D_float'];gamma=F(rho)*F(D)
    fields=constitutive_values(frame['model'],scale)
    valid=all(frame[k]==v for k,v in fields.items()) and frame['rho']==[rho]*len(ids) and packet['ids']==list(map(int,ids))
    gx,gy=gamma/scale,gamma*scale;counts={'x':0,'y':0};wrong=0
    for (i,j),(sx,sy),value in zip(packet['edges'],packet['image_shifts'],packet['conductances_exact']):
        dx=points[j][0]+sx*scale-points[i][0];dy=points[j][1]+sy-points[i][1];g=F(value)
        if dy==0 and abs(dx)==scale/N:counts['x']+=1;wrong+=int(g!=gx)
        elif dx==0 and abs(dy)==F(1,N):counts['y']+=1;wrong+=int(g!=gy)
        else:wrong+=1
    return {'model':frame['model'],'a_exact':str(scale),'Gamma_actual_exact':str(gamma),
      'expected_gx_exact':str(gx),'expected_gy_exact':str(gy),'expected_gx_over_gy_exact':str(1/(scale*scale)),
      'directional_face_counts':counts,'wrong_faces':wrong,'constitutive_inputs_preserved':valid,
      'all_directional_conductances_exact':valid and wrong==0 and counts=={'x':N*N,'y':N*N}}
