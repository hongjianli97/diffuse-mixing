"""Independent periodic-bin density, fixed lifts, exact rings and rate verification."""
from fractions import Fraction as F
import math
import numpy as np


def bin_density(ids,x,m,h):
    # Bin width >= support, so the nine neighboring bins exhaust candidates.
    B=math.floor(1/(2*h));buckets={}
    for j,p in enumerate(x):buckets.setdefault(tuple(np.floor(p*B).astype(int)),[]).append(j)
    result=[];counts=[]
    for i,p in enumerate(x):
        a,b=np.floor(p*B).astype(int);js=set()
        for dx in (-1,0,1):
            for dy in (-1,0,1):js.update(buckets.get(((a+dx)%B,(b+dy)%B),()))
        assert i in js;terms=[]
        for j in sorted(js,key=lambda j:int(ids[j])):
            dx=float(x[j,0]-p[0]);dy=float(x[j,1]-p[1]);dx-=round(dx);dy-=round(dy)
            q=math.hypot(dx,dy)/h
            if q<2:terms.append(float(m[j])*9/(4*math.pi*h*h)*(1-q/2)**6*(1+3*q+35*q*q/12))
        result.append(math.fsum(terms));counts.append(len(terms))
    return np.array(result),counts


def verify_lifts(labels,times,x,w):
    assert w.dtype.kind in 'iu' and w.shape==x.shape and np.array_equal(x[0],labels) and not w[0].any()
    for k in range(1,len(times)):
        for i in range(len(labels)):
            for a in (0,1):
                d=F(float(x[k,i,a]))-F(float(x[k-1,i,a]));assert abs(d)!=F(1,2)
                jump=-1 if d>F(1,2) else (1 if d < -F(1,2) else 0)
                assert w[k,i,a]==w[k-1,i,a]+jump
    # The known map is an independent coordinate check, never adapter input.
    largest=0.
    for k,t in enumerate(times):
        expected=labels.copy();expected[:,0]+=(.25*t/.05)/(2*np.pi)*np.sin(2*np.pi*labels[:,0])+.2*t
        expected[:,1]-=.1*t
        difference=float(np.max(abs(x[k]+w[k]-expected)));largest=max(largest,difference)
        assert difference<=64*np.finfo(float).eps
    return {'integer_winding_verified':True,'maximum_known_map_coordinate_difference':largest,
            'positive_windings_final':np.sum(w[-1]>0,axis=0).tolist(),'negative_windings_final':np.sum(w[-1]<0,axis=0).tolist()}


def verify_ring_density(x,w,rho0,rings,initial_areas,rho,areas):
    lifted=[(F(float(p[0]))+int(q[0]),F(float(p[1]))+int(q[1])) for p,q in zip(x,w)]
    for i,order in enumerate(rings):
        assert len(order)==4 and len({v[0] for v in order})==4
        v=[(lifted[j][0]+sx-lifted[i][0],lifted[j][1]+sy-lifted[i][1]) for j,sx,sy in order]
        # Strict convexity and oriented fan areas prove simplicity for this case.
        cross=lambda a,b:a[0]*b[1]-a[1]*b[0]
        fan=[cross(a,b) for a,b in zip(v,v[1:]+v[:1])]
        assert min(fan)>0
        for j in range(4):
            a,b,c=v[j],v[(j+1)%4],v[(j+2)%4]
            assert cross((b[0]-a[0],b[1]-a[1]),(c[0]-b[0],c[1]-b[1]))>0
        area=sum(fan,F(0))/2
        assert area==F(areas[i])>0 and F(initial_areas[i])>0
        assert float(F(float(rho0[i]))*F(initial_areas[i])/area)==rho[i]


def verify_coefficients(original,packet,ids,rho):
    assert packet['ids']==ids.tolist() and len(rho)==len(ids) and np.isfinite(rho).all() and rho.min()>0
    assert {k:v for k,v in packet.items() if k!='conductances_exact'}=={k:v for k,v in original.items() if k!='conductances_exact'}
    D=F(original['D_exact']);identical=True
    for (i,j),ratio,g,old in zip(packet['edges'],packet['face_over_distance_exact'],packet['conductances_exact'],original['conductances_exact']):
        resistance=1/(2*F(float(rho[i]))*D)+1/(2*F(float(rho[j]))*D)
        expected=F(original['thickness_exact'])*F(ratio)/resistance
        assert expected==F(g)>0;identical=identical and g==old
    return identical
