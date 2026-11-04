"""Exact point-defined periodic Voronoi faces; no material geometry or scalar inputs."""
from dataclasses import dataclass
from fractions import Fraction as F
import hashlib
import math
import numpy as np


def frac(v):return F(float(v))
def cross(a,b):return a[0]*b[1]-a[1]*b[0]
def dot(a,b):return a[0]*b[0]+a[1]*b[1]
def sub(a,b):return (a[0]-b[0],a[1]-b[1])


def enclosure(values):
    """Exact dyadic enclosure of a rational sum, without building its giant LCM."""
    lo=hi=F(0);display=[]
    for q in values:
        v=float(q)
        if not math.isfinite(v):raise ArithmeticError('nonfinite rational-to-float conversion')
        display.append(v)
        if q==0:continue
        lower=F(math.nextafter(v,-math.inf));upper=F(math.nextafter(v,math.inf))
        if not lower<=q<=upper:raise ArithmeticError('rational conversion enclosure failed')
        lo+=lower;hi+=upper
    return {'lower_exact':str(lo),'upper_exact':str(hi),'raw_float_fsum':math.fsum(display)}


def clip(poly,v,label):
    """CCW vertices paired with the label of the incoming segment."""
    d2=dot(v,v)
    if d2<=0:raise ValueError('duplicate periodic point')
    values=[2*dot(v,p)-d2 for p,_ in poly]
    if max(values)<=0:return poly
    if min(values)>=0:raise ArithmeticError('seed-containing cell unexpectedly empty')
    out=[]
    def append(p,tag):
        if not out or out[-1][0]!=p:out.append((p,tag))
    for k,(q,tag) in enumerate(poly):
        p=poly[k-1][0];hp=values[k-1];hq=values[k]
        pin=hp<=0;qin=hq<=0
        if pin and qin:append(q,tag)
        elif pin:
            t=hp/(hp-hq);pnew=(p[0]+t*(q[0]-p[0]),p[1]+t*(q[1]-p[1]))
            append(pnew,tag)
        elif qin:
            t=hp/(hp-hq);pnew=(p[0]+t*(q[0]-p[0]),p[1]+t*(q[1]-p[1]))
            append(pnew,label);append(q,tag)
    if len(out)>1 and out[0][0]==out[-1][0]:
        out[0]=(out[0][0],out[-1][1]);out.pop()
    if len(out)<3:raise ArithmeticError('nonpositive cell after exact clipping')
    return out


def polygon_moments(poly):
    twice=F(0);mx=my=trace=F(0)
    for k,(q,_) in enumerate(poly):
        p=poly[k-1][0];z=cross(p,q);twice+=z
        mx+=(p[0]+q[0])*z;my+=(p[1]+q[1])*z
        trace+=(p[0]*p[0]+p[0]*q[0]+q[0]*q[0]+p[1]*p[1]+p[1]*q[1]+q[1]*q[1])*z
    area=twice/2
    if area<=0:raise ArithmeticError('nonpositive oriented area, no absolute-value repair')
    return area,(mx/(6*area),my/(6*area)),trace/(12*area)


@dataclass(frozen=True)
class PeriodicFaces:
    ids: tuple
    points: tuple
    rho: tuple
    periods: tuple
    D: F
    thickness: F
    polygons: tuple
    areas: tuple
    centroids: tuple
    second_moments: tuple
    search_radii: tuple
    edges: tuple
    image_shifts: tuple
    face_endpoints: tuple
    ratios: tuple
    displacements: tuple
    conductances: tuple
    diagnostics: dict


def build_periodic_faces(ids,positions,rho,D,periods=(1,1),thickness=1,progress=None):
    ident=np.asarray(ids);x=np.asarray(positions,dtype=float);density=np.asarray(rho,dtype=float)
    if ident.ndim!=1 or ident.dtype.kind not in 'iu':raise ValueError('one-dimensional integer identities required')
    n=len(ident)
    if len(set(map(int,ident)))!=n:
        raise ValueError('unique integer IDs required')
    if n<4 or x.shape!=(n,2) or density.shape!=(n,):raise ValueError('complete point/density arrays required')
    if not np.isfinite(x).all() or not np.isfinite(density).all() or np.any(density<=0):raise ValueError('finite points and all-positive density required')
    if len(periods)!=2:raise ValueError('two periodic lengths required')
    if any(isinstance(v,(bool,np.bool_)) or not math.isfinite(float(v)) or float(v)<=0 for v in [*periods,thickness]):raise ValueError('positive finite geometry required')
    if isinstance(D,(bool,np.bool_)) or not math.isfinite(float(D)) or D<0:raise ValueError('finite nonnegative D required')
    box=tuple(map(frac,periods));b=frac(thickness);diffusion=frac(D);point=tuple(tuple(map(frac,v)) for v in x)
    if len(set(point))!=n or any(not 0<=p[k]<box[k] for p in point for k in range(2)):
        raise ValueError('distinct original points in periodic box required; no wrapping or jitter repair')
    ids=tuple(map(int,ident));r=tuple(map(frac,density))
    count=max(1,(n+3)//4);B=math.isqrt(count)
    if B*B<count:B+=1
    B=max(2,B);binwidth=(box[0]/B,box[1]/B);bins={};point_bins=[]
    for i,p in enumerate(point):
        key=(int(p[0]//binwidth[0]),int(p[1]//binwidth[1]));point_bins.append(key);bins.setdefault(key,[]).append(i)
    def vector(i,j,sx,sy):return (point[j][0]+sx*box[0]-point[i][0],point[j][1]+sy*box[1]-point[i][1])
    def nearest(i,j):
        sx=-int(((point[j][0]-point[i][0])/box[0]+F(1,2))//1)
        sy=-int(((point[j][1]-point[i][1])/box[1]+F(1,2))//1)
        return sx,sy,vector(i,j,sx,sy)
    cells=[];areas=[];centroids=[];seconds=[];radii=[];directed={};candidate_counts=[]
    for i,p in enumerate(point):
        lx,ly=box;poly=[((-lx/2,-ly/2),(i,-1,0)),((lx/2,-ly/2),(i,0,-1)),
            ((lx/2,ly/2),(i,1,0)),((-lx/2,ly/2),(i,0,1))]
        pool=set();bx,by=point_bins[i]
        for ring in range(1,B+1):
            for a in range(bx-ring,bx+ring+1):
                for c in range(by-ring,by+ring+1):pool.update(bins.get((a%B,c%B),()))
            pool.discard(i)
            if len(pool)>=min(8,n-1) or ring>=B:break
        ordered=[]
        for j in pool:
            sx,sy,v=nearest(i,j);ordered.append((float(dot(v,v)),ids[j],j,sx,sy,v))
        ordered.sort(key=lambda z:(z[0],z[1],z[3],z[4]));used=set()
        for _,_,j,sx,sy,v in ordered[:8]:poly=clip(poly,v,(j,sx,sy));used.add((j,sx,sy))
        radius=max(abs(q[0])+abs(q[1]) for q,_ in poly);radii.append(radius)
        lower=[int((p[k]-2*radius)//binwidth[k]) for k in range(2)]
        upper=[int((p[k]+2*radius)//binwidth[k]) for k in range(2)]
        candidates=[]
        for a in range(lower[0],upper[0]+1):
            for c in range(lower[1],upper[1]+1):
                for j in bins.get((a%B,c%B),()):
                    sx,sy=a//B,c//B;tag=(j,sx,sy)
                    if j==i or tag in used:continue # all own images are already enforced by the initial box
                    v=vector(i,j,sx,sy)
                    if dot(v,v)>4*radius*radius:continue
                    candidates.append((ids[j],sx,sy,j,v))
        candidates.sort(key=lambda z:(z[0],z[1],z[2]));candidate_counts.append(len(candidates)+len(used))
        for _,sx,sy,j,v in candidates:poly=clip(poly,v,(j,sx,sy))
        start=min(range(len(poly)),key=lambda k:poly[k][0]);poly=poly[start:]+poly[:start]
        area,centroid,second=polygon_moments(poly);pyramid=F(0)
        for k,(q,tag) in enumerate(poly):
            prev=poly[k-1][0];j,sx,sy=tag
            if j==i:raise ValueError('active self-image face unsupported by this dense-cloud adapter')
            v=vector(i,j,sx,sy);d2=dot(v,v);edge=sub(q,prev)
            if 2*dot(v,q)!=d2 or 2*dot(v,prev)!=d2:raise ArithmeticError('face label not on exact bisector')
            ratio=cross(v,edge)/d2
            if ratio<=0:raise ArithmeticError('nonpositive face, no clipping or absolute-value repair')
            if dot(v,edge)!=0:raise ArithmeticError('face not exactly orthogonal')
            pyramid+=ratio*d2/4
            world=tuple(((p[0]+z[0])%box[0],(p[1]+z[1])%box[1]) for z in [prev,q])
            if ids[i]<ids[j]:key=(i,j,sx,sy);canonical=world
            else:key=(j,i,-sx,-sy);canonical=(world[1],world[0])
            entry=(i,ratio,canonical)
            if key in directed:
                old=directed[key]
                if old[3] or old[0]==i or old[1]!=ratio or old[2]!=canonical:raise ArithmeticError('unequal or duplicate opposite faces; no averaging')
                directed[key]=(old[0],old[1],old[2],True)
            else:directed[key]=(i,ratio,canonical,False)
        if pyramid!=area:raise ArithmeticError('exact cell pyramid-area identity failed')
        cells.append(tuple(poly));areas.append(area);centroids.append(centroid);seconds.append(second)
        if progress is not None and ((i+1)%128==0 or i+1==n):progress('exact_cells',i+1,n)
    keys=sorted(directed,key=lambda z:(ids[z[0]],ids[z[1]],z[2],z[3]))
    if any(not directed[k][3] for k in keys):raise ArithmeticError('missing opposite periodic face')
    area_interval=enclosure(areas);domain=box[0]*box[1]
    if not F(area_interval['lower_exact'])<=domain<=F(area_interval['upper_exact']):raise ArithmeticError('raw periodic area enclosure misses domain; no normalization')
    ratios=tuple(directed[k][1] for k in keys);g=[];disp=[]
    for (i,j,sx,sy),ratio in zip(keys,ratios):
        gi=r[i]*diffusion;gj=r[j]*diffusion
        g.append(F(0) if diffusion==0 else b*ratio*2*gi*gj/(gi+gj));disp.append(vector(i,j,sx,sy))
    diagnostics={'bin_count_per_axis':B,'provisional_seed_neighbors':8,'candidate_constraints_total':sum(candidate_counts),
        'maximum_candidates_per_cell':max(candidate_counts),'maximum_faces_per_cell':max(map(len,cells)),
        'all_opposite_faces_and_bisectors_exact':True,'all_local_polygon_pyramid_areas_exact':True,
        'raw_total_area_enclosure':area_interval,'domain_area_exact':str(domain),
        'random_jitter_or_coordinate_change':False,'areas_or_mass_or_density_normalized':False,
        'positive_faces_removed':0,'full_pair_distance_matrix_constructed':False}
    return PeriodicFaces(ids,point,r,box,diffusion,b,tuple(cells),tuple(areas),tuple(centroids),tuple(seconds),tuple(radii),
        tuple((k[0],k[1]) for k in keys),tuple((k[2],k[3]) for k in keys),tuple(directed[k][2] for k in keys),
        ratios,tuple(disp),tuple(g),diagnostics)


def coefficient_packet(graph):
    all_vertices=set()
    for i,poly in enumerate(graph.polygons):
        p=graph.points[i]
        for q,_ in poly:all_vertices.add(((p[0]+q[0])%graph.periods[0],(p[1]+q[1])%graph.periods[1]))
    vertices=sorted(all_vertices);lookup={v:k for k,v in enumerate(vertices)};cell_vertices=[];labels=[]
    for i,poly in enumerate(graph.polygons):
        p=graph.points[i];cell_vertices.append([lookup[((p[0]+q[0])%graph.periods[0],(p[1]+q[1])%graph.periods[1])] for q,_ in poly])
        labels.append([list(tag) for _,tag in poly])
    return {'ids':list(graph.ids),'periods_exact':list(map(str,graph.periods)),'D_exact':str(graph.D),
        'thickness_exact':str(graph.thickness),'vertices_exact':[[str(x),str(y)] for x,y in vertices],
        'cell_vertex_indices':cell_vertices,'cell_incoming_labels':labels,'cell_areas_exact':list(map(str,graph.areas)),
        'initial_search_radius_L1_exact':list(map(str,graph.search_radii)),'edges':[list(v) for v in graph.edges],
        'image_shifts':[list(v) for v in graph.image_shifts],'face_vertex_indices':[[lookup[a],lookup[b]] for a,b in graph.face_endpoints],
        'face_over_distance_exact':list(map(str,graph.ratios)),'conductances_exact':list(map(str,graph.conductances)),
        'geometry_diagnostics':graph.diagnostics,'builder_received_mass_or_scalar_or_reference':False}


def exact_action(graph,mass,cbar):
    m=np.asarray(mass,dtype=float);u=np.asarray(cbar,dtype=float)
    if m.shape!=(len(graph.ids),) or u.shape!=m.shape or not np.isfinite(m).all() or np.any(m<=0) or not np.isfinite(u).all():
        raise ValueError('full finite state and positive actual masses required')
    mm=tuple(map(frac,m));state=tuple(map(frac,u));flux=[F(0)]*len(m)
    for (i,j),g in zip(graph.edges,graph.conductances):
        f=g*(state[j]-state[i]);flux[i]+=f;flux[j]-=f
    return tuple(v/a for v,a in zip(flux,mm))


def rate_digest(rates):
    h=hashlib.sha256()
    for q in rates:h.update((hex(q.numerator)+'/'+hex(q.denominator)+'\n').encode())
    return h.hexdigest()
