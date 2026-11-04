"""Opt-in exact sliding torus; fractional bins, world Euclidean Voronoi faces."""
from dataclasses import dataclass
from fractions import Fraction as F
import math
import numpy as np
from oriented_periodic_boundary import rational,lossless_float
from periodic_voronoi_fv import clip,polygon_moments,dot,cross,sub,enclosure,build_periodic_faces,coefficient_packet


@dataclass(frozen=True)
class SlidingPeriodicDomain:
    shear: F

    def __post_init__(self):
        s=rational(self.shear)
        if not 0<=s<=F(1,4):raise ValueError('only the declared exact sliding family 0<=s<=1/4 is supported')
        object.__setattr__(self,'shear',s)

    def push(self,q):return q[0]+self.shear*q[1],q[1]
    def fractions(self,p):return p[0]-self.shear*p[1],p[1]
    def canonical(self,p):
        a,b=self.fractions(p);return self.push((a%1,b%1))
    def packet(self):return {'origin_exact':['0','0'],'shear_exact':str(self.shear),
      'basis_columns_exact':[['1','0'],[str(self.shear),'1']]}


def build_sliding_faces(ids,world_positions_exact,rho,D,boundary,progress=None):
    if not isinstance(boundary,SlidingPeriodicDomain):raise ValueError('given exact sliding boundary required')
    ident=np.asarray(ids);density=np.asarray(rho,dtype=float);points=tuple(tuple(map(rational,p)) for p in world_positions_exact);n=len(points)
    if ident.shape!=(n,) or ident.dtype.kind not in 'iu' or n<4 or len(set(map(int,ident)))!=n:raise ValueError('complete unique integer IDs required')
    if any(len(p)!=2 for p in points) or len(set(points))!=n:raise ValueError('distinct complete world points required')
    if density.shape!=(n,) or not np.isfinite(density).all() or np.any(density<=0):raise ValueError('complete positive point density required')
    if isinstance(D,(bool,np.bool_)) or not math.isfinite(float(D)) or D<0:raise ValueError('finite nonnegative D required')
    for p in points:
        for v in p:lossless_float(v)
    fractional=tuple(boundary.fractions(p) for p in points)
    if any(not 0<=a<1 or not 0<=b<1 for a,b in fractional):raise ValueError('points outside declared fundamental cell; no wrapping repair')
    ident=tuple(map(int,ident));density=tuple(F(float(v)) for v in density);diffusion=F(float(D));s=boundary.shear
    if s==0:
        graph=build_periodic_faces(np.array(ident,dtype=np.int64),np.array([[float(v) for v in p] for p in points]),
          np.array(list(map(float,density))),float(diffusion),(1,1),1,progress)
        packet=coefficient_packet(graph)
        world={key:value for key,value in packet.items() if key not in ('vertices_exact','periods_exact')}
        world.update(boundary.packet());world['points_world_exact']=[list(map(str,p)) for p in points];world['vertices_world_exact']=packet['vertices_exact']
        return packet,world,{'s0_original_rectangle_dispatched':True,'world_euclidean_geometry_preserved':True,
          'fractional_coordinates_only_for_index_and_canonicalization':True,'points_wrapped_jittered_or_fitted':False,
          'mass_c0_reference_internal_faces_or_F_C_received':False}
    count=max(1,(n+3)//4);binside=math.isqrt(count)+(math.isqrt(count)**2<count);binside=max(2,binside)
    bins={};pointbins=[]
    for i,(a,b) in enumerate(fractional):
        key=(int(a*binside),int(b*binside));pointbins.append(key);bins.setdefault(key,[]).append(i)
    def vector(i,j,sx,sy):return points[j][0]+sx+s*sy-points[i][0],points[j][1]+sy-points[i][1]
    cells=[];areas=[];radii=[];finalradii=[];searchboxes=[];directed={};counts=[]
    for i,p in enumerate(points):
        poly=[((F(-2),F(-2)),None),((F(2),F(-2)),None),((F(2),F(2)),None),((F(-2),F(2)),None)]
        used=set()
        for sx in (-1,0,1):
            for sy in (-1,0,1):
                if sx==sy==0:continue
                tag=(i,sx,sy);poly=clip(poly,vector(i,*tag),tag);used.add(tag)
        pool=set();bx,by=pointbins[i]
        for ring in range(1,binside+1):
            for a in range(bx-ring,bx+ring+1):
                for b in range(by-ring,by+ring+1):pool.update(bins.get((a%binside,b%binside),()))
            pool.discard(i)
            if len(pool)>=min(8,n-1):break
        seed=[]
        for j in pool:
            for sx in (-1,0,1):
                for sy in (-1,0,1):
                    v=vector(i,j,sx,sy);seed.append((dot(v,v),ident[j],sx,sy,j,v))
        seed.sort(key=lambda z:z[:4])
        for _,_,sx,sy,j,v in seed[:8]:poly=clip(poly,v,(j,sx,sy));used.add((j,sx,sy))
        radius=max(abs(q[0])+abs(q[1]) for q,_ in poly);bound=2*radius*(1+s);a0,b0=fractional[i]
        lower=(int((a0-bound)*binside//1),int((b0-bound)*binside//1));upper=(int((a0+bound)*binside//1),int((b0+bound)*binside//1))
        candidates=[]
        for a in range(lower[0],upper[0]+1):
            for b in range(lower[1],upper[1]+1):
                sx,sy=a//binside,b//binside
                for j in bins.get((a%binside,b%binside),()):
                    tag=(j,sx,sy)
                    if (j==i and sx==sy==0) or tag in used:continue
                    v=vector(i,j,sx,sy)
                    if dot(v,v)>4*radius*radius:continue
                    candidates.append((ident[j],sx,sy,j,v))
        candidates.sort(key=lambda z:z[:3])
        for _,sx,sy,j,v in candidates:poly=clip(poly,v,(j,sx,sy))
        start=min(range(len(poly)),key=lambda k:poly[k][0]);poly=poly[start:]+poly[:start]
        area,centroid,second=polygon_moments(poly);radius2=max(dot(q,q) for q,_ in poly)
        if 4*radius2*(1+s)**2>=1:raise ValueError('dense-cloud far-image certificate unavailable; no image truncation repair')
        pyramid=F(0)
        for k,(q,tag) in enumerate(poly):
            if tag is None:raise ArithmeticError('provisional box face remains active')
            prev=poly[k-1][0];j,sx,sy=tag
            if j==i:raise ValueError('active self-image face unsupported in this dense-cloud mode')
            v=vector(i,j,sx,sy);d2=dot(v,v);edge=sub(q,prev)
            if 2*dot(v,q)!=d2 or 2*dot(v,prev)!=d2 or dot(v,edge)!=0:raise ArithmeticError('world face is not an exact Euclidean bisector')
            ratio=cross(v,edge)/d2
            if ratio<=0:raise ArithmeticError('nonpositive world face; no clipping or absolute-value repair')
            pyramid+=ratio*d2/4
            endpoints=tuple(boundary.canonical((p[0]+z[0],p[1]+z[1])) for z in (prev,q))
            if ident[i]<ident[j]:key=(i,j,sx,sy);canonical=endpoints
            else:key=(j,i,-sx,-sy);canonical=endpoints[::-1]
            if key in directed:
                old=directed[key]
                if old[3] or old[0]==i or old[1]!=ratio or old[2]!=canonical:raise ArithmeticError('unequal or duplicate reciprocal world face')
                directed[key]=(old[0],old[1],old[2],True)
            else:directed[key]=(i,ratio,canonical,False)
        if area!=pyramid:raise ArithmeticError('world cell pyramid-area identity failed')
        cells.append(tuple(poly));areas.append(area);radii.append(radius);finalradii.append(radius2)
        searchboxes.append([list(lower),list(upper)]);counts.append(len(candidates)+len(used))
        if progress is not None and ((i+1)%128==0 or i+1==n):progress('exact_sliding_cells',i+1,n)
    keys=sorted(directed,key=lambda k:(ident[k[0]],ident[k[1]],k[2],k[3]))
    if any(not directed[k][3] for k in keys):raise ArithmeticError('missing reciprocal periodic face')
    interval=enclosure(areas)
    if not F(interval['lower_exact'])<=1<=F(interval['upper_exact']):raise ArithmeticError('raw area sum misses domain; no normalization')
    vertices=sorted({boundary.canonical((points[i][0]+q[0],points[i][1]+q[1])) for i,poly in enumerate(cells) for q,_ in poly});lookup={v:k for k,v in enumerate(vertices)}
    conductances=[]
    for i,j,sx,sy in keys:
        gi=density[i]*diffusion;gj=density[j]*diffusion;ratio=directed[i,j,sx,sy][1]
        conductances.append(F(0) if diffusion==0 else ratio*2*gi*gj/(gi+gj))
    packet={'ids':list(ident),'D_exact':str(diffusion),'thickness_exact':'1',**boundary.packet(),
      'vertices_exact':[list(map(str,v)) for v in vertices],
      'cell_vertex_indices':[[lookup[boundary.canonical((points[i][0]+q[0],points[i][1]+q[1]))] for q,_ in poly] for i,poly in enumerate(cells)],
      'cell_incoming_labels':[[list(tag) for _,tag in poly] for poly in cells],'cell_areas_exact':list(map(str,areas)),
      'initial_search_radius_L1_exact':list(map(str,radii)),'final_circumradius_squared_exact':list(map(str,finalradii)),
      'fractional_search_bin_boxes':searchboxes,'edges':[[k[0],k[1]] for k in keys],'image_shifts':[[k[2],k[3]] for k in keys],
      'face_vertex_indices':[[lookup[a],lookup[b]] for k in keys for a,b in [directed[k][2]]],
      'face_over_distance_exact':[str(directed[k][1]) for k in keys],'conductances_exact':list(map(str,conductances)),
      'geometry_diagnostics':{'bin_count_per_axis':binside,'provisional_seed_constraints':8,'candidate_constraints_total':sum(counts),
        'maximum_candidates_per_cell':max(counts),'maximum_faces_per_cell':max(map(len,cells)),'all_opposite_faces_and_bisectors_exact':True,
        'raw_total_area_enclosure':interval,'domain_area_exact':'1','world_euclidean_distances_used':True,
        'fractional_coordinates_only_for_index_and_canonicalization':True,'complete_world_radius_search':True,
        'all_dense_cloud_far_image_certificates_passed':True,'random_jitter_or_coordinate_change':False,
        'areas_or_mass_or_density_normalized':False,'positive_faces_removed':0},'builder_received_mass_or_scalar_or_reference':False}
    world={key:value for key,value in packet.items() if key!='vertices_exact'}
    world['points_world_exact']=[list(map(str,p)) for p in points];world['vertices_world_exact']=packet['vertices_exact']
    cert={'s0_original_rectangle_dispatched':False,'world_euclidean_geometry_preserved':True,
      'fractional_coordinates_only_for_index_and_canonicalization':True,'points_wrapped_jittered_or_fitted':False,
      'mass_c0_reference_internal_faces_or_F_C_received':False,'all_dense_cloud_far_image_certificates_passed':True}
    return packet,world,cert
