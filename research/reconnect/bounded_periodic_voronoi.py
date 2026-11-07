"""Opt-in bounded-x/periodic-y exact Euclidean point Voronoi; no material faces."""
from fractions import Fraction as F
import math
import numpy as np
from scipy.spatial import cKDTree
from periodic_voronoi_fv import clip,polygon_moments,dot,cross,sub,enclosure

LEFT=(-1,0,0);RIGHT=(-2,0,0)


def build_bounded_faces(ids,positions,rho,D,x_bounds,period_y=1,progress=None):
    ident=np.asarray(ids);x=np.asarray(positions,dtype=float);r=np.asarray(rho,dtype=float);n=len(ident)
    if ident.ndim!=1 or ident.dtype.kind not in 'iu' or len(np.unique(ident))!=n or n<4 or x.shape!=(n,2) or r.shape!=(n,):raise ValueError('complete unique integer IDs/points/densities required')
    if not np.isfinite(x).all() or not np.isfinite(r).all() or np.any(r<=0):raise ValueError('finite points and positive densities required')
    if len(x_bounds)!=2 or any(isinstance(a,(bool,np.bool_)) or not math.isfinite(float(a)) for a in x_bounds):raise ValueError('finite external x bounds required')
    if isinstance(period_y,(bool,np.bool_)) or not math.isfinite(float(period_y)) or period_y<=0 or isinstance(D,(bool,np.bool_)) or not math.isfinite(float(D)) or D<0:raise ValueError('positive period and nonnegative D required')
    l,h=map(lambda v:F(float(v)),x_bounds);Ly=F(float(period_y));diff=F(float(D));points=tuple(tuple(F(float(a)) for a in p) for p in x);rr=tuple(F(float(a)) for a in r);ident=tuple(map(int,ident))
    if l>=h or len(set(points))!=n or any(not(l<p[0]<h and 0<=p[1]<Ly) for p in points):raise ValueError('distinct interior points in original strip required; no repair')
    def vector(i,j,sy):return (points[j][0]-points[i][0],points[j][1]+sy*Ly-points[i][1])
    image=np.concatenate([x+np.array([0.,float(sy*Ly)]) for sy in (-1,0,1)]);tree=cKDTree(image)
    cells=[];areas=[];radii=[];directed={};walls=[];checks=0
    for i,p in enumerate(points):
        poly=[((l-p[0],-Ly/2),LEFT),((h-p[0],-Ly/2),(i,0,-1)),((h-p[0],Ly/2),RIGHT),((l-p[0],Ly/2),(i,0,1))]
        _,near=tree.query(x[i],k=min(10,3*n));used=set()
        for item in np.atleast_1d(near):
            j=int(item)%n;sy=int(item)//n-1
            if j==i and sy==0:continue
            poly=clip(poly,vector(i,j,sy),(j,0,sy));used.add((j,sy))
        R=max(abs(q[0])+abs(q[1]) for q,_ in poly)
        for j in range(n):
            if j==i:continue
            sy=-int(((points[j][1]-p[1])/Ly+F(1,2))//1)
            if (j,sy) in used:continue
            v=vector(i,j,sy);checks+=1
            if dot(v,v)>4*R*R:continue
            poly=clip(poly,v,(j,0,sy))
        start=min(range(len(poly)),key=lambda k:poly[k][0]);poly=poly[start:]+poly[:start]
        area,_,_=polygon_moments(poly);R2=max(dot(q,q) for q,_ in poly)
        if not 4*R2<Ly*Ly/4:raise ValueError('nonnearest periodic image exclusion unavailable; no image truncation repair')
        for k,(q,tag) in enumerate(poly):
            prev=poly[k-1][0]
            if tag in (LEFT,RIGHT):
                boundary=l if tag==LEFT else h
                if p[0]+q[0]!=boundary or p[0]+prev[0]!=boundary:raise ArithmeticError('external face moved')
                walls.append({'node':i,'side':'left' if tag==LEFT else 'right','length_exact':str(abs(q[1]-prev[1])),
                  'endpoints_exact':[[str(p[0]+a[0]),str((p[1]+a[1])%Ly)] for a in (prev,q)],'normal_scalar_flux_exact':'0'});continue
            j,_,sy=tag
            if j==i:raise ValueError('active selfimage face unsupported in dense strip mode')
            v=vector(i,j,sy);d2=dot(v,v);edge=sub(q,prev);ratio=cross(v,edge)/d2
            if 2*dot(v,q)!=d2 or 2*dot(v,prev)!=d2 or ratio<=0:raise ArithmeticError('invalid ordinary world face')
            ends=tuple((p[0]+a[0],(p[1]+a[1])%Ly) for a in (prev,q))
            if ident[i]<ident[j]:key=(i,j,sy);canonical=ends
            else:key=(j,i,-sy);canonical=ends[::-1]
            if key in directed:
                old=directed[key]
                if old[0]==i or old[1]!=ratio or old[2]!=canonical or old[3]:raise ArithmeticError('reciprocal internal face mismatch')
                directed[key]=(old[0],old[1],old[2],True)
            else:directed[key]=(i,ratio,canonical,False)
        cells.append(poly);areas.append(area);radii.append(R2)
        if progress and ((i+1)%64==0 or i+1==n):progress('bounded-exact-cells',i+1,n)
    keys=sorted(directed,key=lambda k:(ident[k[0]],ident[k[1]],k[2]))
    if any(not directed[key][3] for key in keys):raise ArithmeticError('missing reciprocal internal face')
    area_sum=enclosure(areas)
    if not F(area_sum['lower_exact'])<=(h-l)*Ly<=F(area_sum['upper_exact']):raise ArithmeticError('raw area sum misses external domain')
    gg=[F(0) if diff==0 else directed[i,j,sy][1]/(1/(2*rr[i]*diff)+1/(2*rr[j]*diff)) for i,j,sy in keys]
    return {'ids':list(ident),'positions_exact':[list(map(str,p)) for p in points],'rho_exact':list(map(str,rr)),'D_exact':str(diff),
      'x_bounds_exact':[str(l),str(h)],'period_y_exact':str(Ly),'cell_vertices_relative_exact':[[list(map(str,q)) for q,_ in poly] for poly in cells],
      'cell_incoming_labels':[[list(tag) for _,tag in poly] for poly in cells],'cell_areas_exact':list(map(str,areas)),
      'final_circumradius_squared_exact':list(map(str,radii)),'edges':[[i,j] for i,j,sy in keys],'image_shifts':[[0,sy] for i,j,sy in keys],
      'face_over_distance_exact':[str(directed[key][1]) for key in keys],'face_endpoints_exact':[[list(map(str,p)) for p in directed[key][2]] for key in keys],
      'conductances_exact':list(map(str,gg)),'external_faces':walls,'geometry_diagnostics':{'raw_area_sum_enclosure':area_sum,'domain_area_exact':str((h-l)*Ly),
        'all_point_nearest_image_radius_search_checks':checks,'all_non_nearest_images_excluded_by_final_radius':True,'provisional_KD_neighbours_only':True,
        'x_periodic':False,'external_zero_flux':True,'positive_internal_faces_removed':0,'points_areas_or_densities_repaired':False},
      'builder_received_mass_c0_reference_internal_material_faces_or_F_C':False}
