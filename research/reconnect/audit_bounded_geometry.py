"""Independent bounded-strip bisectors, external faces and all-cell constraints."""
from fractions import Fraction as F
import numpy as np


def dot(a,b):return a[0]*b[0]+a[1]*b[1]
def cross(a,b):return a[0]*b[1]-a[1]*b[0]


def audit_geometry(packet,positions,rho,ids,D,x_bounds,period_y=1):
    points=tuple(tuple(F(float(a)) for a in p) for p in positions);rr=tuple(F(float(a)) for a in rho);ident=list(map(int,ids));n=len(ident)
    l,h=map(lambda v:F(float(v)),x_bounds);Ly=F(float(period_y));diff=F(float(D))
    assert packet['ids']==ident and packet['positions_exact']==[list(map(str,p)) for p in points] and packet['rho_exact']==list(map(str,rr))
    assert list(map(F,packet['x_bounds_exact']))==[l,h] and F(packet['period_y_exact'])==Ly and F(packet['D_exact'])==diff
    directed={};walls=[];areas=[];candidates=0
    def vector(i,j,sy):return (points[j][0]-points[i][0],points[j][1]+sy*Ly-points[i][1])
    def plane(i,tag):
        j,sx,sy=tag;assert sx==0
        if j==-1:return (F(-1),F(0)),points[i][0]-l
        if j==-2:return (F(1),F(0)),h-points[i][0]
        assert j!=i
        v=vector(i,j,sy);return v,dot(v,v)/2
    for i,tags in enumerate(packet['cell_incoming_labels']):
        poly=[]
        for k,tag in enumerate(tags):
            a,b=plane(i,tag);c,d=plane(i,tags[(k+1)%len(tags)]);det=cross(a,c);assert det
            poly.append(((b*c[1]-a[1]*d)/det,(a[0]*d-b*c[0])/det))
        assert [[str(x),str(y)] for x,y in poly]==packet['cell_vertices_relative_exact'][i]
        area=sum((cross(poly[k-1],q) for k,q in enumerate(poly)),F(0))/2;assert area>0 and area==F(packet['cell_areas_exact'][i]);areas.append(area)
        r2=max(dot(q,q) for q in poly);assert r2==F(packet['final_circumradius_squared_exact'][i]) and 4*r2<Ly*Ly/4
        assert all(l<=points[i][0]+q[0]<=h for q in poly)
        for j in range(n):
            if j==i:continue
            sy=-int(((points[j][1]-points[i][1])/Ly+F(1,2))//1);v=vector(i,j,sy);d2=dot(v,v)
            if d2>4*r2:continue
            assert all(2*dot(v,q)<=d2 for q in poly);candidates+=1
        for k,(q,tag) in enumerate(zip(poly,tags)):
            prev=poly[k-1];a,b=plane(i,tag);assert dot(a,q)==dot(a,prev)==b
            ends=tuple((points[i][0]+v[0],(points[i][1]+v[1])%Ly) for v in (prev,q))
            if tag[0]<0:
                walls.append({'node':i,'side':'left' if tag[0]==-1 else 'right','length_exact':str(abs(q[1]-prev[1])),
                  'endpoints_exact':[list(map(str,v)) for v in ends],'normal_scalar_flux_exact':'0'});continue
            ratio=cross(a,(q[0]-prev[0],q[1]-prev[1]))/dot(a,a);assert ratio>0
            key=(i,tag[0],tag[2]);assert key not in directed;directed[key]=(ratio,ends)
    assert walls==packet['external_faces'] and sum(areas,F(0))==(h-l)*Ly
    assert sum((F(v['length_exact']) for v in walls if v['side']=='left'),F(0))==Ly
    assert sum((F(v['length_exact']) for v in walls if v['side']=='right'),F(0))==Ly
    edges=[];gg=[];degree=[F(0)]*n;seen=set()
    for k,((i,j),(sx,sy)) in enumerate(zip(packet['edges'],packet['image_shifts'])):
        assert sx==0 and ident[i]<ident[j] and (i,j,sy) not in seen;seen.add((i,j,sy))
        ratio,ends=directed[i,j,sy];other=directed[j,i,-sy];assert ratio==other[0] and ends==other[1][::-1]
        assert F(packet['face_over_distance_exact'][k])==ratio and packet['face_endpoints_exact'][k]==[list(map(str,v)) for v in ends]
        g=F(0) if diff==0 else ratio/(1/(2*rr[i]*diff)+1/(2*rr[j]*diff));assert g==F(packet['conductances_exact'][k]) and g>=0
        edges.append((i,j));gg.append(g);degree[i]+=g;degree[j]+=g
    assert 2*len(edges)==len(directed)
    return np.array(edges,dtype=np.int64),gg,degree,{'all_vertices_rebuilt_from_boundary_lines_and_bisectors':True,'all_cell_world_halfplanes_verified':True,
      'all_non_nearest_y_images_excluded':True,'nearest_candidate_halfplanes_checked':candidates,'exact_domain_area':str(sum(areas,F(0))),
      'left_external_faces':sum(v['side']=='left' for v in walls),'right_external_faces':sum(v['side']=='right' for v in walls),
      'shared_faces':len(edges),'external_scalar_flux_exact':'0','x_periodic':False,'mass_scalar_reference_or_builder_called':False}
