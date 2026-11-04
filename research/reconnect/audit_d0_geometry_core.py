"""Independent bisector/reciprocal-face audit for exact D=0; positive-D audit unchanged."""
from fractions import Fraction as F
import numpy as np
from audit_shear_voronoi_fv import fixed_binary_enclosure


def dot(a,b):return a[0]*b[0]+a[1]*b[1]
def cross(a,b):return a[0]*b[1]-a[1]*b[0]
def sub(a,b):return (a[0]-b[0],a[1]-b[1])


def audit_zero_geometry(packet,positions,rho,ids,probe_ids):
    points=tuple(tuple(F(float(x)) for x in row) for row in positions)
    density=tuple(F(float(x)) for x in rho)
    periods=tuple(map(F,packet['periods_exact']));D=F(packet['D_exact']);b=F(packet['thickness_exact'])
    assert list(map(int,ids))==packet['ids'] and D==0 and b==1 and periods==(1,1) and min(density)>0
    vertices=tuple(tuple(map(F,row)) for row in packet['vertices_exact'])
    directed={};areas=[];selected={}
    def vector(i,label):
        j,sx,sy=label
        return (points[j][0]+sx*periods[0]-points[i][0],points[j][1]+sy*periods[1]-points[i][1])
    for i,(vi,labels) in enumerate(zip(packet['cell_vertex_indices'],packet['cell_incoming_labels'])):
        assert len(vi)==len(labels)>=3
        poly=[]
        for k,label in enumerate(labels):
            a=vector(i,label);v=vector(i,labels[(k+1)%len(labels)])
            ca=dot(a,a)/2;cv=dot(v,v)/2;det=cross(a,v);assert det
            q=((ca*v[1]-a[1]*cv)/det,(a[0]*cv-ca*v[0])/det)
            world=((points[i][0]+q[0])%periods[0],(points[i][1]+q[1])%periods[1])
            assert world==vertices[vi[k]]
            poly.append(q)
        twice=pyramid=F(0)
        for k,(q,label) in enumerate(zip(poly,labels)):
            prev=poly[k-1];nxt=poly[(k+1)%len(poly)]
            assert cross(sub(q,prev),sub(nxt,q))>0
            v=vector(i,label);d2=dot(v,v);edge=sub(q,prev)
            assert d2>0 and dot(v,edge)==0 and 2*dot(v,prev)==2*dot(v,q)==d2
            ratio=cross(v,edge)/d2;assert ratio>0
            twice+=cross(prev,q);pyramid+=ratio*d2/4
            tag=(i,*label);assert tag not in directed
            directed[tag]=(ratio,(vi[k-1],vi[k]))
        area=twice/2;assert area>0 and area==pyramid==F(packet['cell_areas_exact'][i])
        areas.append(area)
        if int(ids[i]) in probe_ids:selected[int(ids[i])]=(i,poly)
    lo,hi=fixed_binary_enclosure(areas)
    assert lo<=periods[0]*periods[1]<=hi
    original=packet['geometry_diagnostics']['raw_total_area_enclosure']
    assert F(original['lower_exact'])<=hi and lo<=F(original['upper_exact'])
    g=[];edges=[];degree=[F(0)]*len(ids);face_keys=set()
    for k,((i,j),(sx,sy)) in enumerate(zip(packet['edges'],packet['image_shifts'])):
        assert ids[i]<ids[j] and (i,j,sx,sy) not in face_keys
        face_keys.add((i,j,sx,sy))
        first=directed[(i,j,sx,sy)];other=directed[(j,i,-sx,-sy)]
        assert first[0]==other[0]==F(packet['face_over_distance_exact'][k])
        assert first[1]==tuple(packet['face_vertex_indices'][k])==other[1][::-1]
        # Independent sum of two half-resistances; not the builder's harmonic product.
        conductance=F(0)  # exact zero-diffusion limit, no division by zero
        assert conductance==F(packet['conductances_exact'][k])
        g.append(conductance);edges.append((i,j));degree[i]+=conductance;degree[j]+=conductance
    assert 2*len(edges)==len(directed)
    probes=[]
    for ident in probe_ids:
        i,poly=selected[ident];R=max(abs(q[0])+abs(q[1]) for q in poly)
        assert R<min(periods)/4
        possible=0;worst=None
        for j,p in enumerate(points):
            if j==i:continue
            delta=[]
            for k,L in enumerate(periods):
                q=p[k]-points[i][k]
                if q>L/2:q-=L
                elif q<-L/2:q+=L
                delta.append(q)
            if abs(delta[0])>2*R or abs(delta[1])>2*R:continue
            v=tuple(delta);d2=dot(v,v);possible+=1
            residual=max(2*dot(v,q)-d2 for q in poly);assert residual<=0
            worst=residual if worst is None else max(worst,residual)
        probes.append({'id':ident,'other_points_checked':len(ids)-1,'possible_cutters_tested':possible,
            'maximum_signed_halfplane_residual_exact':str(worst),'final_radius_L1_exact':str(R)})
    result={'all_vertices_independently_rebuilt_from_bisectors':True,
        'all_face_counterparts_and_zero_diffusion_limit_exact':True,
        'shared_geometric_faces':len(edges),'shared_zero_conductances':len(g),'zero_diffusion_limit_without_division_by_zero':True,'area_sum_256bit_enclosure':{'lower_exact':str(lo),'upper_exact':str(hi)},
        'fixed_all_point_halfplane_probes':{'fixed_ids':probe_ids,'all_selected_cells_passed':True,'records':probes}}
    return np.array(edges,dtype=np.int64),g,degree,result

