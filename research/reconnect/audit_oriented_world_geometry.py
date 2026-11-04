"""Independent exact world-space bisectors/periodic faces; never builds a local cloud."""
from fractions import Fraction as F
import numpy as np
from audit_shear_voronoi_fv import fixed_binary_enclosure


def dot(a,b):return a[0]*b[0]+a[1]*b[1]
def cross(a,b):return a[0]*b[1]-a[1]*b[0]
def sub(a,b):return (a[0]-b[0],a[1]-b[1])


def audit_world_geometry(packet,world_input,ids,rho,D,probe_ids):
    boundary=world_input['boundary'];points=tuple(tuple(map(F,p)) for p in world_input['world_positions_exact'])
    origin=tuple(map(F,boundary['origin_exact']));L=tuple(map(F,boundary['periods_exact']))
    B=tuple(tuple(map(F,p)) for p in boundary['basis_columns_exact']);det=cross(B[0],B[1])
    assert min(L)>0 and dot(B[0],B[1])==0 and dot(B[0],B[0])==L[0]**2 and dot(B[1],B[1])==L[1]**2 and det==L[0]*L[1]
    for key in ('origin_exact','periods_exact'):assert tuple(map(F,packet[key]))==tuple(map(F,boundary[key]))
    for key in ('orientation_rows_exact','basis_columns_exact'):assert tuple(tuple(map(F,row)) for row in packet[key])==tuple(tuple(map(F,row)) for row in boundary[key])
    E=tuple(tuple(map(F,row)) for row in boundary['orientation_rows_exact'])
    assert all(E[i][j]==B[j][i]/L[j] for i in range(2) for j in range(2))
    assert tuple(tuple(map(F,p)) for p in packet['points_world_exact'])==points and packet['ids']==list(map(int,ids))
    density=tuple(F(float(v)) for v in rho);diffusion=F(float(D));thickness=F(packet['thickness_exact'])
    assert min(density)>0 and diffusion>0 and F(packet['D_exact'])==diffusion and thickness==1
    vertices=tuple(tuple(map(F,p)) for p in packet['vertices_world_exact'])
    assert len(points)==len(ids)==len(density)==len(packet['cell_vertex_indices'])==len(packet['cell_incoming_labels'])==len(packet['cell_areas_exact'])
    assert len(packet['edges'])==len(packet['image_shifts'])==len(packet['face_vertex_indices'])==len(packet['face_over_distance_exact'])==len(packet['conductances_exact'])
    # Inverse lattice is used only to compare canonical representatives, never to rebuild local geometry.
    def fractions(p):
        q=sub(p,origin);return cross(q,B[1])/det,cross(B[0],q)/det
    def canonical(p):
        q=fractions(p);return tuple(origin[k]+B[0][k]*(q[0]%1)+B[1][k]*(q[1]%1) for k in range(2))
    for p in points+vertices:assert all(0<=q<1 for q in fractions(p))
    def vector(i,label):
        j,sx,sy=label
        return tuple(points[j][k]+sx*B[0][k]+sy*B[1][k]-points[i][k] for k in range(2))
    directed={};areas=[];selected={}
    for i,(indices,labels) in enumerate(zip(packet['cell_vertex_indices'],packet['cell_incoming_labels'])):
        assert len(indices)==len(labels)>=3;poly=[]
        for k,label in enumerate(labels):
            a=vector(i,label);v=vector(i,labels[(k+1)%len(labels)]);ca=dot(a,a)/2;cv=dot(v,v)/2;den=cross(a,v);assert den
            q=((ca*v[1]-a[1]*cv)/den,(a[0]*cv-ca*v[0])/den)
            assert canonical(tuple(points[i][k]+q[k] for k in range(2)))==vertices[indices[k]]
            poly.append(q)
        twice=pyramid=F(0)
        for k,(q,label) in enumerate(zip(poly,labels)):
            prev=poly[k-1];nxt=poly[(k+1)%len(poly)];assert cross(sub(q,prev),sub(nxt,q))>0
            v=vector(i,label);d2=dot(v,v);edge=sub(q,prev)
            assert d2>0 and dot(v,edge)==0 and 2*dot(v,q)==2*dot(v,prev)==d2
            ratio=cross(v,edge)/d2;assert ratio>0;twice+=cross(prev,q);pyramid+=ratio*d2/4
            key=(i,*label);assert key not in directed;directed[key]=(ratio,(indices[k-1],indices[k]))
        area=twice/2;assert area>0 and area==pyramid==F(packet['cell_areas_exact'][i]);areas.append(area)
        if int(ids[i]) in probe_ids:selected[int(ids[i])]=(i,poly)
    lo,hi=fixed_binary_enclosure(areas);assert lo<=det<=hi
    old=packet['geometry_diagnostics']['raw_total_area_enclosure'];assert F(old['lower_exact'])<=hi and lo<=F(old['upper_exact'])
    conductances=[];edges=[];degree=[F(0)]*len(ids);keys=set()
    for k,((i,j),(sx,sy)) in enumerate(zip(packet['edges'],packet['image_shifts'])):
        assert ids[i]<ids[j] and (i,j,sx,sy) not in keys;keys.add((i,j,sx,sy))
        first=directed[(i,j,sx,sy)];other=directed[(j,i,-sx,-sy)]
        assert first[0]==other[0]==F(packet['face_over_distance_exact'][k])
        assert first[1]==tuple(packet['face_vertex_indices'][k])==other[1][::-1]
        g=thickness*first[0]/(1/(2*density[i]*diffusion)+1/(2*density[j]*diffusion))
        assert g>0 and g==F(packet['conductances_exact'][k]);conductances.append(g);edges.append((i,j));degree[i]+=g;degree[j]+=g
    assert 2*len(edges)==len(directed)
    probes=[]
    for ident in probe_ids:
        i,poly=selected[ident];radius2=max(dot(q,q) for q in poly);assert radius2<min(L)**2/16
        possible=0;worst=None
        for j in range(len(points)):
            for sx in (-1,0,1):
                for sy in (-1,0,1):
                    if j==i and sx==sy==0:continue
                    v=vector(i,(j,sx,sy));d2=dot(v,v)
                    if d2>4*radius2:continue
                    possible+=1;residual=max(2*dot(v,q)-d2 for q in poly);assert residual<=0
                    worst=residual if worst is None else max(worst,residual)
        probes.append({'id':ident,'world_images_checked':9*len(points)-1,'possible_cutters_tested':possible,
          'maximum_signed_world_halfplane_residual_exact':str(worst),'circumradius_squared_exact':str(radius2)})
    certificate={'all_world_vertices_rebuilt_directly_from_world_bisectors':True,'all_world_face_counterparts_and_half_resistances_exact':True,
      'no_local_cloud_or_original_geometry_builder_called':True,'orthogonal_world_periodic_basis_verified':True,
      'area_sum_256bit_enclosure':{'lower_exact':str(lo),'upper_exact':str(hi)},'shared_positive_faces':len(edges),
      'fixed_all_world_image_probes':{'fixed_ids':probe_ids,'records':probes,'all_selected_cells_passed':True},
      'farther_image_exclusion':'canonical lattice fractions differ by<1; any image outside{-1,0,1}^2 has norm>minL,while2R<minL/2'}
    return np.array(edges,dtype=np.int64),conductances,degree,certificate
