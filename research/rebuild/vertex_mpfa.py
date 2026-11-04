"""One fixed vertex-local multipoint flux candidate for mass-average cells.

Internal continuity points are centroid-line/edge intersections; boundary
points are edge midpoints. Each vertex has its own auxiliary face traces.
No flux coefficient, normal, state, mass or negative matrix entry is repaired.
This is an MPFA-type diagnostic, not an accepted production backend.
"""
import numpy as np


def continuity_points(history,positions,centroids):
    x=np.asarray(positions,dtype=float);b=np.asarray(centroids,dtype=float)
    if x.shape!=(len(history.vertex_ids),2) or b.shape!=(len(history.cell_ids),2):
        raise ValueError('matching 2D vertex and true mass-centroid arrays required')
    if not np.isfinite(x).all() or not np.isfinite(b).all():raise ValueError('finite geometry required')
    pairs=history.edge_vertex_indices;a,z=pairs.T;v=x[z]-x[a];length=np.linalg.norm(v,axis=1)
    if np.any(length<=0):raise ValueError('strictly positive edge length required')
    normal=np.column_stack((v[:,1],-v[:,0]))/length[:,None]
    points=(x[a]+x[z])/2;fraction=np.full(len(a),.5);line_fraction=np.full(len(a),np.nan)
    for e in np.flatnonzero(~history.boundary_edge_mask):
        i=np.flatnonzero(history.incidence[:,e]==1).item();j=np.flatnonzero(history.incidence[:,e]==-1).item()
        d=b[j]-b[i]
        s,r=np.linalg.solve(np.column_stack((d,-v[e])),x[a[e]]-b[i])
        if not (0<s<1 and 0<r<1):raise ValueError('centroid-line continuity point outside open internal edge; no fallback')
        points[e]=x[a[e]]+r*v[e];fraction[e]=r;line_fraction[e]=s
    return points,normal,fraction,line_fraction


def build_vertex_mpfa(history,positions,centroids,half_coefficients):
    """Eliminate vertex-local auxiliary traces, retaining Neumann data explicitly.

    F maps cell mass averages to canonical internal face flux. B maps outward
    prescribed boundary half-face flux to internal face flux. Boundary unknowns
    are ordered by increasing edge index then endpoint. Physical half integrals
    are supplied, not fitted. Each cell trace polynomial interpolates its mean
    at its true mass centroid, which is exact for linear scalar fields.
    """
    points,normals,fraction,line_fraction=continuity_points(history,positions,centroids)
    coefficient=np.asarray(half_coefficients,dtype=float);E=len(history.edge_vertex_indices)
    if coefficient.shape!=(E,2) or not np.isfinite(coefficient).all() or np.any(coefficient<0):
        raise ValueError('finite nonnegative coefficients for both halves of every edge required')
    if np.any(coefficient==0) and not np.all(coefficient==0):
        raise ValueError('mixed degenerate diffusion not implemented; no coefficient floor')
    internal=np.flatnonzero(~history.boundary_edge_mask);boundary=np.flatnonzero(history.boundary_edge_mask)
    boundary_edges=np.repeat(boundary,2);boundary_endpoint=np.tile([0,1],len(boundary))
    boundary_lookup={(int(e),int(v)):k for k,(e,v) in enumerate(zip(boundary_edges,boundary_endpoint))}
    internal_lookup={int(e):k for k,e in enumerate(internal)}
    C=len(history.cell_ids);V=len(history.vertex_ids);F=np.zeros((len(internal),C));B=np.zeros((len(internal),len(boundary_edges)))
    diagnostics={'method':'vertex-local multipoint flux, fixed mass-centroid intersection continuity points',
      'new_physical_backend':False,'zero_diffusion':bool(np.all(coefficient==0)),
      'maximum_local_condition':0.,'maximum_local_elimination_residual':0.,
      'auxiliary_traces_are_independent_between_vertex_interactions':True}
    raw={'continuity_points':points,'continuity_edge_fraction':fraction,'continuity_centroid_line_fraction':line_fraction,
         'internal_edges':internal,'boundary_half_edges':boundary_edges,'boundary_half_endpoints':boundary_endpoint,
         'boundary_outward_sign':history.incidence[:,boundary_edges].sum(axis=0),'canonical_normal':normals,
         'local_A':np.full((V,4,4),np.nan),'local_R':np.full((V,4,4),np.nan),
         'local_edges':np.full((V,4),-1,dtype=int),'local_cells':np.full((V,4),-1,dtype=int)}
    if diagnostics['zero_diffusion']:
        return {'F':F,'B':B,'raw':raw,'diagnostics':diagnostics}
    for vertex in range(V):
        es=np.flatnonzero(np.any(history.edge_vertex_indices==vertex,axis=1))
        cs=np.flatnonzero(np.any(history.cell_vertex_indices==vertex,axis=1))
        ne,nc=len(es),len(cs)
        if ne>4 or nc>4:raise ValueError('only the current at-most-four-cell vertex topology is implemented')
        gl=np.zeros((nc,2,ne));gc=np.zeros((nc,2,nc));A=np.zeros((ne,ne));R=np.zeros((ne,nc))
        for cpos,cell in enumerate(cs):
            adjacent=np.flatnonzero(history.incidence[cell,es]!=0)
            if len(adjacent)!=2:raise ValueError('quadrilateral vertex must have exactly two incident cell edges')
            distances=points[es[adjacent]]-centroids[cell]
            inv=np.linalg.solve(distances,np.eye(2))
            gl[cpos][:,adjacent]=inv;gc[cpos,:,cpos]=-inv.sum(axis=1)
        for epos,e in enumerate(es):
            endpoint=int(np.flatnonzero(history.edge_vertex_indices[e]==vertex).item())
            k=coefficient[e,endpoint]
            for cpos,cell in enumerate(cs):
                sign=history.incidence[cell,e]
                if sign:
                    vector=-k*sign*normals[e]
                    A[epos]+=vector@gl[cpos];R[epos]+=vector@gc[cpos]
        bpos=np.flatnonzero(history.boundary_edge_mask[es]);J=np.eye(ne)[:,bpos]
        trace_c=np.linalg.solve(A,-R);trace_h=np.linalg.solve(A,J)
        diagnostics['maximum_local_condition']=max(diagnostics['maximum_local_condition'],float(np.linalg.cond(A)))
        diagnostics['maximum_local_elimination_residual']=max(diagnostics['maximum_local_elimination_residual'],
            float(np.max(abs(A@trace_c+R))),float(np.max(abs(A@trace_h-J))) if len(bpos) else 0.)
        raw['local_A'][vertex,:ne,:ne]=A;raw['local_R'][vertex,:ne,:nc]=R
        raw['local_edges'][vertex,:ne]=es;raw['local_cells'][vertex,:nc]=cs
        columns=[]
        for p in bpos:
            e=es[p];endpoint=int(np.flatnonzero(history.edge_vertex_indices[e]==vertex).item())
            columns.append(boundary_lookup[(int(e),endpoint)])
        for epos,e in enumerate(es):
            if history.boundary_edge_mask[e]:continue
            owner=np.flatnonzero(history.incidence[cs,e]==1).item()
            endpoint=int(np.flatnonzero(history.edge_vertex_indices[e]==vertex).item())
            vector=-coefficient[e,endpoint]*normals[e]
            q_c=vector@(gl[owner]@trace_c+gc[owner]);q_h=vector@gl[owner]@trace_h
            row=internal_lookup[int(e)];F[row,cs]+=q_c
            if columns:B[row,columns]+=q_h
    return {'F':F,'B':B,'raw':raw,'diagnostics':diagnostics}


def internal_flux(operator,state,boundary_outward_half_flux):
    c=np.asarray(state);h=np.asarray(boundary_outward_half_flux)
    if c.shape[0]!=operator['F'].shape[1] or h.shape[0]!=operator['B'].shape[1]:raise ValueError('matching cell and boundary arrays required')
    if not np.isfinite(c).all() or not np.isfinite(h).all():raise ValueError('finite state and boundary flux required')
    if operator['diagnostics']['zero_diffusion'] and np.any(h!=0):raise ValueError('nonzero prescribed diffusion flux incompatible with zero diffusion')
    return operator['F']@c+operator['B']@h
