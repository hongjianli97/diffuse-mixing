"""Time actions for a saved, transversely repeated shared exchange stencil.

No kernel or density-path construction. All discrete initial modes and raw
eigenvalues are retained; uniformization weights are never renormalized.
"""
from dataclasses import dataclass
from decimal import Decimal,localcontext
from fractions import Fraction as F
import math
import numpy as np


@dataclass(frozen=True)
class SavedStencil:
    N: int
    offsets: tuple
    conductances: tuple
    coefficient_map: tuple
    mass: tuple
    sqrt_mass: tuple
    outgoing: tuple


def load_stencil(cache,mass,ids,positions):
    N=cache['N']
    if isinstance(N,bool) or not isinstance(N,int) or N<4 or N&(N-1):raise ValueError('dyadic square grid required')
    ident=np.asarray(ids);m=np.asarray(mass,dtype=float);x=np.asarray(positions,dtype=float)
    if ident.shape!=(N*N,) or ident.dtype.kind not in 'iu' or not np.array_equal(ident,np.arange(N*N)):
        raise ValueError('unaltered original canonical IDs required')
    if m.shape!=(N*N,) or not np.isfinite(m).all() or np.any(m<=0):raise ValueError('all masses finite and positive required')
    if not np.array_equal(m.reshape(N,N),np.broadcast_to(m[:N],(N,N))):raise ValueError('mass must repeat transversely')
    expected=np.array([((c+.5)/N,(r+.5)/N) for r in range(N) for c in range(N)])
    if not np.array_equal(x,expected):raise ValueError('original complete point-array coordinates required')
    offsets=tuple(tuple(v) for v in cache['offsets_positive_half'])
    if not offsets or len(set(offsets))!=len(offsets):raise ValueError('unique positive-half offsets required')
    if any(len(v)!=2 or any(isinstance(q,bool) or not isinstance(q,int) for q in v) or
        not (v[0]>0 or v[0]==0 and v[1]>0) or max(abs(q) for q in v)>=N//2 for v in offsets):
        raise ValueError('unique short minimum-image positive-half offsets required')
    g=tuple(F(v) for v in cache['conductances_exact'])
    if not g or any(v<0 for v in g):raise ValueError('raw nonnegative saved conductances required')
    mapping=tuple(tuple(v) for v in cache['coefficient_map'])
    if len(mapping)!=N or any(len(v)!=len(offsets) for v in mapping) or any(
        isinstance(k,bool) or not isinstance(k,int) or not 0<=k<len(g) for row in mapping for k in row):
        raise ValueError('complete valid coefficient cache mapping required')
    offset_lookup={v:k for k,v in enumerate(offsets)}
    for k,(dx,dy) in enumerate(offsets):
        if dx>0:
            if (dx,-dy) not in offset_lookup:raise ValueError('real block acceleration requires exact transverse reflection')
            other=offset_lookup[(dx,-dy)]
            if any(g[row[k]]!=g[row[other]] for row in mapping):raise ValueError('unequal reflected saved coefficients; no averaging repair')
    masses=tuple(F(float(v)) for v in m[:N]);roots=[]
    for v in masses:
        a,b=math.isqrt(v.numerator),math.isqrt(v.denominator)
        if a*a!=v.numerator or b*b!=v.denominator:raise ValueError('this exact conversion control needs rational mass roots')
        roots.append(F(a,b))
    degree=[F(0)]*N
    for c in range(N):
        for k,(dx,dy) in enumerate(offsets):
            w=g[mapping[c][k]];degree[c]+=w;degree[(c+dx)%N]+=w
    outgoing=tuple(w/m for w,m in zip(degree,masses))
    return SavedStencil(N,offsets,g,mapping,masses,tuple(roots),outgoing)


def _cosine_table(N):
    binary=tuple(math.cos(2*math.pi*j/N) for j in range(N//2+1))
    with localcontext() as ctx:
        ctx.prec=110
        def atan_inverse(q):
            z=Decimal(1)/q;power=z;total=z;n=1
            while True:
                power*=-z*z;new=total+power/(2*n+1)
                if new==total:return total
                total=new;n+=1
        pi=16*atan_inverse(5)-4*atan_inverse(239)
        precise=[]
        for j in range(N//2+1):
            z=2*pi*j/N;term=Decimal(1);total=term;n=1
            while True:
                term*=-z*z/((2*n-1)*(2*n));new=total+term
                if new==total:break
                total=new;n+=1
            precise.append(F(total))
    return binary,tuple(precise)


@dataclass(frozen=True)
class ModalBlocks:
    stencil: SavedStencil
    blocks: np.ndarray
    eigenvalues: np.ndarray
    eigenvectors: np.ndarray
    conversion_bounds: tuple
    phase_bounds: tuple
    eigen_residuals: np.ndarray
    orthogonality_residuals: np.ndarray
    cosine_values: tuple
    cosine_precise: tuple


def build_modal_blocks(stencil,progress=None):
    s=stencil;N=s.N;cos,precise=_cosine_table(N)
    matrices=[];values=[];vectors=[];rounding=[];phases=[];residuals=[];orthogonal=[]
    for mode in range(N//2+1):
        exact=[[F(0) for _ in range(N)] for _ in range(N)]
        phase_rows=[F(0)]*N
        for c in range(N):exact[c][c]=-s.outgoing[c]
        for c in range(N):
            for k,(dx,dy) in enumerate(s.offsets):
                g=s.conductances[s.coefficient_map[c][k]]
                phase=(mode*dy)%N;phase=min(phase,N-phase)
                cosine=F(cos[phase]);err=abs(cosine-precise[phase])
                if dx==0:
                    factor=2*g/s.mass[c];exact[c][c]+=factor*cosine;phase_rows[c]+=factor*err
                else:
                    j=(c+dx)%N;factor=g/(s.sqrt_mass[c]*s.sqrt_mass[j]);v=factor*cosine
                    exact[c][j]+=v;exact[j][c]+=v
                    phase_rows[c]+=factor*err;phase_rows[j]+=factor*err
        H=np.array([[float(v) for v in row] for row in exact])
        if not np.array_equal(H,H.T):raise ArithmeticError('raw modal block not symmetric; no repair')
        bound=max(sum((abs(F(float(v))-q) for v,q in zip(row,original)),F(0)) for row,original in zip(H,exact))
        lam,V=np.linalg.eigh(H)
        matrices.append(H);values.append(lam);vectors.append(V);rounding.append(bound);phases.append(max(phase_rows))
        residuals.append(float(np.linalg.norm(H@V-V*lam,ord='fro')))
        orthogonal.append(float(np.linalg.norm(V.T@V-np.eye(N),ord='fro')))
        if progress is not None:progress('modal_blocks',mode+1,N//2+1)
    return ModalBlocks(s,np.array(matrices),np.array(values),np.array(vectors),tuple(rounding),tuple(phases),
        np.array(residuals),np.array(orthogonal),cos,precise)


def _state(stencil,cbar):
    a=np.asarray(cbar,dtype=float)
    if a.size!=stencil.N**2 or not np.isfinite(a).all():raise ValueError('full finite original state required')
    return a.reshape(stencil.N,stencil.N)


def modal_actions(modes,cbar,T):
    if not np.isfinite(T) or T<0:raise ValueError('finite T>=0 required')
    s=modes.stencil;N=s.N;state=_state(s,cbar);root=np.array(list(map(float,s.sqrt_mass)))
    z=np.fft.fft(state*root[None,:],axis=0,norm='ortho')
    coefficients=np.empty_like(z);initial_z=np.empty_like(z);final_z=np.empty_like(z);action_z=np.empty_like(z)
    for k in range(N):
        index=min(k,N-k);V=modes.eigenvectors[index];lam=modes.eigenvalues[index]
        coefficients[k]=z[k]@V
        initial_z[k]=coefficients[k]@V.T
        final_z[k]=(coefficients[k]*np.exp(float(T)*lam))@V.T
        action_z[k]=z[k]@modes.blocks[index].T
    def physical(a):
        result=np.fft.ifft(a,axis=0,norm='ortho')/root[None,:]
        return result.real.ravel().copy(),result.imag.ravel().copy()
    initial,imag0=physical(initial_z);final,imagT=physical(final_z);action,imagQ=physical(action_z)
    return {'state':final,'initial_roundtrip':initial,'initial_action':action,
        'imaginary_initial':imag0,'imaginary_T':imagT,'imaginary_initial_action':imagQ,
        'full_Fourier_coefficients':z,'full_eigen_coefficients':coefficients}


def stencil_action(stencil,cbar):
    s=stencil;N=s.N;state=_state(s,cbar);flux=np.zeros_like(state)
    for k,(dx,dy) in enumerate(s.offsets):
        g=np.array([float(s.conductances[s.coefficient_map[c][k]]) for c in range(N)])
        f=g[None,:]*(np.roll(state,(-dy,-dx),(0,1))-state)
        flux+=f;flux-=np.roll(f,(dy,dx),(0,1))
    return (flux/np.array(list(map(float,s.mass)))[None,:]).ravel()


def uniformization(stencil,cbar,T,Lambda,M,progress=None):
    s=stencil;N=s.N;state=_state(s,cbar).copy()
    if isinstance(Lambda,bool) or not isinstance(Lambda,int) or Lambda<=0 or F(Lambda)<max(s.outgoing):
        raise ValueError('positive integer Lambda bounding all exact outgoing rates required')
    if isinstance(M,bool) or not isinstance(M,int) or M<0 or not np.isfinite(T) or T<0:raise ValueError('valid degree and time required')
    diagonal=np.array([float(1-d/F(Lambda)) for d in s.outgoing])
    out=[];incoming=[]
    for k,(dx,dy) in enumerate(s.offsets):
        out.append(np.array([float(s.conductances[s.coefficient_map[c][k]]/(s.mass[c]*Lambda)) for c in range(N)]))
        incoming.append(np.array([float(s.conductances[s.coefficient_map[(c-dx)%N][k]]/(s.mass[c]*Lambda)) for c in range(N)]))
    raw_row_sum=diagonal.copy()
    for a,b in zip(out,incoming):raw_row_sum+=a+b
    z=Lambda*float(T);weight=math.exp(-z);answer=weight*state;weights=[weight]
    for n in range(1,M+1):
        value=diagonal[None,:]*state
        for k,(dx,dy) in enumerate(s.offsets):
            value+=out[k][None,:]*np.roll(state,(-dy,-dx),(0,1))
            value+=incoming[k][None,:]*np.roll(state,(dy,dx),(0,1))
        state=value;weight*=z/n;weights.append(weight);answer+=weight*state
        if progress is not None:progress('uniformization_terms',n,M)
    return {'state':answer.ravel(),'weights':np.array(weights),'raw_weight_sum':math.fsum(weights),
        'embedded_row_sum_minus_one':raw_row_sum-1,'raw_minimum_embedded_diagonal':float(diagonal.min())}


def poisson_tail_bound(z,M):
    z=F(z)
    if z<0 or isinstance(M,bool) or not isinstance(M,int) or M<0 or z>=M+2:
        raise ValueError('nonnegative exact z and M+2>z required')
    return z**(M+1)/math.factorial(M+1)/(1-z/F(M+2))
