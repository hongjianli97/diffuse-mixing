"""Exact fixed-material mass restriction; no normalization or mass repair."""
from dataclasses import dataclass
from fractions import Fraction as F

from periodic_density_path import _fraction


@dataclass(frozen=True)
class Restriction:
    means: tuple
    inventory: F
    exact_inventory_residual: F
    jensen_square_gap: F


def restrict_material_means(fine_mass,fine_means,coarse_mass,coarse_N,ratio=2):
    if isinstance(coarse_N,bool) or not isinstance(coarse_N,int) or coarse_N<1:
        raise ValueError('positive integer coarse grid size required')
    if isinstance(ratio,bool) or not isinstance(ratio,int) or ratio<1:
        raise ValueError('positive integer nesting ratio required')
    nf=coarse_N*ratio
    mf=tuple(_fraction(x,'fine mass') for x in fine_mass)
    cf=tuple(_fraction(x,'fine material mean') for x in fine_means)
    mc=tuple(_fraction(x,'original coarse mass') for x in coarse_mass)
    if len(mf)!=nf*nf or len(cf)!=len(mf) or len(mc)!=coarse_N*coarse_N:
        raise ValueError('complete row-major nested material grids required')
    if any(m<=0 for m in mf+mc):raise ValueError('positive fixed masses required')
    result=[]
    for j in range(coarse_N):
        for i in range(coarse_N):
            children=[(j*ratio+b)*nf+i*ratio+a for b in range(ratio) for a in range(ratio)]
            target=mc[j*coarse_N+i]
            if sum((mf[k] for k in children),F(0))!=target:
                raise ValueError('fine mass must exactly sum to the original coarse mass; no repair')
            result.append(sum((mf[k]*cf[k] for k in children),F(0))/target)
    fine_inventory=sum((m*c for m,c in zip(mf,cf)),F(0))
    inventory=sum((m*c for m,c in zip(mc,result)),F(0))
    square_gap=sum((m*c*c for m,c in zip(mf,cf)),F(0))-sum((m*c*c for m,c in zip(mc,result)),F(0))
    assert inventory==fine_inventory and square_gap>=0
    return Restriction(tuple(result),inventory,inventory-fine_inventory,square_gap)
