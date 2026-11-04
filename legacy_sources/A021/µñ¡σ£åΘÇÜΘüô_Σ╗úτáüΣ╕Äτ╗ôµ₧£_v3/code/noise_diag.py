import sys, numpy as np, tcx, afe, afe_cases as AC
name=sys.argv[1]; c=AC.load(name); fr=c['fr']; S,N=fr.shape[:2]; h=c['h']; m=c['inner']
rng=np.random.default_rng(5); frn=fr+rng.normal(0,0.1*h,fr.shape)
mus,_=tcx.mu_threshold(frn); Ks=[tcx.cplx(frn[s])['tri'] for s in range(S-1)]
ch=tcx.run_chain(frn,mus,Ks=Ks); phi,lavd,nseg,psib=tcx.rotation_fields(ch)
ch2=tcx.run_chain(frn,np.inf,cplx_cache=ch['cache']); phi2,_,_,_=tcx.rotation_fields(ch2)
sg=afe.node_noise(ch,psib); env=sg*np.sqrt(nseg)
rem=np.array([sum(abs(r) for _,_,r in ch['segs'][n]) for n in range(N)])
print('hot_frac',ch['hot_frac'],'nseg med',np.median(nseg))
for lab,f in [('merge',phi),('nomerge',phi2)]:
    r=np.abs(f[m])/env[m]; print(lab,'|phi| q50,q99,max',np.quantile(np.abs(f[m]),[.5,.99]).round(3),np.abs(f[m]).max().round(3),' ratio q50 q90 q99 max',np.quantile(r,[.5,.9,.99]).round(2),r.max().round(1))
print('rem sum q50 q99',np.quantile(rem[m],[.5,.99]).round(3))
# sum of per-seg noise if segments independent: compare empirical std of phi vs env
print('std phi / median env', np.std(phi[m])/np.median(env[m]))
