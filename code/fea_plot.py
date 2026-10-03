import numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt, json, sys, os
sys.path.append(os.getcwd()); import fea_ratchet as F
fig,axes=plt.subplots(1,2,figsize=(13,5.6)); out={}
for ax,M in zip(axes,(70.0,150.0)):
    r=F.run_sector_only(M, tag=f"_p{int(M)}") if hasattr(F,"run_sector_only") else None
    b,u,vm,_=r; m=b.mesh
    p=m.p; t=m.t; vmax=np.percentile(vm,99.5)
    tc=ax.tripcolor(p[0],p[1],t.T,facecolors=np.clip(vm,0,vmax),cmap="inferno",vmin=0,vmax=vmax)
    ax.triplot(p[0],p[1],t.T,lw=0.08,color="w",alpha=0.3)
    ax.set_aspect("equal"); ax.set_title(f"Toothed shank plate, 5 mm steel, M = {M:.0f} N·m\nF = {M/(F.R_c/1000):.0f} N shared 50/30/20 % over three teeth",fontsize=10)
    cb=fig.colorbar(tc,ax=ax,fraction=0.046); cb.set_label("von Mises stress (MPa), capped at 99.5th pct")
    ax.set_xlabel("x anterior (mm)"); ax.set_ylabel("y proximal (mm)"); ax.set_xlim(-28,22); ax.set_ylim(14,40)
    # stress at 0.3 mm from the hottest root corner (hot-spot excluding the corner singularity): take p99 and p99.5
    out[M]=dict(vm_p99=float(np.percentile(vm,99)),vm_p995=float(vmax),vm_absmax=float(vm.max()),umax_mm=float(np.abs(u).max()),n_el=int(t.shape[1]))
    print(M,out[M])
fig.tight_layout(); fig.savefig("fig/fig_fea_vm.png",dpi=160); json.dump(out,open("fea_results.json","w"),indent=1)
