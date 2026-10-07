import sys, glob, json, numpy as np
sys.path.insert(0,"/home/user/hardcore_square_shoulder")
import op, op_melting as m
P_,LAM=0.735,1.93
Ts=[0.04,0.05,0.06,0.08,0.10,0.12,0.15]
for run,order in (("runs2/heat_hexlat",Ts),("runs2/cool_fluid",[0.15,0.12,0.10,0.08,0.06,0.05,0.04])):
    print(run)
    for k,T in enumerate(order,1):
        f=f"{run}/snap_{40000*k:08d}.npz"
        try: z=np.load(f)
        except Exception as e: print(" missing",f); continue
        P,M=z["pos"],z["box"]; tol=0.10+2*T/(P_*LAM)
        o,(L,lf)=op.order_parameters(P,M,lam=LAM,tol_in=0.06,tol_out=tol)
        r,g=m.g_corr(P,M,m.local_psi(P,L,12))
        print(f" T={T:.2f} sweep={40000*k} contact {1-o['x_other']:.3f} eta {o['eta']:+.3f} psi6 {o['psi6']:.3f} psi12 {o['psi12']:.3f} g12(10) {g[np.argmin(abs(r-10))]:.3f} S/N {m.s_peak(P,M):.4f} defects {o['n_defect_edges']}")
