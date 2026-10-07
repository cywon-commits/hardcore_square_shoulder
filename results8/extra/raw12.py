import sys, numpy as np, cmath, math
sys.path.insert(0,"/home/user/hardcore_square_shoulder")
import op, op_melting as m
LAM,P_=1.93,0.735
def raw(P,M,L,n=12):
    inv=np.linalg.inv(M); acc=np.zeros(len(P),complex); c=np.zeros(len(P)); allz=[]
    for (i,j,k) in L:
        d=P[j]-P[i]; f=d@inv.T; f-=np.round(f); d=f@M.T; th=math.atan2(d[1],d[0]); z=cmath.exp(1j*n*th)
        acc[i]+=z; c[i]+=1; acc[j]+=z; c[j]+=1; allz.append(z)   # n=12: sign of bond direction irrelevant
    loc=acc/np.maximum(c,1); return abs(np.mean(allz)), loc
def gr(P,M,f):
    r,g=m.g_corr(P,M,f); return g[np.argmin(abs(r-10))], g[np.argmin(abs(r-2))]
def do(name,f,T):
    z=np.load(f); P,M=z["pos"],z["box"]; tol=0.10+2*T/(P_*LAM)
    o,(L,lf)=op.order_parameters(P,M,lam=LAM,tol_in=0.06,tol_out=tol)
    g12,loc=raw(P,M,L,12); g6,loc6=raw(P,M,L,6)
    a,b=gr(P,M,loc); a6,b6=gr(P,M,loc6)
    print(f"{name:28s} T={T:.2f} raw|psi12|={g12:.3f} g12raw(r~10)={a:.3f} (r~2 {b:.3f})  raw|psi6|={g6:.3f} g6raw(10)={a6:.3f}  S/N {m.s_peak(P,M):.4f} bonds {len(L)}")
Ts=[0.04,0.05,0.06,0.08,0.10,0.12,0.15]
for k,T in enumerate(Ts,1): do("heat_hexlat",f"runs2/heat_hexlat/snap_{40000*k:08d}.npz",T)
for k,T in enumerate([0.15,0.12,0.10,0.08,0.06,0.05,0.04],1): do("cool_fluid",f"runs2/cool_fluid/snap_{40000*k:08d}.npz",T)
for T in ("0.04","0.06","0.08","0.10","0.12"):
    for init in ("rows","hexlat"): do(f"T{T}_{init}",f"runs2/T{T}_{init}/snap_00200000.npz",float(T))
do("pilot1_rows_T0.15","runs/T0.15_rows/snap_01000000.npz",0.15)
