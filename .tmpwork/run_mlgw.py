import os, sys, time, contextlib, argparse
os.environ["TF_CPP_MIN_LOG_LEVEL"]="3"
import jax; jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp, numpy as np
import gwgpu_jax

ap=argparse.ArgumentParser()
ap.add_argument("--grid",choices=["fixed","old"],default="fixed")
ap.add_argument("--modes",choices=["all","22"],default="all")
ap.add_argument("--num-live",type=int,default=400)
ap.add_argument("--out",default="/Users/ut/Documents/GitHub/GWgpu_jax/.tmpwork/res_fixed.txt")
A=ap.parse_args()

with open(os.devnull,'w') as dn, contextlib.redirect_stdout(dn), contextlib.redirect_stderr(dn):
    from gwgpu_jax.mlgw_jax.mlgw_bbh_jax_waveform_generator import MLGWBBHGenerator, hphc_td_to_fd
    bbh=MLGWBBHGenerator(model=4)

grid=gwgpu_jax.TimeFrequencyGrid(duration=4.0,sampling_rate=4096.0,f_min=20.0,f_max=1024.0)
net=gwgpu_jax.Network.from_names(["H1","L1"],grid)
gwgpu_jax.compat.attach_event_to_network(net,"GW150914",estimate_psd=True,
    psd_segment_duration=32.0,psd_offset=8.0,verbose=False)

dt=grid.dt; N=grid.n_samples
t_fixed=grid.time_domain_array_mlgw
t_old=jnp.arange(N)*dt-4.0
t_grid = t_fixed if A.grid=="fixed" else t_old
modes = None if A.modes=="all" else (2,2)

def waveform_fn(params,freqs):
    s=params["m1"]<params["m2"]
    m1=jnp.where(s,params["m2"],params["m1"]); m2=jnp.where(s,params["m1"],params["m2"])
    c1=jnp.where(s,params["chi_2"],params["chi_1"]); c2=jnp.where(s,params["chi_1"],params["chi_2"])
    th=jnp.stack([m1,m2,c1,c2,params["distance"],params["inclination"],params["phi_c"]])
    hp,hc=bbh.generate(th,t_grid,modes=modes)
    if A.grid=="fixed":
        return hphc_td_to_fd(hp,hc,dt)
    return jnp.fft.rfft(hp)*dt, jnp.fft.rfft(hc)*dt

TC=0.875*grid.duration
PB={"m1":(10.,80.),"m2":(10.,80.),"chi_1":(-0.9,0.9),"chi_2":(-0.9,0.9),
    "distance":(50.,2000.),"inclination":(0.,float(jnp.pi)),
    "ra":(0.,2*float(jnp.pi)),"dec":(-float(jnp.pi)/2,float(jnp.pi)/2),
    "psi":(0.,float(jnp.pi)),"phi_c":(0.,2*float(jnp.pi)),
    "tc":(TC-0.5,TC+0.5)}
sampler=gwgpu_jax.GWgpu_jaxTwoPhaseNestedSampler(network=net,waveform_fn=waveform_fn,
    param_bounds=PB,fixed_params={},gmst=None)

t0=time.perf_counter()
res=sampler.run_two_phase(rng_key=jax.random.PRNGKey(0),num_live=A.num_live,
    num_inner_steps=50,phase1_num_delete=max(1,A.num_live//20),
    phase1_delta_logz_threshold=-1.0,phase1_max_iterations=1500,
    phase2_num_delete=1,phase2_max_iterations=8000,log_dlogz_target=-3.0,
    num_posterior_samples=2000,verbose=False)
el=time.perf_counter()-t0

# heavier-first relabel
ps=dict(res.posterior_samples)
m1,m2=ps["m1"],ps["m2"]; sw=m1<m2
ps["m1"]=jnp.maximum(m1,m2); ps["m2"]=jnp.minimum(m1,m2)
ps["chi_1"]=jnp.where(sw,ps["chi_2"],ps["chi_1"]); ps["chi_2"]=jnp.where(sw,ps["chi_1"],ps["chi_2"])

L=[]
L.append("grid=%s modes=%s num_live=%d  wall=%.1fs  logZ=%.2f ESS=%.1f iters(p1=%s,p2=%s)"%(
    A.grid,A.modes,A.num_live,el,float(res.logZ),float(res.ess),
    res.phase1_iterations,res.phase2_iterations))
ref=dict(m1=35.6,m2=30.6,distance=440.,inclination=2.5,ra=2.21,dec=-1.25,tc=TC)
for n in ["m1","m2","chi_1","chi_2","distance","inclination","ra","dec","psi","phi_c","tc"]:
    s=np.asarray(ps[n]); lo,mid,hi=np.percentile(s,[16,50,84])
    lob,hib=PB[n]; railed=""
    if (mid-lob)<0.03*(hib-lob) or (hib-mid)<0.03*(hib-lob): railed=" <-- RAILED"
    r="%+.3f"%ref[n] if n in ref else "—"
    L.append("  %-11s %+10.3f  -%.3f +%.3f   ref=%s%s"%(n,mid,mid-lo,hi-mid,r,railed))
open(A.out,"w").write("\n".join(L)+"\n")
print("DONE")
