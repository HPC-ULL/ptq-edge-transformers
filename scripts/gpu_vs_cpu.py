import sys
from pathlib import Path
import pandas as pd, numpy as np
ROOT=Path(__file__).resolve().parents[1]
U=str(ROOT/'results'/'raw')+'/'
OUT=ROOT/'results'/'investigations'/'llm'/'gpu_vs_cpu'
models=['tinyllama_1_1b','qwen2_5_1_5b','ministral_3_3b']
fmts={'Q4_K_M':'phase1_baseline_default_q4km','Q4_0':'phase2_quant_scheme_legacy_corpus_none_q4km',
      'Q3_K_M':'phase3_baseline_q3_k','Q2_K':'phase3_baseline_q2_k','F16':'reference_f16'}
modes=['15w','30w','50w']; K=(0.11,0.39)
def load(b,m):
    d=pd.read_csv(f'{U}results_jetson_{b}_{m}.csv').set_index('run_id'); return d
def read(r): return 'GPU mejor' if r>1.1 else ('CPU mejor' if r<0.9 else 'sin diferencia apreciable')
rows=[]
for m in modes:
    C,G=load('cpu',m),load('gpu',m)
    for mod in models:
        for f,s in fmts.items():
            rid=f'llm_{mod}_{s}'; c,g=C.loc[rid],G.loc[rid]
            assert len(C.loc[[rid]])==1 and len(G.loc[[rid]])==1
            def phi(x):
                N=x.num_inferences_measured; trep=128/x.throughput_prefill_tps+32/x.throughput_decode_tps
                w=x.energy_raw_j/x.avg_power_w_raw; return (w-N*trep)/w, trep
            pc,tc=phi(c); pg,tg=phi(g)
            rpp=g.throughput_prefill_tps/c.throughput_prefill_tps
            rtg=g.throughput_decode_tps/c.throughput_decode_tps
            rE=c.energy_per_inference/g.energy_per_inference
            # k_GPU=0 (indistinguishable from zero); k_CPU in [0.11, 0.39]
            rk=[rE*(1-k*pc)/(1-0*pg) for k in K]
            rows.append(dict(mode=m.upper().replace('W',' W'),model=mod,format=f,run_id=rid,
              pp_cpu=c.throughput_prefill_tps,pp_gpu=g.throughput_prefill_tps,r_pp=rpp,read_pp=read(rpp),
              tg_cpu=c.throughput_decode_tps,tg_gpu=g.throughput_decode_tps,r_tg=rtg,read_tg=read(rtg),
              energy_cpu_j=c.energy_per_inference,energy_gpu_j=g.energy_per_inference,r_energy=rE,read_energy=read(rE),
              power_net_cpu_w=c.avg_power_w_net,power_net_gpu_w=g.avg_power_w_net,
              n_cpu=c.num_inferences_measured,n_gpu=g.num_inferences_measured,
              decode_time_frac_cpu=(32/c.throughput_decode_tps)/tc,decode_time_frac_gpu=(32/g.throughput_decode_tps)/tg,
              fixed_frac_cpu=pc,fixed_frac_gpu=pg,r_energy_k011=rk[0],r_energy_k039=rk[1],
              energy_bias_max_pct=(rE/rk[1]-1)*100,read_energy_bounds=read(rk[0])+'/'+read(rk[1])))
df=pd.DataFrame(rows); OUT.mkdir(parents=True,exist_ok=True); df.to_csv(OUT/'ratios.csv',index=False,float_format='%.4f')
print(df[['n_cpu','n_gpu']].drop_duplicates())
for c in ['read_pp','read_tg','read_energy','read_energy_bounds']: print(df[c].value_counts().to_dict())
print(df[['r_pp','r_tg','r_energy','energy_bias_max_pct','fixed_frac_cpu','fixed_frac_gpu','decode_time_frac_cpu','decode_time_frac_gpu']].describe().round(3).to_string())
fmt=lambda x:f'{x:.2f}'.replace('.',',')
for mod in models:
    print('\n',mod)
    t=df[df.model==mod]
    for m in t['mode'].unique():
        s=t[t['mode']==m]
        print(m,' | '.join(f"{fmt(r.r_pp)} / {fmt(r.r_tg)} / {fmt(r.r_energy)}" for r in s.itertuples()),
              ' | P cpu/gpu', ' '.join(f"{r.power_net_cpu_w:.1f}/{r.power_net_gpu_w:.1f}" for r in s.itertuples()))
