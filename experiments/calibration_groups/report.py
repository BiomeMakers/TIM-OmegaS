import numpy as np, sys, warnings; warnings.filterwarnings("ignore")
sys.path.insert(0,'.'); from read_pt import cargar; import tim_calib as t
from sklearn.metrics import adjusted_rand_score as ari
base='Amatrices_final/'   # unzip Amatrices.zip from Zenodo 10.5281/zenodo.20125262 here
M={'Llama-8B':('Llama-3-8B-Instruct_n_samples_2048',[14,16,17,18,19,20,21,22,23,24,26,27,28,29,30,31]),
   'Llama-70B':('Llama-3-70B-Instruct_n_samples_2048',[46,55,56,57,58,59,60,61,62,64,65,66,68,71,72,74]),
   'Qwen-14B':('Qwen3-14B_n_samples_2048_think',[2,26,30,31,32,33,34,35,36,37,38,39]),
   'Nemotron':('Nemotron2048',[8,10,38])}
G={}
for nom,(d,R) in M.items():
    A=cargar(base+d+'/A.pt').astype(np.float64); g=t.descubrir_grupos(A,kmax=6,n_boot=15); G[nom]=g
    np.save(f'results/grupos_{nom}.npy',g['etiquetas'])
    P=t.perfil_bloques(A,g['etiquetas']); D=t.dano(A,g['etiquetas'],R)
    print(f"\n=== {nom}: k={g['k']} silueta={g['silueta']:.3f}")
    for c in range(g['k']):
        top=np.argsort(-P[c]['rel'])[:4]
        print(f"  grupo {c}: {g['tamanos'][c]:4d} muestras ({g['tamanos'][c]/len(A):.0%}) estab={g['estabilidad'][c]:.2f} "
              f"Spearman={P[c]['spearman']:.2f}  bloques que mas valora vs conjunto={top.tolist()}  "
              f"dano de su seleccion={D[c]['dano_relativo']:.2f} [nulo {D[c]['nulo_p05']:.2f}-{D[c]['nulo_p95']:.2f}]")
print("\n=== Comprobacion 1 repetida (mismas muestras en los tres modelos)")
rng=np.random.default_rng(3); n=['Llama-8B','Llama-70B','Qwen-14B']
for i in range(3):
    for j in range(i+1,3):
        a,b=G[n[i]]['etiquetas'],G[n[j]]['etiquetas']; r=ari(a,b)
        p99=np.percentile([ari(a,rng.permutation(b)) for _ in range(1000)],99)
        print(f"  {n[i]} vs {n[j]}: ARI={r:.3f} nulo p99={p99:.4f} -> {'PASA' if r>p99 else 'no'}")
