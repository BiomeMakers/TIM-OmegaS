import numpy as np, pandas as pd, sys, warnings; warnings.filterwarnings("ignore")
sys.path.insert(0,'.'); from read_pt import cargar
from scipy.stats import spearmanr, chi2_contingency
from sklearn.metrics import roc_auc_score
d=pd.read_csv('muestras_openhermes_2048.csv').sort_values('fila_A').reset_index(drop=True)
base='Amatrices_final/'   # unzip Amatrices.zip from Zenodo 10.5281/zenodo.20125262 here
M={'Llama-8B':'Llama-3-8B-Instruct_n_samples_2048','Llama-70B':'Llama-3-70B-Instruct_n_samples_2048','Qwen-14B':'Qwen3-14B_n_samples_2048_think'}
def cramer(x,y):
    t=pd.crosstab(x,y); c=chi2_contingency(t)[0]; n=t.values.sum(); r,k=t.shape
    return np.sqrt(c/(n*(min(r,k)-1)))
src=d.source.fillna('').astype(str)
vc=src[src!=''].value_counts(); raros=set(vc[vc<10].index)
src1=src.where(~src.isin(raros),'otras')
rng=np.random.default_rng(5); pasa_b=0
for nom,dd in M.items():
    A=cargar(base+dd+'/A.pt').astype(np.float64); nrm=np.linalg.norm(A,axis=1)
    lab=np.load(f'results/grupos_{nom}.npy')
    rho,p=spearmanr(nrm,d.caracteres)
    print(f"\n=== {nom}")
    print(f"  (a) alineacion: Spearman(norma gradiente, caracteres)={rho:+.3f} p={p:.1e} -> {'OK' if abs(rho)>0.15 and p<1e-3 else 'FALLA'}")
    m=(src1!='').values
    V=cramer(lab[m],src1[m].values); nul=[cramer(rng.permutation(lab[m]),src1[m].values) for _ in range(1000)]
    p99=np.percentile(nul,99); ok=V>p99; pasa_b+=ok
    V2=cramer(lab,np.where(src1=='','sin_fuente',src1))
    print(f"  (b) grupo vs fuente (997 con fuente): V={V:.3f} nulo p99={p99:.3f} -> {'PASA' if ok else 'no'} | con 'sin fuente' como categoria: V={V2:.3f}")
    g_may=np.bincount(lab).argsort()[-2:]; mm=np.isin(lab,g_may)
    auc=roc_auc_score(lab[mm]==g_may[0], d.caracteres[mm]); auc=max(auc,1-auc)
    print(f"  control longitud: AUC de la longitud para separar los dos grupos principales={auc:.2f} -> {'LOS GRUPOS SON LONGITUD' if auc>0.8 else 'no se reducen a longitud'}")
    t=pd.crosstab(lab[m],src1[m].values,normalize='index')
    glob=src1[m].value_counts(normalize=True)
    for g in range(lab.max()+1):
        if g in t.index:
            ratio=(t.loc[g]/glob).sort_values(ascending=False)
            print(f"   grupo {g}: sobrerrepresentadas "+", ".join(f"{s} x{r:.1f}" for s,r in ratio.head(3).items())
                  +" | infrarrepresentadas "+", ".join(f"{s} x{r:.1f}" for s,r in ratio.tail(2).items()))
print(f"\n(b) pasa en {pasa_b} de 3 modelos -> {'LOS GRUPOS SE ASOCIAN A FUENTES' if pasa_b>=2 else 'NO se asocian a fuentes'}")
