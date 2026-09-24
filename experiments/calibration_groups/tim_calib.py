"""TIM aplicado a la calibración: qué grupos de muestras esconde un conjunto de calibración y qué bloques necesita cada uno.

Entrada: A (muestras x bloques), el gradiente de la pérdida de cada muestra respecto del coeficiente de cada bloque
(el mismo objeto que usa el CBO de Multiverse). Salida: grupos descubiertos sin etiquetas, su estabilidad, la
importancia de cada bloque para cada grupo, y cuánto daña a cada grupo una configuración de bloques retirados.
"""
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from scipy.stats import spearmanr

def normalizar(A):
    """Perfil de importancia de cada muestra: reparto de g^2 entre bloques, en coordenadas de Hellinger.
    Agrupa por QUÉ BLOQUES le importan a cada muestra, no por hacia dónde empuja su gradiente."""
    P = A ** 2; s = P.sum(1, keepdims=True); s[s == 0] = 1
    return np.sqrt(P / s)

def descubrir_grupos(A, kmax=6, n_boot=20, semilla=0):
    X = normalizar(A); rng = np.random.default_rng(semilla); mejor = None
    for k in range(2, kmax + 1):
        lab = KMeans(k, n_init=5, random_state=semilla).fit(X).labels_
        s = silhouette_score(X, lab, sample_size=min(len(X), 2048), random_state=semilla)
        if mejor is None or s > mejor[0]: mejor = (s, k, lab)
    s, k, lab = mejor
    estab = []                                   # Jaccard mediano de cada grupo en remuestreos del 80 %
    for g in range(k):
        ref = set(np.where(lab == g)[0]); js = []
        for b in range(n_boot):
            idx = rng.choice(len(X), int(0.8 * len(X)), replace=False); sidx = set(idx)
            lb = KMeans(k, n_init=3, random_state=100 + b).fit(X[idx]).labels_
            r = ref & sidx
            js.append(max(len(r & set(idx[lb == c])) / max(1, len(r | set(idx[lb == c]))) for c in range(k)))
        estab.append(float(np.median(js)))
    return dict(etiquetas=lab, k=k, silueta=float(s), estabilidad=estab,
                tamanos=np.bincount(lab, minlength=k).tolist())

def importancia(A, mascara=None):
    M = A if mascara is None else A[mascara]
    return (M ** 2).mean(0)                      # diagonal de la matriz de Gram del grupo

def perfil_bloques(A, lab):
    tot = importancia(A); out = {}
    for g in np.unique(lab):
        im = importancia(A, lab == g)
        out[int(g)] = dict(rel=(im / im.sum()) / (tot / tot.sum()),       # >1: el grupo valora ese bloque más que el conjunto
                           spearman=float(spearmanr(im, tot).correlation))
    return out

def dano(A, lab, R, n_nulo=200, semilla=0):
    """Fracción de importancia del grupo que se lleva la retirada R, relativa a la del conjunto; con nulo."""
    rng = np.random.default_rng(semilla); tot = importancia(A); base = tot[R].sum() / tot.sum(); out = {}
    for g in np.unique(lab):
        m = lab == g; im = importancia(A, m); d = (im[R].sum() / im.sum()) / base
        nul = []
        for _ in range(n_nulo):
            mm = np.zeros(len(A), bool); mm[rng.choice(len(A), m.sum(), replace=False)] = True
            i2 = importancia(A, mm); nul.append((i2[R].sum() / i2.sum()) / base)
        out[int(g)] = dict(dano_relativo=float(d), nulo_p05=float(np.percentile(nul, 5)),
                           nulo_p95=float(np.percentile(nul, 95)))
    return out

def prueba_sintetica(semilla=0):
    """Tres grupos plantados con perfiles de bloque distintos; comprueba que se recuperan y que el daño se detecta."""
    from sklearn.metrics import adjusted_rand_score
    rng = np.random.default_rng(semilla); N = 32; n = [1200, 600, 248]
    base = rng.lognormal(0, 0.3, N); perf = [base.copy() for _ in n]
    perf[1][5:10] *= 4; perf[2][24:30] *= 6                      # el grupo 2 (12 %) depende de los bloques 24-29
    A = np.vstack([rng.normal(0, 1, (ni, N)) * p for ni, p in zip(n, perf)])
    real = np.repeat(np.arange(3), n)
    G = descubrir_grupos(A, kmax=5, n_boot=10, semilla=semilla)
    R = list(range(22, 32))                                     # retirada que se lleva los bloques del grupo 2
    D = dano(A, G['etiquetas'], R)
    g2 = np.bincount(G['etiquetas'][real == 2]).argmax()
    return dict(ari=adjusted_rand_score(real, G['etiquetas']), k=G['k'], estabilidad=G['estabilidad'],
                dano_grupo_plantado=D[int(g2)])
