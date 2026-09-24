"""Reconstruye las 2.048 muestras de OpenHermes-2.5 que usó Multiverse para Llama-3.1-8B, Llama-3.3-70B y Qwen3-14B.

Su código (configs/cbo_configs y src/binary_optimization/prepare_binary_optimization.py) hace:
    dataset = load_from_disk(...)   # OpenHermes-2.5 train completo, solo tokenizado (map no filtra ni reordena)
    dataset = dataset.shuffle(seed=42)
    dataset = dataset.select(range(2048))
Aquí se repite lo mismo sobre el conjunto sin tokenizar, que tiene la misma longitud y el mismo orden.

Uso (necesita internet y unos 2 GB libres):
    pip install datasets
    python3 exportar_muestras.py
Deja muestras_openhermes_2048.csv; súbelo.
"""
import csv
from datasets import load_dataset

ds = load_dataset("teknium/OpenHermes-2.5", split="train")
print("filas en OpenHermes-2.5:", len(ds))          # anótalo: si no es 1.001.551, el orden puede no coincidir
ds = ds.add_column("indice_original", list(range(len(ds))))
sel = ds.shuffle(seed=42).select(range(2048))

with open("muestras_openhermes_2048.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["fila_A", "indice_original", "source", "category", "turnos", "caracteres", "inicio_usuario"])
    for i, ex in enumerate(sel):
        conv = ex.get("conversations") or []
        texto = " ".join((c.get("value") or "") for c in conv)
        usuario = next(((c.get("value") or "") for c in conv if c.get("from") == "human"), "")
        w.writerow([i, ex["indice_original"], ex.get("source") or "", ex.get("category") or "",
                    len(conv), len(texto), usuario[:300].replace("\n", " ")])
print("escrito muestras_openhermes_2048.csv")
