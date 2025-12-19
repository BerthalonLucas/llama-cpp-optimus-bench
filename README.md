# Stack llama.cpp + llama-optimus (Docker)

## Streamlit Dashboard (local)
Voir **README_DASHBOARD.md** pour lancer l'app Streamlit (gestion GGUF, bench, sweeps, historique).

## Vue d'ensemble
- Image : `ghcr.io/ggml-org/llama.cpp:full-cuda` + Python + `llama-optimus`.
- Orchestration : `docker compose` (GPU requis, port 8080 pour `llama-server`).
- Stockage : tout vit dans `./models` (modèles + cache HF). Mappé dans le conteneur sur `/models`.
- Script helper : `./llama.sh` simplifie le build, le bench, l'optimisation et le support MoE.

## Prérequis
- Docker + docker compose avec runtime GPU configuré (ex. `nvidia-container-toolkit` pour CUDA).
- Pilotes GPU installés, `nvidia-smi` fonctionnel si CUDA.
- Espace disque dans `./models`. Optionnel : exporte `HF_TOKEN` ou `HUGGINGFACEHUB_API_TOKEN` pour les modèles privés.

## Démarrage rapide
```bash
# 1) Shell dans l'image (build automatique si nécessaire)
./llama.sh shell

# 2) Benchmark simple (remplace par ton modèle local ou -hf)
./llama.sh bench --model /models/mistral.gguf -ngl 99

# 3) Lancer un serveur HTTP
./llama.sh server --model /models/mistral.gguf -ngl 99 --ctx-size 4096
```

## Gestion des modèles
- Local : place les fichiers `.gguf` dans `./models` et référence-les via `models/<fichier>` ou chemin absolu.
- Hugging Face : `-hf <repo>` télécharge dans `./models` (cache `./models/hf-cache`). Exemple :
  ```bash
  ./llama.sh bench -hf TheBloke/Mixtral-8x7B-GGUF --hf-file mixtral-8x7b.Q4_K_M.gguf -ngl 99
  ```

## Commandes principales (`./llama.sh`)
- `server [args...]` : lance `llama-server` sur 8080 (build auto).
- `bench [args...]` : lance `llama-bench` avec tes options.
- `shell [args...]` : shell interactif dans l'image.
- `optimus-fast|optimus-mid|optimus-high <model>|-hf <repo> [--hf-file file] [args...]`  
  - fast ~10‑15 min (smoke) : `--trials 12 --repeat 1 --no-warmup --n-tokens 192`
  - mid ~30‑60 min (équilibré) : `--trials 40 --repeat 3 --n-tokens 512`
  - high 1‑3 h (approfondi) : `--trials 100 --repeat 5 --n-tokens 1024`
  - Exemple : `./llama.sh optimus-mid models/mistral.gguf --metric tg`

## Workflow MoE (`-ncmoe`)
> llama-optimus **n'optimise pas** `-ncmoe`. Utilise le scan dédié puis optimise le reste.

### ⚡ Stratégie GPU-First (Recommandée pour Haute Performance)

**Pour maximiser les performances sur GPU puissant** : gardez **TOUTES** les couches sur GPU et offloadez **SEULEMENT** les experts sur CPU.

```bash
# Optimisation complète automatique (GPU-First)
./legacy/optimize-moe-advanced.sh models/gpt-oss-20b.gguf

# Résultat : config optimale avec -ngl 99 -nkvo 0 -ncmoe <optimal>
```

📖 **Voir** `STRATEGIE-MOE-GPU-FIRST.md` pour les détails.

### 🔧 Workflow Standard (Flexibilité VRAM)

1) **Scanner -ncmoe** (alias `moe-scan` / `moe-sweep`)  
   ```bash
   # Valeurs par défaut : "0 5 10 15 20 25 30"
   ./llama.sh moe-sweep models/mixtral.gguf --ngl 99 --ncmoe-values "0 5 10 15 20 25 30"
   # Ajouter des args llama-bench après -- (ex: flash attn activé)
   ./llama.sh moe-sweep -hf TheBloke/Mixtral-8x7B-GGUF --hf-file mixtral-8x7b.Q4_K_M.gguf \
     --ngl 99 --ncmoe-values "0 10 20 30" -- -fa 1
   ```
   - Le script boucle sur chaque `-ncmoe`, récupère le meilleur `tg` (tokens/s) et imprime la valeur gagnante.

2) **Optimiser le reste avec llama-optimus**  
   - Choisis un preset (`optimus-fast/mid/high`) **sans** `-ncmoe` (il optimise threads/batch/flash/ubatch/ngl, etc.).

3) **Combiner**  
   - Utilise les flags optimus + `-ncmoe <best>` dans tes commandes finales `llama-server` / `llama-bench`.

## Conseils rapides
- `LLAMA_BIN_DIR` (défaut `/app`) peut être pointé vers un autre binaire si besoin.
- Les caches HF : `HF_HOME=/models`, `HF_HUB_CACHE=/models/hf-cache`, `HUGGINGFACE_HUB_CACHE=/models/hf-cache`.
- Pour tester rapidement un MoE : réduis `--n-gen`/`--n-prompt` et `--repeat` dans `moe-sweep`.
- Pour un usage production, préfère `optimus-mid` ou `optimus-high` après le scan MoE.
