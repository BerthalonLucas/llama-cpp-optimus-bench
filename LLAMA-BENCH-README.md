# Guide Complet : llama-bench

## Table des Matières

1. [Introduction](#introduction)
2. [Installation et Compilation](#installation-et-compilation)
3. [Syntaxe de Base](#syntaxe-de-base)
4. [Options Générales](#options-générales)
5. [Paramètres de Test](#paramètres-de-test)
6. [Paramètres Backend et GPU](#paramètres-backend-et-gpu)
7. [Paramètres Mémoire et Cache](#paramètres-mémoire-et-cache)
8. [Comprendre les Résultats](#comprendre-les-résultats)
9. [Formats de Sortie](#formats-de-sortie)
10. [Exemples d'Utilisation](#exemples-dutilisation)
11. [Optimisation des Performances](#optimisation-des-performances)
12. [Résolution de Problèmes](#résolution-de-problèmes)

---

## Introduction

`llama-bench` est l'outil officiel de benchmarking de **llama.cpp** pour tester les performances d'inférence des modèles LLM. Il mesure deux métriques critiques :

- **pp** (Prompt Processing) : Vitesse de traitement du contexte/prompt
- **tg** (Text Generation) : Vitesse de génération de nouveaux tokens

### Points Importants

⚠️ **Les mesures n'incluent PAS** :
- La tokenisation (conversion texte → tokens)
- Le sampling (sélection du prochain token)

✅ **Les prompts sont générés aléatoirement** (séquences de tokens aléatoires) - tu ne contrôles que la longueur, pas le contenu.

---

## Installation et Compilation

### Compilation de Base

```bash
# Cloner llama.cpp
git clone https://github.com/ggml-org/llama.cpp.git
cd llama.cpp

# Compilation CPU uniquement
cmake -B build
cmake --build build --config Release
```

### Compilation avec Backends Spécifiques

#### CUDA (NVIDIA)
```bash
cmake -B build -DGGML_CUDA=ON
cmake --build build --config Release
```

#### ROCm (AMD)
```bash
cmake -B build -DGGML_ROCM=ON
cmake --build build --config Release
```

#### Vulkan (NVIDIA/AMD/Intel)
```bash
cmake -B build -DGGML_VULKAN=ON
cmake --build build --config Release
```

#### Metal (Apple Silicon)
```bash
cmake -B build -DGGML_METAL=ON
cmake --build build --config Release
```

L'exécutable sera dans : `build/bin/llama-bench`

---

## Syntaxe de Base

```bash
llama-bench [options]
```

**Utilisation minimale** :
```bash
llama-bench -m /path/to/model.gguf
```

---

## Options Générales

### Options de Contrôle

| Flag | Type | Défaut | Description |
|------|------|--------|-------------|
| `-h, --help` | - | - | Affiche l'aide |
| `-r, --repetitions <n>` | entier | 5 | Nombre de répétitions par test |
| `--delay <0...N>` | secondes | 0 | Délai entre chaque test |
| `--prio <0\|1\|2\|3>` | entier | 0 | Priorité du processus/thread |
| `-v, --verbose` | booléen | false | Mode verbeux avec détails |
| `--progress` | booléen | false | Affiche des indicateurs de progression |
| `--list-devices` | - | - | Liste les devices disponibles et quitte |

### Options de Sortie

| Flag | Type | Défaut | Description |
|------|------|--------|-------------|
| `-o, --output <format>` | string | md | Format de sortie vers stdout |
| `-oe, --output-err <format>` | string | none | Format de sortie vers stderr |

**Formats disponibles** : `csv`, `json`, `jsonl`, `md` (markdown), `sql`

### Options NUMA

| Flag | Description |
|------|-------------|
| `--numa <distribute\|isolate\|numactl>` | Mode NUMA pour systèmes multi-socket |
| - `disabled` (défaut) | Pas d'optimisation NUMA |
| - `distribute` | Distribue les threads sur tous les nœuds NUMA |
| - `isolate` | Isole les threads sur un nœud NUMA |
| - `numactl` | Utilise numactl externe |

### RPC (Remote Procedure Call)

| Flag | Description |
|------|-------------|
| `-rpc, --rpc <rpc_servers>` | Liste de serveurs RPC (séparés par virgules) |

---

## Paramètres de Test

### Modèles

| Flag | Type | Défaut | Description |
|------|------|--------|-------------|
| `-m, --model <filename>` | string | models/7B/ggml-model-q4_0.gguf | Chemin vers le modèle GGUF |

**💡 Astuce** : Tu peux spécifier plusieurs modèles pour les tester d'un coup :
```bash
llama-bench -m model1.gguf -m model2.gguf -m model3.gguf
```

### Paramètres de Prompt et Génération

| Flag | Type | Défaut | Description |
|------|------|--------|-------------|
| `-p, --n-prompt <n>` | entier | 512 | Taille du prompt en tokens |
| `-n, --n-gen <n>` | entier | 128 | Nombre de tokens à générer |
| `-pg <pp,tg>` | string | - | Test mixte : prompt + génération |
| `-d, --n-depth <n>` | entier | 0 | Profondeur de test (0 = auto) |

#### Utilisation de -pg (Prompt + Generation combinés)

Le flag `-pg` permet de tester un scénario plus réaliste : traiter un prompt puis générer immédiatement.

```bash
# Test avec 1024 tokens de prompt suivi de 256 tokens générés
llama-bench -m model.gguf -pg 1024,256
```

**Sortie** :
```
| test          | t/s    |
|---------------|--------|
| pp512         | 162.54 |  <- Traitement du prompt seul
| tg128         | 22.50  |  <- Génération seule
| pp1024+tg256  | 63.51  |  <- Test combiné réaliste
```

### Tailles de Batch

| Flag | Type | Défaut | Description |
|------|------|--------|-------------|
| `-b, --batch-size <n>` | entier | 2048 | Taille du batch principal |
| `-ub, --ubatch-size <n>` | entier | 512 | Taille du micro-batch |

**📌 Explication** :
- `batch-size` : Nombre maximum de tokens traités ensemble
- `ubatch-size` : Subdivision du batch pour optimiser la mémoire GPU

---

## Paramètres Backend et GPU

### Threads CPU

| Flag | Type | Défaut | Description |
|------|------|--------|-------------|
| `-t, --threads <n>` | entier | détection auto | Nombre de threads CPU |
| `-C, --cpu-mask <hex,hex>` | hex | 0x0 | Masque CPU pour affinité |
| `--cpu-strict <0\|1>` | booléen | 0 | Affinité CPU stricte |
| `--poll <0...100>` | entier | 50 | Pourcentage de polling |

### GPU et Offloading

| Flag | Type | Défaut | Description |
|------|------|--------|-------------|
| `-ngl, --n-gpu-layers <n>` | entier | 99 | Nombre de couches sur GPU |
| `-sm, --split-mode <mode>` | string | layer | Mode de division |
| `-mg, --main-gpu <i>` | entier | 0 | GPU principal (multi-GPU) |
| `-nkvo, --no-kv-offload <0\|1>` | booléen | 0 | Désactive l'offload du cache KV |
| `-ts, --tensor-split <ts0/ts1/...>` | string | 0 | Split des tensors sur multi-GPU |
| `-dev, --device <dev0/dev1/...>` | string | auto | Sélection manuelle des devices |

#### Split Modes

- **none** : Pas de split
- **layer** : Split par couche (recommandé)
- **row** : Split par ligne de tenseur

#### Exemples GPU

```bash
# Toutes les couches sur GPU
llama-bench -m model.gguf -ngl 999

# 40 couches sur GPU, reste sur CPU
llama-bench -m model.gguf -ngl 40

# Multi-GPU : split 50/50
llama-bench -m model.gguf -ts 0.5,0.5

# Spécifier le device
llama-bench -m model.gguf -dev cuda:0
```

---

## Paramètres Mémoire et Cache

### Types de Cache KV

| Flag | Type | Défaut | Description |
|------|------|--------|-------------|
| `-ctk, --cache-type-k <t>` | string | f16 | Type de cache pour K |
| `-ctv, --cache-type-v <t>` | string | f16 | Type de cache pour V |

**Types disponibles** :
- `f16` : Float16 (équilibre mémoire/précision)
- `f32` : Float32 (plus précis, plus de mémoire)
- `q8_0`, `q4_0`, `q4_1` : Quantifiés (économie mémoire)

### Memory Mapping

| Flag | Type | Défaut | Description |
|------|------|--------|-------------|
| `-mmp, --mmap <0\|1>` | booléen | 1 | Active le memory mapping |

**mmap** : Mappe le fichier en mémoire au lieu de le charger entièrement. Réduit l'utilisation RAM.

### Embeddings

| Flag | Type | Défaut | Description |
|------|------|--------|-------------|
| `-embd, --embeddings <0\|1>` | booléen | 0 | Mode embeddings uniquement |

### Override Tensors (Avancé)

| Flag | Description |
|------|-------------|
| `-ot, --override-tensor <pattern>=<type>;...` | Override le type de buffer pour certains tensors |

**Exemple** :
```bash
llama-bench -m model.gguf -ot "*.weight=f16;output.weight=f32"
```

---

## Paramètres d'Optimisation

### Flash Attention

| Flag | Type | Défaut | Description |
|------|------|--------|-------------|
| `-fa, --flash-attn <0\|1>` | booléen | 0 | Active Flash Attention |

**Flash Attention** :
- ✅ Réduit l'utilisation mémoire du contexte
- ✅ Légère amélioration de vitesse sur CUDA
- ⚠️ Peut ralentir certains backends (limitations d'implémentation actuelle)

```bash
llama-bench -m model.gguf -fa 1
```

### Expert Models (MoE)

| Flag | Type | Défaut | Description |
|------|------|--------|-------------|
| `-ncmoe, --n-cpu-moe <n>` | entier | 0 | Nombre d'experts MoE sur CPU |

---

## Comprendre les Résultats

### Format de Sortie Markdown (Défaut)

```
| model              | size      | params   | backend    | threads | test      | t/s           |
|--------------------|-----------|----------|------------|---------|-----------|---------------|
| llama 7B Q4_0      | 3.83 GiB  | 6.74 B   | CUDA       | 8       | pp512     | 2413.34 ± 8.3 |
| llama 7B Q4_0      | 3.83 GiB  | 6.74 B   | CUDA       | 8       | tg128     | 132.05 ± 0.4  |
```

### Colonnes Expliquées

| Colonne | Description |
|---------|-------------|
| **model** | Nom et type du modèle |
| **size** | Taille du fichier GGUF |
| **params** | Nombre de paramètres |
| **backend** | Backend utilisé (CPU, CUDA, Metal, Vulkan, etc.) |
| **threads** | Nombre de threads CPU |
| **ngl** | Couches sur GPU (si applicable) |
| **fa** | Flash Attention activée (0 ou 1) |
| **test** | Type de test (voir ci-dessous) |
| **t/s** | Tokens par seconde ± écart-type |

### Types de Tests

| Test | Format | Description |
|------|--------|-------------|
| **pp512** | `pp<N>` | Prompt Processing avec N tokens |
| **tg128** | `tg<N>` | Text Generation de N tokens |
| **pp1024+tg256** | `pp<N>+tg<M>` | Test combiné |

**Interprétation** :

- **pp512 = 2413 t/s** : Traite 2413 tokens de contexte par seconde
  - Si tu as 4000 tokens de contexte → temps = 4000/2413 = **1.66 secondes**
  
- **tg128 = 132 t/s** : Génère 132 tokens par seconde (batch size = 1)
  - Pour générer 500 tokens → temps = 500/132 = **3.8 secondes**

---

## Formats de Sortie

### 1. Markdown (md) - Défaut

```bash
llama-bench -m model.gguf -o md
```

Tableau lisible, parfait pour documentation.

### 2. CSV

```bash
llama-bench -m model.gguf -o csv
```

**Exemple de sortie** :
```csv
build_commit,build_number,cpu_info,gpu_info,backends,model_filename,model_type,model_size,model_n_params,n_batch,n_ubatch,n_threads,n_gpu_layers,flash_attn,n_prompt,n_gen,test_time,avg_ns,stddev_ns,avg_ts,stddev_ts
"8cf427ff","5163","AMD Ryzen 7 7800X3D","NVIDIA RTX 4080","CUDA","model.gguf","qwen2 7B Q4_K",4677120000,7615616512,2048,512,8,99,0,512,0,"2025-04-24T11:57:09Z",70285660,982040,7285.68,100.06
```

Idéal pour analyse dans Excel, pandas, etc.

### 3. JSON

```bash
llama-bench -m model.gguf -o json
```

**Exemple** :
```json
[
  {
    "build_commit": "8cf427ff",
    "build_number": 5163,
    "cpu_info": "AMD Ryzen 7 7800X3D",
    "gpu_info": "NVIDIA RTX 4080",
    "model_type": "qwen2 7B Q4_K",
    "n_batch": 2048,
    "n_threads": 8,
    "n_gpu_layers": 99,
    "flash_attn": false,
    "n_prompt": 512,
    "avg_ts": 7285.68,
    "stddev_ts": 100.06
  }
]
```

### 4. JSONL (JSON Lines)

```bash
llama-bench -m model.gguf -o jsonl
```

Une ligne JSON par test, parfait pour streaming.

### 5. SQL

```bash
llama-bench -m model.gguf -o sql
```

**Exemple** :
```sql
CREATE TABLE IF NOT EXISTS test (
  build_commit TEXT,
  build_number INTEGER,
  cpu_info TEXT,
  gpu_info TEXT,
  model_type TEXT,
  n_batch INTEGER,
  avg_ts REAL,
  ...
);

INSERT INTO test (...) VALUES (...);
```

Directement importable dans une base SQL.

---

## Exemples d'Utilisation

### Test Simple d'un Modèle

```bash
# Test basique avec réglages par défaut
llama-bench -m models/llama-7b-q4_0.gguf
```

### Test avec Flash Attention

```bash
llama-bench -m model.gguf -fa 1
```

### Test Multiple Prompt Sizes

```bash
# Teste avec différentes tailles de prompt
llama-bench -m model.gguf -p 128,256,512,1024,2048
```

**Sortie** :
```
| test   | t/s    |
|--------|--------|
| pp128  | 3200   |
| pp256  | 2800   |
| pp512  | 2400   |
| pp1024 | 2100   |
| pp2048 | 1800   |
```

### Test Multiple Generation Sizes

```bash
llama-bench -m model.gguf -n 64,128,256,512
```

### Test Combiné Réaliste

```bash
# Simule un usage réel : prompt de 1024 tokens + génération de 512 tokens
llama-bench -m model.gguf -pg 1024,512 -r 10
```

### Comparer Plusieurs Modèles

```bash
llama-bench \
  -m models/llama2-7b-q4.gguf \
  -m models/llama2-7b-q8.gguf \
  -m models/mistral-7b-q4.gguf \
  -p 512 -n 128 -o csv > results.csv
```

### Test GPU Offloading (progression)

```bash
# Teste différents niveaux d'offload GPU
llama-bench -m model.gguf -ngl 0,10,20,30,40 -p 512 -n 128
```

### Test Range de Valeurs

llama-bench supporte des **ranges** avec 3 formats :

```bash
# Format 1: first-last (pas de 1)
llama-bench -m model.gguf -p 128-512  # → 128,129,130...512

# Format 2: first-last+step
llama-bench -m model.gguf -p 128-512+128  # → 128,256,384,512

# Format 3: first-last*mult
llama-bench -m model.gguf -p 128-2048*2  # → 128,256,512,1024,2048
```

### Benchmark Complet du Matériel

```bash
#!/bin/bash
MODEL="models/llama-7b-q4_0.gguf"

echo "=== Benchmark CPU Only ==="
llama-bench -m $MODEL -ngl 0 -t 4,8,12,16 -o csv > cpu_bench.csv

echo "=== Benchmark GPU Offload ==="
llama-bench -m $MODEL -ngl 10,20,30,40,999 -o csv > gpu_bench.csv

echo "=== Benchmark Flash Attention ==="
llama-bench -m $MODEL -fa 0,1 -ngl 999 -o csv > flash_bench.csv

echo "=== Benchmark Batch Sizes ==="
llama-bench -m $MODEL -b 512,1024,2048,4096 -o csv > batch_bench.csv
```

### Output vers Fichier ET Écran

```bash
# Markdown à l'écran, CSV dans fichier
llama-bench -m model.gguf -o md | tee results.txt
llama-bench -m model.gguf -o csv > results.csv
```

### Benchmark Multi-GPU

```bash
# Test split sur 2 GPUs
llama-bench -m model.gguf -ts 0.5,0.5 -mg 0

# Test split inégal (70/30)
llama-bench -m model.gguf -ts 0.7,0.3
```

### Test avec Delay entre Tests (refroidissement)

```bash
# 30 secondes entre chaque test pour stabiliser température
llama-bench -m model.gguf -p 512,1024,2048 --delay 30
```

### Augmenter la Précision (plus de répétitions)

```bash
# 20 répétitions au lieu de 5 par défaut
llama-bench -m model.gguf -r 20
```

### Test de Stabilité Longue Durée

```bash
# 100 répétitions pour détecter throttling thermique
llama-bench -m model.gguf -r 100 -o csv > stability_test.csv
```

---

## Optimisation des Performances

### 1. Trouver la Configuration Optimale

#### Script d'Optimisation Manuelle

```bash
#!/bin/bash
MODEL="model.gguf"

echo "Testing thread count..."
llama-bench -m $MODEL -t 2,4,6,8,10,12,16 -ngl 999 -o csv | grep tg

echo "Testing batch sizes..."
llama-bench -m $MODEL -b 256,512,1024,2048,4096 -ngl 999 -o csv | grep tg

echo "Testing ubatch sizes..."
llama-bench -m $MODEL -ub 128,256,512,1024 -ngl 999 -o csv | grep tg

echo "Testing Flash Attention..."
llama-bench -m $MODEL -fa 0,1 -ngl 999 -o csv | grep tg
```

#### Utilisation de llama-optimus (automatique)

```bash
# Installer llama-optimus
pip install llama-optimus

# Optimisation automatique
llama-optimus --model model.gguf --n-tokens 512

# Sortie exemple :
# Best config: {'batch': 4096, 'flash': 1, 'u_batch': 1024, 'threads': 4, 'gpu_layers': 93}
# Best tg tokens/sec: 73.5
```

### 2. Profiling par Backend

#### CPU (BLAS)

```bash
# Tester threads optimaux
llama-bench -m model.gguf -ngl 0 -t 1-16 -o csv
```

Généralement optimal = nombre de P-cores (pas les E-cores).

#### GPU CUDA/ROCm

```bash
# Test GPU layers optimal
llama-bench -m model.gguf -ngl 10-50+5 -o csv

# Test batch sizes
llama-bench -m model.gguf -b 512,1024,2048,4096,8192 -ngl 999
```

#### GPU Vulkan

```bash
# Vulkan peut être sensible aux batch sizes
llama-bench -m model.gguf -ngl 999 -b 512,1024,2048 -ub 256,512
```

### 3. Configurations Recommandées par Scénario

#### Latence Minimale (chatbot interactif)

```bash
llama-bench -m model.gguf \
  -ngl 999 \          # Tout sur GPU
  -fa 1 \             # Flash Attention
  -b 2048 \           # Batch size modéré
  -ub 512 \           # Ubatch optimisé
  -p 512 -n 128       # Test prompt typique
```

#### Débit Maximum (batch inference)

```bash
llama-bench -m model.gguf \
  -ngl 999 \
  -fa 1 \
  -b 8192 \           # Gros batch
  -ub 1024 \
  -p 2048 -n 512
```

#### Économie Mémoire (low VRAM)

```bash
llama-bench -m model.gguf \
  -ngl 20 \           # Seulement 20 couches
  -fa 1 \             # Flash Attention économise mémoire
  -ctk q8_0 \         # Cache K quantifié
  -ctv q8_0 \         # Cache V quantifié
  -b 1024 -ub 256
```

---

## Résolution de Problèmes

### Erreur: "Out of Memory"

**Solutions** :
```bash
# 1. Réduire les couches GPU
llama-bench -m model.gguf -ngl 20

# 2. Quantifier le cache
llama-bench -m model.gguf -ctk q8_0 -ctv q8_0

# 3. Réduire batch size
llama-bench -m model.gguf -b 512 -ub 128

# 4. Désactiver Flash Attention
llama-bench -m model.gguf -fa 0
```

### Performance Très Lente

**Diagnostic** :
```bash
# 1. Vérifier si GPU est utilisé
llama-bench -m model.gguf --list-devices

# 2. Tester avec toutes couches sur GPU
llama-bench -m model.gguf -ngl 999

# 3. Vérifier pas de swap
llama-bench -m model.gguf -v  # Mode verbose
```

### Backend Non Détecté

```bash
# Vérifier les backends compilés
llama-bench --help | grep -i backend

# Lister devices disponibles
llama-bench --list-devices
```

**Solutions** :
- Recompiler llama.cpp avec le bon backend
- Vérifier drivers GPU installés
- Pour CUDA : vérifier `nvidia-smi`
- Pour ROCm : vérifier `rocm-smi`

### Résultats Instables (écart-type élevé)

**Solutions** :
```bash
# 1. Augmenter répétitions
llama-bench -m model.gguf -r 20

# 2. Ajouter un délai entre tests
llama-bench -m model.gguf --delay 10

# 3. Fixer les threads (éviter auto-détection)
llama-bench -m model.gguf -t 8
```

### GPU Sous-Utilisé

```bash
# Augmenter batch sizes
llama-bench -m model.gguf -b 4096 -ub 1024 -ngl 999

# Vérifier avec nvidia-smi pendant le test
watch -n 1 nvidia-smi
```

---

## Comparaison des Métriques

### pp (Prompt Processing)

**Ce qui compte** :
- ✅ **Compute bound** : limité par la puissance de calcul
- ✅ Bénéficie des gros batch sizes
- ✅ Très sensible à l'accélération GPU

**Optimiser** :
```bash
llama-bench -m model.gguf -p 2048 -b 4096 -ngl 999 -fa 1
```

### tg (Text Generation)

**Ce qui compte** :
- ✅ **Memory bandwidth bound** : limité par la bande passante mémoire
- ✅ Moins sensible aux batch sizes (test en batch=1)
- ✅ Scaling limité avec plus de GPU

**Optimiser** :
```bash
# Modèle quantifié léger
llama-bench -m model-q4.gguf -n 128 -ngl 999

# Formule approximative
# tg_speed ≈ Memory_Bandwidth / Model_Size
```

---

## Tips et Astuces

### 1. Benchmark Reproductible

```bash
# Fixe tout pour résultats identiques
llama-bench -m model.gguf \
  -t 8 \              # Threads fixés
  -ngl 40 \           # GPU layers fixées
  -b 2048 -ub 512 \   # Batches fixés
  -r 10 \             # Répétitions constantes
  --prio 3            # Priorité max
```

### 2. Monitoring GPU Pendant Test

```bash
# Terminal 1
llama-bench -m model.gguf -ngl 999 -r 20

# Terminal 2
watch -n 0.5 nvidia-smi
```

### 3. Tester Throttling Thermique

```bash
# Long test pour détecter baisse de perf
llama-bench -m model.gguf -ngl 999 -r 100 --delay 5 -o csv > thermal_test.csv

# Analyser les résultats
python3 -c "
import pandas as pd
df = pd.read_csv('thermal_test.csv')
print('Variation: ', df['avg_ts'].std() / df['avg_ts'].mean() * 100, '%')
"
```

### 4. Test Réaliste de Production

```bash
# Simule usage avec prompt variable et génération
llama-bench -m model.gguf \
  -pg 512,128 \       # Chat courte
  -pg 2048,256 \      # Chat longue
  -pg 4096,512 \      # Document analysis
  -r 10 -o csv
```

### 5. Comparer Quantizations

```bash
# Script de comparaison
for quant in q4_0 q4_k_m q8_0 f16; do
  echo "Testing $quant..."
  llama-bench -m "model-$quant.gguf" -p 512 -n 128 -o csv >> quant_compare.csv
done
```

---

## Références

- **Repo officiel** : https://github.com/ggml-org/llama.cpp
- **Documentation README** : https://github.com/ggml-org/llama.cpp/blob/master/tools/llama-bench/README.md
- **llama-optimus** : https://pypi.org/project/llama-optimus/
- **Guide llama.cpp complet** : https://blog.steelph0enix.dev/posts/llama-cpp-guide/

---

## Synthèse des Commandes Courantes

```bash
# Test simple
llama-bench -m model.gguf

# Test complet avec optimisations
llama-bench -m model.gguf -ngl 999 -fa 1 -b 4096 -ub 1024 -r 10

# Comparer plusieurs modèles
llama-bench -m model1.gguf -m model2.gguf -m model3.gguf -o csv

# Test prompt sizes variées
llama-bench -m model.gguf -p 128,256,512,1024,2048

# Test réaliste combiné
llama-bench -m model.gguf -pg 1024,256 -r 10

# Benchmark hardware complet
llama-bench -m model.gguf -ngl 0-50+10 -t 4,8,12,16 -b 512,1024,2048 -o csv

# Output vers CSV pour analyse
llama-bench -m model.gguf -o csv > results.csv

# Multi-GPU
llama-bench -m model.gguf -ts 0.5,0.5

# Économie mémoire
llama-bench -m model.gguf -ngl 20 -ctk q8_0 -ctv q8_0 -fa 1
```

---

**Enjoy benchmarking ! 🚀**
