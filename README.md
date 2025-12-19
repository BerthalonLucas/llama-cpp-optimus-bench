# llama-cpp-optimus-bench

Dashboard Streamlit pour benchmarker et optimiser automatiquement les paramètres de **llama.cpp**.

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.40+-red)
![Docker](https://img.shields.io/badge/Docker-Required-blue)
![License](https://img.shields.io/badge/License-MIT-green)

## ✨ Fonctionnalités

- **📦 Gestion des modèles** : Scan local + téléchargement depuis Hugging Face
- **⚡ Benchmark** : Exécution de `llama-bench` avec paramètres personnalisables
- **📊 Sweeps** : Exploration de paramètres (1D ou multi-paramètres)
- **🤖 HyperOptimus** : Optimisation automatique via Optuna (TPE)
- **📈 Historique** : Visualisation et comparaison des runs
- **🖥️ Multi-hardware** : Détection automatique GPU/CPU

## 🚀 Quickstart

```bash
# 1. Cloner et setup (une seule fois)
git clone https://github.com/BerthalonLucas/llama-cpp-optimus-bench.git
cd llama-cpp-optimus-bench
./llama.sh setup

# 2. Placer un modèle GGUF dans models/
cp /path/to/model.gguf models/

# 3. Optimiser !
./llama.sh optimize model.gguf --preset fast
```

## 📦 Prérequis

- **Python 3.10+**
- **Docker** + docker compose plugin
- **GPU NVIDIA** + nvidia-container-toolkit *(recommandé, CPU aussi supporté)*

### Vérifier Docker

```bash
docker --version          # Docker version 24.0+
docker compose version    # Docker Compose version v2.0+
```

### Vérifier GPU (optionnel)

```bash
nvidia-smi               # Doit afficher votre GPU
```

## 🔧 Installation

### Option 1 : Setup automatique (recommandé)

```bash
git clone https://github.com/BerthalonLucas/llama-cpp-optimus-bench.git
cd llama-cpp-optimus-bench
./llama.sh setup
```

Cette commande :
- ✅ Crée les dossiers `models/` et `runs/`
- ✅ Crée l'environnement Python `.venv/`
- ✅ Installe les dépendances
- ✅ Build l'image Docker llama.cpp

### Option 2 : Installation manuelle

```bash
# Cloner
git clone https://github.com/BerthalonLucas/llama-cpp-optimus-bench.git
cd llama-cpp-optimus-bench

# Créer les dossiers
mkdir -p models models/hf-cache runs

# Python venv
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements_dashboard.txt

# Docker
docker compose build --pull llama-cpp
```

### Variables d'environnement (optionnel)

```bash
# Pour télécharger des modèles privés depuis Hugging Face
export HF_TOKEN="hf_xxx..."
```

## 🎯 Utilisation

### Commande `setup` - Installation initiale

```bash
./llama.sh setup
```

### Commande `dashboard` - Interface Web

```bash
./llama.sh dashboard [--port 8510] [--host 0.0.0.0]
```

Ouvre `http://localhost:8510` dans votre navigateur.

### Commande `optimize` - Optimisation CLI

```bash
# Optimisation rapide (~5 min, 20 trials)
./llama.sh optimize my-model.gguf --preset fast

# Optimisation équilibrée (~30 min, 50 trials)
./llama.sh optimize my-model.gguf --preset mid

# Optimisation approfondie (~1-2h, 100 trials)
./llama.sh optimize my-model.gguf --preset high

# Options avancées
./llama.sh optimize my-model.gguf --trials 30 --ctx-max 16384 --ctx-min 4096
```

### Commandes Docker

```bash
# Lancer un serveur avec config optimisée
./llama.sh server --model models/my-model.gguf -ngl 99 -c 16384 --flash-attn

# Benchmark manuel
./llama.sh bench --model /models/my-model.gguf -ngl 99 -p 512 -n 128

# Shell interactif
./llama.sh shell
```

## 📋 Référence des commandes

| Commande | Description |
|----------|-------------|
| `./llama.sh setup` | Installation initiale (venv, deps, docker) |
| `./llama.sh dashboard` | Lance le dashboard web (port 8510) |
| `./llama.sh optimize <model>` | Optimisation HyperOptimus en CLI |
| `./llama.sh server [args]` | Lance llama-server |
| `./llama.sh bench [args]` | Lance llama-bench |
| `./llama.sh shell` | Shell interactif dans le conteneur |

### Options de `optimize`

| Option | Description | Défaut |
|--------|-------------|--------|
| `--preset` | Intensité : `fast`, `mid`, `high` | `fast` |
| `--trials` | Nombre d'essais Optuna | selon preset |
| `--ctx-min` | Contexte minimum | 2048 |
| `--ctx-max` | Contexte maximum | auto (selon VRAM) |
| `--weight-tg` | Poids Text Generation | 0.5 |
| `--weight-pp` | Poids Prompt Processing | 0.3 |
| `--weight-ctx` | Poids Context bonus | 0.2 |

## 🤖 HyperOptimus

L'optimiseur automatique utilise **Optuna** avec l'algorithme TPE pour trouver les meilleurs paramètres :

| Paramètre | Plage | Description |
|-----------|-------|-------------|
| `ctx` | 2048-65536 | Taille du contexte |
| `batch` | 512-8192 | Taille du batch |
| `ubatch` | 256-4096 | Micro-batch |
| `threads` | 4-16 | Threads CPU |
| `flash_attn` | 0/1 | Flash Attention |
| `ngl` | 0-999 | Couches GPU |

### Scoring

```
score = α × TG + β × PP_normalized + γ × CTX_bonus

PP_normalized = PP × (ctx^0.4 / 1000)
CTX_bonus = log2(ctx/1000) × 20
```

- **TG** : vitesse de génération (tokens/s)
- **PP** : vitesse de traitement du prompt (tokens/s)
- **CTX** : bonus pour les grands contextes

## 📁 Structure du projet

```
llama-cpp-optimus-bench/
├── llama.sh                  # 🔧 Script principal (point d'entrée)
├── cli.py                    # 🖥️ CLI Python pour HyperOptimus
├── compose.yml               # 🐳 Docker Compose config
├── Dockerfile                # 🐳 Image llama.cpp + CUDA
├── requirements_dashboard.txt
│
├── models/                   # 📦 Vos modèles GGUF (gitignored)
│   └── hf-cache/             # Cache Hugging Face
│
├── runs/                     # 📊 Historique des runs (gitignored)
│   ├── bench/
│   ├── hyperoptimus/
│   └── sweeps/
│
└── streamlit_dashboard/      # 🌐 Application web
    ├── streamlit_app.py      # Point d'entrée Streamlit
    ├── core/                 # Logique métier
    │   ├── hyperoptimus.py   # Optimisation Optuna
    │   ├── bench.py          # Exécution llama-bench
    │   └── ...
    ├── pages/                # Pages Streamlit
    └── ui/                   # Composants UI
```

## 📝 Formats Hugging Face

Pour télécharger des modèles depuis le dashboard :

| Format | Exemple |
|--------|---------|
| `hf.co/<org>/<repo>:<quant>` | `hf.co/unsloth/Qwen3-0.6B-GGUF:Q4_K_M` |
| `<org>/<repo>:<quant>` | `TheBloke/Mistral-7B-GGUF:Q5_K_M` |
| `<org>/<repo>` | `unsloth/Qwen3-0.6B-GGUF` (sélection manuelle) |

## 🐛 Dépannage

### "Model not found"

```bash
# Le modèle doit être dans models/
ls models/
# ou spécifier le chemin complet
./llama.sh optimize /chemin/absolu/model.gguf
```

### Docker build échoue

```bash
# Vérifier que Docker fonctionne
docker run hello-world

# Rebuild avec logs
docker compose build --pull --no-cache llama-cpp
```

### GPU non détecté

```bash
# Vérifier nvidia-container-toolkit
nvidia-smi
docker run --rm --gpus all nvidia/cuda:12.0-base nvidia-smi
```

## 📄 License

MIT
