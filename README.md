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

## 🚀 Quickstart (2 commandes!)

```bash
# 1. Setup
git clone https://github.com/BerthalonLucas/llama-cpp-optimus-bench.git
cd llama-cpp-optimus-bench
python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements_dashboard.txt
docker compose build --pull llama-cpp

# 2. Optimiser un modèle
./llama.sh optimize models/my-model.gguf --preset fast
```

## 🎯 Utilisation

### Option 1: Interface Web (Dashboard)

```bash
./llama.sh dashboard
```

Ouvre `http://localhost:8510` dans votre navigateur.

### Option 2: Ligne de commande

```bash
# Optimisation rapide (~5 min)
./llama.sh optimize models/my-model.gguf --preset fast

# Optimisation complète (~30 min)
./llama.sh optimize models/my-model.gguf --preset mid

# Optimisation approfondie (~1-2h)
./llama.sh optimize models/my-model.gguf --preset high
```

### Option 3: Opérations Docker directes

```bash
./llama.sh server --model models/my-model.gguf -ngl 99 -c 16384
./llama.sh bench --model /models/my-model.gguf
./llama.sh shell
```

## 📦 Installation complète

### Prérequis

- Python 3.10+
- Docker + docker compose
- GPU NVIDIA + nvidia-container-toolkit (recommandé, CPU aussi supporté)

### Setup

```bash
# Cloner le repo
git clone https://github.com/BerthalonLucas/llama-cpp-optimus-bench.git
cd llama-cpp-optimus-bench

# Créer l'environnement virtuel
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements_dashboard.txt

# Build du conteneur llama.cpp
docker compose build --pull llama-cpp
```

### Variables d'environnement (optionnel)

```bash
export HF_TOKEN="..."  # Pour modèles Hugging Face privés
```

## 🔧 Commandes llama.sh

| Commande | Description |
|----------|-------------|
| `./llama.sh dashboard` | Lance le dashboard web (port 8510) |
| `./llama.sh optimize <model>` | Optimisation HyperOptimus en CLI |
| `./llama.sh server [args]` | Lance llama-server |
| `./llama.sh bench [args]` | Lance llama-bench |
| `./llama.sh shell` | Shell interactif dans le conteneur |

### Options de `optimize`

```bash
./llama.sh optimize <model> [options]

Options:
  --preset fast|mid|high    Intensité (default: fast)
  --trials N                Nombre d'essais
  --ctx-max N               Contexte maximum
  --ctx-min N               Contexte minimum
  --weight-tg F             Poids TG (default: 0.5)
  --weight-pp F             Poids PP (default: 0.3)
```

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

### Score combiné

```
score = α × TG + β × PP_normalized + γ × CTX_bonus

PP_normalized = PP × (ctx^0.4 / 1000)
CTX_bonus = log2(ctx/1000) × 20
```

## 📁 Structure du projet

```
├── cli.py                    # CLI HyperOptimus
├── llama.sh                  # Script principal
├── streamlit_dashboard/
│   ├── streamlit_app.py      # Dashboard web
│   ├── core/                 # Logique métier
│   │   ├── hyperoptimus.py   # Optimisation Optuna
│   │   └── ...
│   └── pages/                # Pages Streamlit
├── compose.yml               # Docker Compose
├── models/                   # Modèles GGUF (gitignored)
└── runs/                     # Historique (gitignored)
```

## 📝 Formats Hugging Face

Le dashboard accepte plusieurs formats :

- `hf.co/<org>/<repo>:<quant>` 
- `<org>/<repo>:<quant>`
- `<org>/<repo>` (sélection manuelle du fichier)

Exemples :
```
unsloth/Ministral-8B-Instruct-2410-GGUF:Q4_K_M
TheBloke/Mistral-7B-Instruct-v0.2-GGUF:Q5_K_M
```

## 📄 License

MIT
