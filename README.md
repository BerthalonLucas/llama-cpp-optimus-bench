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

## 🚀 Installation

### Prérequis

- Python 3.10+
- Docker + docker compose
- GPU NVIDIA + nvidia-container-toolkit (recommandé)

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

## 🎯 Utilisation

```bash
source .venv/bin/activate
streamlit run streamlit_dashboard/streamlit_app.py
```

Le dashboard sera accessible sur `http://localhost:8501`

## 📁 Structure du projet

```
├── streamlit_dashboard/
│   ├── streamlit_app.py      # Point d'entrée
│   ├── core/                 # Logique métier
│   │   ├── bench.py          # Exécution llama-bench
│   │   ├── hyperoptimus.py   # Optimisation Optuna
│   │   ├── hf_download.py    # Téléchargement HF
│   │   └── ...
│   ├── pages/                # Pages Streamlit
│   │   ├── 01_Models.py
│   │   ├── 02_Bench.py
│   │   ├── 03_Sweeps.py
│   │   ├── 04_History.py
│   │   └── 05_Optimus.py
│   └── ui/                   # Composants UI
├── compose.yml               # Docker Compose config
├── Dockerfile                # Image llama.cpp + CUDA
├── models/                   # Modèles GGUF (gitignored)
└── runs/                     # Historique des runs (gitignored)
```

## 🤖 HyperOptimus

L'optimiseur automatique utilise **Optuna** avec l'algorithme TPE pour trouver les meilleurs paramètres :

| Paramètre | Plage | Description |
|-----------|-------|-------------|
| `ngl` | 0-999 | Couches GPU |
| `ctx` | 512-32768 | Taille du contexte |
| `batch` | 32-4096 | Taille du batch |
| `ubatch` | 32-2048 | Micro-batch |
| `threads` | 1-32 | Threads CPU |
| `flash_attn` | 0/1 | Flash Attention |

### Score combiné

Le scoring utilise une formule pondérée qui récompense les grands contextes :

```
score = α × TG + β × PP_normalized

PP_normalized = PP × (ctx^0.4 / 1000)
```

- **TG** (Text Generation) : tokens/s pour la génération
- **PP** (Prompt Processing) : tokens/s pour le traitement du prompt
- Les poids α/β sont configurables (défaut: 50/50)

## 🔧 Scripts utilitaires

### llama.sh

Script helper pour les opérations Docker :

```bash
./llama.sh shell              # Shell interactif
./llama.sh bench [args...]    # Lancer llama-bench
./llama.sh server [args...]   # Lancer llama-server
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
