# GGUF Dashboard (Streamlit, local)

Cette partie du repo transforme les scripts en une application web locale Streamlit
multi-pages, focalisée sur :
- Gestion des modèles GGUF (scan local + téléchargement Hugging Face)
- Bench reproductible (llama-bench) + historique
- Sweeps (1D / multi-paramètres) avec garde-fou
- Score combiné pour classer les configs
- Analyse + exports

## Prérequis

- Ubuntu 24.04
- Python 3.10+
- Docker Engine + docker compose plugin
- (Optionnel) NVIDIA GPU + nvidia-container-toolkit

### Vérifier docker

```bash
docker --version
docker compose version
```

### Vérifier GPU (optionnel)

```bash
nvidia-smi
```

## Installation (venv)

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements_dashboard.txt
```

Si vous utilisez des modèles Hugging Face privés :

```bash
export HF_TOKEN="..."
```

## Build du conteneur llama.cpp

```bash
docker compose -f compose.yml build --pull llama-cpp
```

> Le dashboard lance les benches via `docker compose run --rm llama-cpp ...`.

## Lancer le dashboard

```bash
source .venv/bin/activate
streamlit run streamlit_dashboard/streamlit_app.py
```

- Modèles téléchargés/copied dans `./models`
- Historique des runs dans `./runs`

## Formats Hugging Face acceptés

Dans la page **Modèles** :
- `hf.co/<org>/<repo>:<quant>`
- `<org>/<repo>:<quant>`
- `<org>/<repo>` puis sélection manuelle du `.gguf` si plusieurs fichiers

Exemples :
- `hf.co/unsloth/Ministral-8B-Instruct-2410-GGUF:Q4_K_M`
- `TheBloke/Mistral-7B-Instruct-v0.2-GGUF:Q5_K_M`

## Notes

- Le fichier GGUF final est **copié** dans `./models` (pas de symlink).
- Sécurité : aucune commande n'est exécutée via un shell, toutes les commandes sont
  construites en listes d'arguments.
- Stop : le bouton stop tente d'envoyer un SIGINT au process `docker compose`.
