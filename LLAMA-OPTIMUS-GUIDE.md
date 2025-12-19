# Guide Complet : llama-optimus

## Qu'est-ce que llama-optimus ?

**llama-optimus** est un outil Python **TIERS** (pas inclus dans llama.cpp) développé par Bruno Arsioli qui **automatise l'optimisation** des paramètres de `llama.cpp` pour maximiser les performances.

### 🎯 Le Problème qu'il Résout

Quand tu utilises llama.cpp, tu as PLEIN de paramètres à régler :
- Nombre de threads CPU (`-t`)
- Batch sizes (`-b`, `-ub`)
- Couches GPU (`-ngl`)
- Flash Attention (`-fa`)
- Override tensors (`-ot`)
- etc.

**Trouver la config optimale manuellement = galère absolue** 😤

Tu peux tester manuellement avec llama-bench :
```bash
# Test 1
llama-bench -m model.gguf -t 4 -b 2048 -ub 512 -ngl 40

# Test 2  
llama-bench -m model.gguf -t 8 -b 4096 -ub 1024 -ngl 40

# Test 3...
# Test 4...
# Test 247...
# 🤯 C'est sans fin !
```

### ✅ La Solution : llama-optimus

**llama-optimus automatise ce processus** en utilisant :
- **Optimisation Bayésienne** (via Optuna) : apprentissage intelligent des meilleurs paramètres
- **Benchmarking automatique** : lance llama-bench pour toi
- **Warm-up du système** : évite les faux résultats (cold start vs steady state)

**Résultat** : Il trouve la config optimale pour TON hardware en ~30-50 tests au lieu de tests manuels infinis.

---

## 📦 C'est Quoi Exactement ?

### Nature du Projet

| Question | Réponse |
|----------|---------|
| **Inclus dans llama.cpp ?** | ❌ NON - Projet tiers séparé |
| **Langage** | 🐍 Python |
| **Dépendance** | Requiert llama.cpp compilé |
| **Licence** | MIT (Open Source) |
| **Repo GitHub** | https://github.com/BrunoArsioli/llama-optimus |
| **PyPI** | https://pypi.org/project/llama-optimus/ |
| **Auteur** | Bruno Arsioli |

### Architecture

```
┌─────────────────┐
│  llama-optimus  │ (Outil Python - optimisation)
└────────┬────────┘
         │ appelle
         ▼
┌─────────────────┐
│  llama-bench    │ (Binaire llama.cpp - benchmarking)
└────────┬────────┘
         │ teste
         ▼
┌─────────────────┐
│  Ton modèle     │ (fichier .gguf)
└─────────────────┘
```

---

## 🚀 Installation

### Prérequis

1. **llama.cpp compilé** avec ton backend (CUDA, ROCm, Vulkan, etc.)
2. **Python 3.8+**
3. **Un modèle GGUF**

### Option 1 : Installation via pip (Recommandé)

```bash
# Simple et rapide
pip install llama-optimus
```

### Option 2 : Installation depuis le repo GitHub

```bash
# Clone le repo
git clone https://github.com/BrunoArsioli/llama-optimus.git
cd llama-optimus

# Crée un environnement virtuel (recommandé)
python3 -m venv .venv
source .venv/bin/activate  # Linux/Mac
# ou
.venv\Scripts\activate  # Windows

# Installation en mode développement
pip install -e .

# OU installation normale
pip install -r requirements.txt
```

### Vérification de l'Installation

```bash
llama-optimus --help
```

Si ça affiche l'aide, c'est bon ! ✅

---

## ⚙️ Configuration

### Variables d'Environnement (Optionnel mais Pratique)

Tu peux définir ces variables pour ne pas avoir à les passer à chaque fois :

```bash
# Linux/Mac - ajoute à ton ~/.bashrc ou ~/.zshrc
export LLAMA_BIN="/path/to/llama.cpp/build/bin"
export MODEL_PATH="/path/to/model.gguf"

# Ou en une ligne pour un test rapide
export LLAMA_BIN="$HOME/llama.cpp/build/bin" && export MODEL_PATH="$HOME/models/llama-7b-q4.gguf"
```

```powershell
# Windows PowerShell
$env:LLAMA_BIN = "C:\path\to\llama.cpp\build\bin"
$env:MODEL_PATH = "C:\path\to\model.gguf"
```

---

## 💡 Utilisation de Base

### Commande Minimale

Si tu as défini les variables d'environnement :

```bash
llama-optimus
```

Sinon, spécifie les chemins :

```bash
llama-optimus \
  --llama-bin /path/to/llama.cpp/build/bin \
  --model /path/to/model.gguf
```

### Ce Qui Se Passe

1. **Phase de Warm-up** (par défaut) :
   - Lance plusieurs benchmarks pour "chauffer" ton système
   - Stabilise CPU/GPU à température de fonctionnement
   - Évite les résultats faussés par le cold start

2. **Phase d'Estimation** :
   - Détermine automatiquement le max de couches GPU supportées (`-ngl`)

3. **Phase d'Optimisation** :
   - Teste différentes combinaisons de paramètres
   - Utilise Optuna (Bayesian optimization) pour être intelligent
   - Converge vers la config optimale

4. **Résultats** :
   - Affiche la meilleure config trouvée
   - Te donne les commandes ready-to-use pour `llama-server` et `llama-bench`

---

## 🎛️ Options et Paramètres

### Paramètres Essentiels

| Flag | Description | Défaut |
|------|-------------|--------|
| `--llama-bin` | Chemin vers le dossier bin de llama.cpp | Var env `LLAMA_BIN` |
| `--model` | Chemin vers le modèle .gguf | Var env `MODEL_PATH` |
| `--metric` | Métrique à optimiser : `pp`, `tg`, ou `both` | `both` |
| `--trials` | Nombre d'essais d'optimisation | 50 |
| `--repeat` | Répétitions par test | 3 |

### Paramètres de Benchmark

| Flag | Description | Défaut |
|------|-------------|--------|
| `--n-tokens` | Nombre de tokens pour le benchmark | 512 |
| `--n-warmup-tokens` | Tokens pour le warm-up | 128 |
| `--warmup-runs` | Nombre de warm-up runs | 30 |
| `--no-warmup` | Désactive le warm-up | false |

### Paramètres Avancés

| Flag | Description |
|------|-------------|
| `--ngl-max` | Force un max de GPU layers (skip auto-detection) |
| `--test-override-tensor` | Active l'optimisation de --override-tensor |
| `--test-flash-attn` | Active l'optimisation de Flash Attention |

---

## 📋 Exemples d'Utilisation

### Exemple 1 : Optimisation Rapide pour Test

```bash
# Test rapide avec peu d'essais (utile pour déboguer)
llama-optimus \
  --llama-bin ~/llama.cpp/build/bin \
  --model ~/models/llama-7b-q4.gguf \
  --trials 10 \
  --repeat 2 \
  --no-warmup \
  --n-tokens 128
```

**Durée** : ~5-10 minutes

### Exemple 2 : Optimisation Standard

```bash
# Config par défaut - bon équilibre temps/qualité
llama-optimus \
  --llama-bin ~/llama.cpp/build/bin \
  --model ~/models/mistral-7b-q8.gguf
```

**Durée** : ~30-60 minutes

### Exemple 3 : Optimisation Complète et Précise

```bash
# Maximum de précision
llama-optimus \
  --llama-bin ~/llama.cpp/build/bin \
  --model ~/models/llama-70b-q4.gguf \
  --trials 100 \
  --repeat 5 \
  --n-tokens 1024
```

**Durée** : 2-3 heures

### Exemple 4 : Optimiser Uniquement la Génération de Texte

```bash
# Focus sur tg (text generation) pour un chatbot interactif
llama-optimus \
  --model ~/models/qwen-7b-q4.gguf \
  --metric tg \
  --trials 50
```

### Exemple 5 : Optimiser Uniquement le Prompt Processing

```bash
# Focus sur pp (prompt processing) pour RAG/search
llama-optimus \
  --model ~/models/embedding-model.gguf \
  --metric pp \
  --trials 50
```

### Exemple 6 : Avec Max GPU Layers Forcé

```bash
# Si tu sais que ton GPU supporte max 35 couches
llama-optimus \
  --model ~/models/llama-13b-q4.gguf \
  --ngl-max 35 \
  --trials 30
```

### Exemple 7 : Test des Override Tensors (Avancé)

```bash
# Pour gros modèles ou VRAM limitée
llama-optimus \
  --model ~/models/deepseek-v3-q4.gguf \
  --test-override-tensor \
  --trials 50
```

---

## 📊 Exemple de Sortie

```
╔══════════════════════════════════════════════════════════╗
║           llama-optimus v0.1.9                           ║
║       Automatic llama.cpp Parameter Optimization         ║
╚══════════════════════════════════════════════════════════╝

📁 Model: /home/user/models/llama-7b-q4.gguf
🎯 Metric: both (pp + tg)
🔢 Trials: 50
🔁 Repeats per test: 3

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

🔥 Warming up system...
[Warmup 1/30] pp: 1234 t/s, tg: 67 t/s
[Warmup 2/30] pp: 1256 t/s, tg: 68 t/s
...
✅ System warmed up (stable performance detected)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

🔍 Estimating max GPU layers...
Testing with -ngl 99...
✅ Estimated max GPU layers: 43

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

🎯 Starting Bayesian optimization...

[Trial 1/50] batch=2048, ubatch=512, threads=8, ngl=43, flash=0
            → pp: 1289 t/s, tg: 71 t/s, score: 680.0

[Trial 2/50] batch=4096, ubatch=1024, threads=4, ngl=43, flash=1
            → pp: 1456 t/s, tg: 73 t/s, score: 764.5
            🎉 New best!

[Trial 3/50] batch=4096, ubatch=512, threads=6, ngl=40, flash=1
            → pp: 1401 t/s, tg: 69 t/s, score: 735.0

...

[Trial 50/50] batch=3584, ubatch=896, threads=4, ngl=43, flash=1
             → pp: 1467 t/s, tg: 74 t/s, score: 770.5
             🎉 New best!

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

✅ OPTIMIZATION COMPLETE!

Best configuration found:
┌─────────────────┬────────┐
│ batch_size      │ 3584   │
│ ubatch_size     │ 896    │
│ threads         │ 4      │
│ gpu_layers      │ 43     │
│ flash_attention │ 1      │
└─────────────────┴────────┘

Performance:
• Prompt Processing: 1467 t/s
• Text Generation:   74 t/s
• Combined Score:    770.5

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

📋 READY-TO-USE COMMANDS:

For optimal inference with llama-server:
─────────────────────────────────────────
llama-server \
  --model /home/user/models/llama-7b-q4.gguf \
  -t 4 \
  --batch-size 3584 \
  --ubatch-size 896 \
  -ngl 43 \
  --flash-attn

For benchmarking:
─────────────────
llama-bench \
  --model /home/user/models/llama-7b-q4.gguf \
  -t 4 \
  --batch-size 3584 \
  --ubatch-size 896 \
  -ngl 43 \
  --flash-attn 1 \
  -n 128 -p 512 -r 3 -o csv

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

---

## 🔧 Workflow Recommandé

### 1. Première Optimisation (Complète)

```bash
# Lance une optimisation complète la première fois
llama-optimus \
  --model ~/models/ton-modele.gguf \
  --trials 50 \
  --repeat 3
```

**Sauvegarde les résultats** (copie la commande llama-server générée) !

### 2. Tests Rapides Après Changement Hardware

```bash
# Si tu changes ton hardware ou drivers
llama-optimus \
  --model ~/models/ton-modele.gguf \
  --trials 20 \
  --repeat 2 \
  --no-warmup
```

### 3. Optimisation par Usage

```bash
# Pour un chatbot (priorité tg)
llama-optimus --model model.gguf --metric tg --trials 30

# Pour RAG/Embedding (priorité pp)
llama-optimus --model model.gguf --metric pp --trials 30
```

---

## 🎯 Comprendre les Métriques

### `--metric pp` (Prompt Processing)

**Optimise pour** :
- Traitement rapide de gros contextes
- Applications RAG
- Embedding models
- Search/retrieval

**Cas d'usage** :
- Tu charges beaucoup de documents
- Tu fais de la recherche sémantique
- Ton app traite de gros prompts

### `--metric tg` (Text Generation)

**Optimise pour** :
- Génération de tokens rapide
- Chatbots interactifs
- Streaming de réponses

**Cas d'usage** :
- Chatbot en temps réel
- Code completion
- Génération créative

### `--metric both` (Défaut)

**Optimise pour** :
- Équilibre entre pp et tg
- Usage général

**Cas d'usage** :
- Assistant polyvalent
- Tu fais un peu de tout

---

## 🏆 Avantages vs Tests Manuels

| Aspect | Tests Manuels | llama-optimus |
|--------|---------------|---------------|
| **Temps** | Heures à jours | 30-60 min |
| **Nombre de configs testées** | 10-50 (si courageux) | 50-100+ |
| **Approche** | Random/Intuition | Bayesian (intelligent) |
| **Warm-up** | Souvent oublié | Automatique |
| **Reproductibilité** | Difficile | Excellente |
| **Risque d'erreur** | Élevé | Faible |

---

## 💡 Tips et Astuces

### 1. Warm-up : Important ou Pas ?

**OUI, très important !** ❗

Le "cold start" (première exécution) donne des résultats **20-30% plus rapides** que la steady state (après chauffage).

```bash
# Sans warm-up : résultats faussés !
llama-optimus --no-warmup  # ❌

# Avec warm-up : résultats réalistes
llama-optimus  # ✅
```

**Exception** : Désactive le warm-up seulement pour debug rapide.

### 2. Nombre de Trials : Combien ?

| Trials | Durée | Qualité | Usage |
|--------|-------|---------|-------|
| 10-20 | 10-15 min | Basique | Test rapide |
| 30-50 | 30-60 min | Bien | Usage standard ⭐ |
| 80-100 | 1-2h | Excellent | Production |
| 150+ | 2-4h | Optimal | Benchmarking sérieux |

### 3. Répétitions : 3 ou 5 ?

```bash
# 3 répétitions : bon équilibre (défaut)
llama-optimus --repeat 3

# 5 répétitions : plus précis mais plus long
llama-optimus --repeat 5
```

**Conseil** : 3 est largement suffisant dans 99% des cas.

### 4. Optimiser pour Ton Use Case Réel

```bash
# Si ton use case = prompts de 2048 tokens + génération de 256
llama-optimus \
  --n-tokens 2304 \  # 2048 + 256
  --metric both
```

### 5. Multi-GPU ?

llama-optimus ne gère **pas encore** l'optimisation multi-GPU automatique.

Pour multi-GPU, fais l'optimisation sur 1 GPU puis ajoute manuellement :
```bash
# Après optimisation
llama-server \
  [paramètres optimisés par llama-optimus] \
  -ts 0.5,0.5  # Split 50/50 sur 2 GPUs
```

---

## 🐛 Troubleshooting

### Erreur : "llama-bench not found"

```bash
# Vérifie que llama-bench existe
ls ~/llama.cpp/build/bin/llama-bench

# Si absent, recompile llama.cpp
cd ~/llama.cpp
cmake --build build --config Release
```

### Erreur : "Model not found"

```bash
# Vérifie le chemin
ls /path/to/model.gguf

# Utilise le chemin absolu
llama-optimus --model "$(realpath ~/models/model.gguf)"
```

### Résultats Instables

```bash
# Augmente les répétitions
llama-optimus --repeat 5

# Augmente le warm-up
llama-optimus --warmup-runs 50
```

### Out of Memory pendant l'Optimisation

```bash
# Force un max GPU layers plus bas
llama-optimus --ngl-max 30

# Ou réduis les tokens de test
llama-optimus --n-tokens 256
```

### Optimisation Très Lente

```bash
# Réduis les trials pour test rapide
llama-optimus --trials 20 --repeat 2

# Désactive warm-up (attention : résultats moins fiables)
llama-optimus --no-warmup
```

---

## 🔍 Comparaison avec Alternatives

### vs Tests Manuels avec llama-bench

**llama-optimus** :
- ✅ Automatique et intelligent
- ✅ Explore plus de configs
- ✅ Gère le warm-up
- ❌ Nécessite installation Python

**Tests manuels** :
- ✅ Pas de dépendance Python
- ✅ Contrôle total
- ❌ Chronophage
- ❌ Pas d'approche intelligente

### vs Grid Search Script

**llama-optimus** (Bayesian) :
- ✅ Intelligent : apprend des tests précédents
- ✅ Converge plus vite
- ✅ Moins de tests nécessaires

**Grid Search** (tous les combos) :
- ❌ Bête : teste tout
- ❌ Extrêmement long
- ✅ Garantie de trouver l'optimum (si espace petit)

---

## 📚 Ressources

- **Repo GitHub** : https://github.com/BrunoArsioli/llama-optimus
- **PyPI** : https://pypi.org/project/llama-optimus/
- **llama.cpp** : https://github.com/ggml-org/llama.cpp
- **Optuna** : https://optuna.org/

---

## 🎬 Conclusion

**llama-optimus** est un outil **essentiel** si tu veux obtenir les **meilleures performances** de llama.cpp sans passer des heures à tester manuellement.

### Quand l'utiliser ?

✅ **OUI** si :
- Tu veux les meilleures perfs possibles
- Tu changes de hardware
- Tu déploies en production
- Tu as 30-60 min devant toi

❌ **NON** si :
- Tu débutes avec llama.cpp (utilise les défauts d'abord)
- Tu veux juste tester rapidement
- Tu changes souvent de modèle

### Quick Start Résumé

```bash
# Installation
pip install llama-optimus

# Usage simple
llama-optimus \
  --llama-bin ~/llama.cpp/build/bin \
  --model ~/models/model.gguf

# Résultat = Commande optimale pour llama-server !
```

**C'est tout ! 🚀**
