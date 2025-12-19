# GPT-OSS-20B : Guide de Démarrage Rapide

## 🎯 Votre Découverte : La Stratégie GPU-First

Vous avez découvert que pour GPT-OSS-20B (et autres MoE), la **meilleure** configuration est :

```bash
-ngl 99              # ✅ TOUTES les couches sur GPU
-nkvo 0              # ✅ KV cache sur GPU
-ncmoe <optimal>     # ✅ SEULEMENT les experts sur CPU
```

**❌ Ne PAS faire** :
```bash
-ngl 35              # Enlever ne serait-ce qu'une couche = chute drastique des perfs
-nkvo 1              # KV cache sur CPU = beaucoup plus lent
```

## 🚀 Démarrage Ultra-Rapide

### Option 1 : Script Interactif (Recommandé)

```bash
./gpt-oss-quickstart.sh
```

Le script vous propose :
1. **Optimisation complète** (~30-45 min) - trouve la meilleure config
2. **Test rapide** (~2 min) - teste avec des paramètres par défaut
3. **Serveur manuel** - vous spécifiez les paramètres

### Option 2 : Optimisation Manuelle

```bash
# 1. Optimisation complète automatique
./optimize-moe-advanced.sh models/gpt-oss-20b-q4.gguf

# Le script teste :
# - Meilleur -ncmoe (0, 5, 10, 15, 20, 25, 30)
# - Meilleure quantisation KV (f16, q8_0, q4_0)
# - Meilleurs batch sizes (512-8192)
# - Meilleurs threads CPU (4-16)

# 2. Résultat : commande prête à copier-coller
```

### Option 3 : Configuration Manuelle Rapide

```bash
# Si vous connaissez déjà votre config optimale
./llama.sh server \
  --model /models/gpt-oss-20b-q4.gguf \
  -ngl 99 -nkvo 0 -ncmoe 20 \
  -ctk q8_0 -ctv q8_0 \
  -b 4096 -ub 1024 -t 8 \
  --ctx-size 8192
```

## 📊 Configurations Recommandées par GPU

### RTX 4090 (24GB VRAM)

```bash
./llama.sh server \
  --model /models/gpt-oss-20b-q4.gguf \
  -ngl 99 -nkvo 0 -ncmoe 18 \
  -ctk q8_0 -ctv q8_0 \
  -b 4096 -ub 1024 -t 8 \
  --ctx-size 8192

# Performance attendue : ~55-65 tokens/s
```

### RTX 3090 / RTX 4080 (24GB VRAM)

```bash
./llama.sh server \
  --model /models/gpt-oss-20b-q4.gguf \
  -ngl 99 -nkvo 0 -ncmoe 22 \
  -ctk q8_0 -ctv q8_0 \
  -b 2048 -ub 512 -t 6 \
  --ctx-size 8192

# Performance attendue : ~45-55 tokens/s
```

### RTX 4070 Ti (12GB VRAM)

```bash
./llama.sh server \
  --model /models/gpt-oss-20b-q4.gguf \
  -ngl 99 -nkvo 0 -ncmoe 28 \
  -ctk q4_0 -ctv q8_0 \
  -b 2048 -ub 512 -t 10 \
  --ctx-size 4096

# Performance attendue : ~35-45 tokens/s
```

## 🔧 Paramètres Expliqués

| Paramètre | Valeur | Pourquoi |
|-----------|--------|----------|
| `-ngl` | `99` | **CRUCIAL** : Toutes les couches sur GPU. Ne JAMAIS réduire ! |
| `-nkvo` | `0` | KV cache sur GPU (bien plus rapide que CPU) |
| `-ncmoe` | `18-28` | Nombre d'experts (sur 32) à mettre sur CPU pour économiser VRAM |
| `-ctk` | `q8_0` | Quantisation du cache K (économise ~50% VRAM, perte qualité minime) |
| `-ctv` | `q8_0` | Quantisation du cache V |
| `-b` | `2048-8192` | Batch size (plus grand = meilleur débit, plus de VRAM) |
| `-ub` | `512-1024` | Micro-batch (subdivision du batch) |
| `-t` | `6-10` | Threads CPU pour traiter les experts |
| `--ctx-size` | `4096-8192` | Taille du contexte (plus grand = plus de VRAM) |

## 📈 Ajustements Selon Votre Situation

### Vous avez de la VRAM à Revendre ?

```bash
# Baissez -ncmoe pour garder plus d'experts sur GPU
-ncmoe 15   # Au lieu de 22

# Utilisez f16 pour le KV cache
-ctk f16 -ctv f16

# Augmentez le contexte
--ctx-size 16384
```

### Vous Manquez de VRAM ?

```bash
# Montez -ncmoe pour mettre plus d'experts sur CPU
-ncmoe 28   # Au lieu de 22

# Quantifiez plus le KV cache
-ctk q4_0 -ctv q8_0

# Réduisez le contexte
--ctx-size 4096

# Réduisez les batch sizes
-b 2048 -ub 512
```

### Vous Voulez Minimiser la Latence ?

```bash
# Batch sizes plus petits
-b 2048 -ub 512

# Moins d'experts sur CPU (plus rapide)
-ncmoe 18

# KV cache non quantifié
-ctk f16 -ctv f16
```

## 🧪 Benchmarking

### Test Rapide

```bash
./llama.sh bench \
  --model /models/gpt-oss-20b-q4.gguf \
  -ngl 99 -nkvo 0 -ncmoe 20 \
  -p 512 -n 128 -r 3
```

### Test Complet

```bash
./llama.sh bench \
  --model /models/gpt-oss-20b-q4.gguf \
  -ngl 99 -nkvo 0 -ncmoe 20 \
  -ctk q8_0 -ctv q8_0 \
  -b 4096 -ub 1024 -t 8 \
  -p 512 -n 128 -r 10 -o csv > benchmark.csv
```

### Monitoring en Temps Réel

```bash
# Terminal 1 : Serveur
./llama.sh server --model /models/gpt-oss-20b-q4.gguf -ngl 99 -nkvo 0 -ncmoe 20 ...

# Terminal 2 : Monitoring GPU
watch -n 1 nvidia-smi

# Terminal 3 : Monitoring RAM
watch -n 1 free -h
```

## 🆘 Dépannage

### Erreur "Out of Memory" (GPU)

```bash
# Solution 1 : Plus d'experts sur CPU
-ncmoe 28    # Au lieu de 20

# Solution 2 : Quantifier KV cache
-ctk q4_0 -ctv q8_0

# Solution 3 : Réduire contexte
--ctx-size 4096
```

### Erreur "Out of Memory" (RAM)

```bash
# Solution : Moins d'experts sur CPU
-ncmoe 15    # Au lieu de 22

# Note : Vous aurez besoin de plus de VRAM GPU
```

### Performances Décevantes

```bash
# Vérifier que TOUT est sur GPU :
-ngl 99      # Pas 98, pas 50, TOUT !

# Vérifier le KV cache :
-nkvo 0      # Sur GPU, pas sur CPU

# Monitorer pendant le test :
nvidia-smi -l 1
```

## 📚 Documentation Complète

- `STRATEGIE-MOE-GPU-FIRST.md` - Explication détaillée de la stratégie
- `README.md` - Documentation générale de la stack
- `LLAMA-BENCH-README.md` - Tous les paramètres de llama-bench
- `LLAMA-OPTIMUS-MOE-NCMOE.md` - Pourquoi llama-optimus n'est pas adapté

## 🎓 Pour Aller Plus Loin

### Tester Différentes Quantisations du Modèle

```bash
# Q4_K_M (plus petit, plus rapide, qualité correcte)
# Q5_K_M (équilibre)
# Q6_K (meilleure qualité)
# Q8_0 (qualité quasi-parfaite)
```

### Comparer avec d'Autres Modèles MoE

- GPT-OSS-120B (plus gros, 32 experts)
- DeepSeek-V3 (671B paramètres, 256 experts)
- Qwen3-235B-A22B

### Contribuer

Si vous trouvez d'autres optimisations, partagez-les !

## ⚡ TL;DR

```bash
# 1. Lancer l'optimisation automatique
./optimize-moe-advanced.sh models/gpt-oss-20b-q4.gguf

# 2. Copier-coller la commande générée
./llama.sh server --model /models/gpt-oss-20b-q4.gguf ...

# 3. Profiter ! 🚀
```

**Règle d'or** : `-ngl 99 -nkvo 0` toujours, ajustez seulement `-ncmoe` selon votre VRAM.
