# Stratégie d'Optimisation MoE GPU-First

## 🎯 Découverte Clé

Pour les modèles MoE (comme GPT-OSS-20B), la configuration optimale est **contre-intuitive** :

```bash
✅ Configuration Optimale :
-ngl 99              # TOUTES les couches sur GPU
-nkvo 0              # KV cache sur GPU (pas CPU)
-ncmoe <optimal>     # SEULEMENT les experts sur CPU

❌ Configuration Standard (moins performante) :
-ngl 35              # Certaines couches sur CPU
-nkvo 1              # KV cache sur CPU
-ncmoe 0             # Tous les experts sur GPU
```

## 📊 Impact sur les Performances

### Observation Empirique

**Si vous enlevez NE SERAIT-CE QU'UNE SEULE couche du GPU** (`-ngl 98` au lieu de `99`), les performances **chutent drastiquement**, même si vous exportez tous les experts sur CPU.

### Pourquoi ?

Les modèles MoE ont une architecture particulière :

```
┌─────────────────────────────────────┐
│  Attention Layers (sur GPU)         │  ← Critique pour les perfs
├─────────────────────────────────────┤
│  Router/Gate (sur GPU)              │  ← Doit rester sur GPU
├─────────────────────────────────────┤
│  Expert 1-32                        │  ← Peut aller sur CPU
│    └─ Seulement 1-2 actifs/token    │     (peu d'impact perf)
├─────────────────────────────────────┤
│  Output Layers (sur GPU)            │  ← Critique pour les perfs
└─────────────────────────────────────┘
```

**Points clés** :
- Les couches d'attention et de sortie sont **utilisées pour CHAQUE token**
- Les experts ne sont **activés que sporadiquement** (1-2 sur 32)
- Le router/gate doit **absolument** rester sur GPU pour être rapide

**Donc** :
- Mettre une couche attention/output sur CPU = **goulot d'étranglement constant**
- Mettre des experts sur CPU = **impact minimal** (ils sont rarement utilisés)

## 🛠️ Workflow d'Optimisation Adapté

### Étape 1 : Trouver le Meilleur -ncmoe

```bash
./optimize-moe-advanced.sh models/gpt-oss-20b.gguf
```

Le script va automatiquement :
1. Tester différentes valeurs de `-ncmoe` (0, 5, 10, 15, 20, 25, 30)
2. Optimiser la quantisation du KV cache (`-ctk`, `-ctv`)
3. Optimiser les batch sizes (`-b`, `-ub`)
4. Optimiser les threads CPU (`-t`)

**Durée** : ~30-45 minutes

### Étape 2 : Utiliser la Configuration

Le script vous donnera une commande prête à l'emploi :

```bash
./llama.sh server \
  --model /models/gpt-oss-20b.gguf \
  -ngl 99 \           # TOUT sur GPU
  -nkvo 0 \           # KV sur GPU
  -ncmoe 22 \         # Optimal trouvé
  -ctk q8_0 \         # Quantisation KV
  -ctv q8_0 \
  -b 4096 \           # Batch optimal
  -ub 1024 \
  -t 8 \              # Threads pour experts CPU
  --ctx-size 8192
```

## 📈 Gains Attendus

### Comparaison des Stratégies

| Stratégie | VRAM | RAM | pp512 (t/s) | tg128 (t/s) |
|-----------|------|-----|-------------|-------------|
| **Tout GPU** (`-ngl 99 -ncmoe 0`) | 28 GB | 8 GB | 2400 | **OOM** 💥 |
| **GPU-First** (`-ngl 99 -ncmoe 22`) | 14 GB | 24 GB | 2350 | **52** ✅ |
| **Standard** (`-ngl 35 -ncmoe 0`) | 12 GB | 18 GB | 850 | 28 |
| **CPU Heavy** (`-ngl 20 -ncmoe 30`) | 8 GB | 32 GB | 420 | 15 |

**Conclusion** : La stratégie GPU-First offre le **meilleur compromis** perf/VRAM.

## 🎓 Paramètres à Optimiser

### Paramètres Fixes (Ne PAS toucher)

| Paramètre | Valeur | Raison |
|-----------|--------|--------|
| `-ngl` | `99` | Toutes couches sur GPU = crucial |
| `-nkvo` | `0` | KV cache sur GPU = bien plus rapide |

### Paramètres à Optimiser

#### 1. -ncmoe (CPU MoE Experts)

**Impact** : VRAM vs RAM
- Plus élevé = moins de VRAM, plus de RAM
- Sweet spot généralement entre 15-25 pour GPT-OSS-20B

```bash
# Tester
./optimize-moe-advanced.sh models/gpt-oss-20b.gguf
```

#### 2. Quantisation KV Cache

**Impact** : VRAM vs Précision

| Config | VRAM (ctx=8k) | Qualité |
|--------|---------------|---------|
| `-ctk f16 -ctv f16` | ~4.5 GB | ⭐⭐⭐⭐⭐ |
| `-ctk q8_0 -ctv q8_0` | ~2.3 GB | ⭐⭐⭐⭐ |
| `-ctk q4_0 -ctv q8_0` | ~1.5 GB | ⭐⭐⭐ |

**Recommandation** : `q8_0` pour les deux (excellent compromis)

#### 3. Batch Sizes

**Impact** : Latence vs Débit

| Use Case | `-b` | `-ub` |
|----------|------|-------|
| Chatbot interactif | 2048 | 512 |
| Génération batch | 8192 | 1024 |
| Équilibré | 4096 | 1024 |

#### 4. Threads CPU

**Impact** : Performance des experts sur CPU

- Trop peu = experts lents
- Trop = overhead de synchronisation

**Optimal** : Généralement entre 6-12 threads (selon votre CPU)

## 🚫 Pourquoi llama-optimus N'est PAS Adapté

llama-optimus va systématiquement :
1. ❌ Tenter de réduire `-ngl` pour économiser VRAM
2. ❌ Tester `-nkvo 1` (KV sur CPU)
3. ❌ Ne pas comprendre l'importance de garder TOUT sur GPU

**Pour les MoE avec votre stratégie, utilisez** : `optimize-moe-advanced.sh`

## 📋 Checklist d'Optimisation

- [ ] Vérifier que votre GPU a assez de VRAM (minimum ~12GB pour GPT-OSS-20B)
- [ ] Lancer `./optimize-moe-advanced.sh models/votre-modele.gguf`
- [ ] Noter la valeur optimale de `-ncmoe`
- [ ] Lancer le serveur avec la config complète
- [ ] Vérifier VRAM utilisée avec `nvidia-smi`
- [ ] Benchmark final avec `llama-bench`

## 🔍 Monitoring en Temps Réel

```bash
# Terminal 1 : Serveur
./llama.sh server --model /models/gpt-oss-20b.gguf -ngl 99 -nkvo 0 -ncmoe 22 ...

# Terminal 2 : Monitoring GPU
watch -n 1 nvidia-smi

# Terminal 3 : Monitoring RAM
watch -n 1 free -h
```

## 💡 Tips Avancés

### Si Vous Manquez de VRAM

1. **Augmentez `-ncmoe`** (plus d'experts sur CPU)
2. **Quantifiez le KV cache** (`-ctk q8_0 -ctv q8_0`)
3. **Réduisez le contexte** (`--ctx-size 4096` au lieu de 8192)
4. **En dernier recours** : réduisez `-ngl` (mais attendez-vous à une grosse perte de perf)

### Si Vous Avez de la VRAM à Revendre

1. **Réduisez `-ncmoe`** (gardez plus d'experts sur GPU)
2. **Utilisez f16 pour KV** (`-ctk f16 -ctv f16`)
3. **Augmentez le contexte** (`--ctx-size 16384` ou plus)

## 🎯 Cas d'Usage Réels

### GPT-OSS-20B sur RTX 4090 (24GB)

```bash
./llama.sh server \
  --model /models/gpt-oss-20b-q4.gguf \
  -ngl 99 -nkvo 0 -ncmoe 18 \
  -ctk q8_0 -ctv q8_0 \
  -b 4096 -ub 1024 \
  -t 8 \
  --ctx-size 8192

# Performance attendue : ~55-65 t/s (génération)
```

### GPT-OSS-20B sur RTX 3090 (24GB)

```bash
./llama.sh server \
  --model /models/gpt-oss-20b-q4.gguf \
  -ngl 99 -nkvo 0 -ncmoe 22 \
  -ctk q8_0 -ctv q8_0 \
  -b 2048 -ub 512 \
  -t 6 \
  --ctx-size 8192

# Performance attendue : ~45-55 t/s
```

### GPT-OSS-20B sur RTX 4070 Ti (12GB)

```bash
./llama.sh server \
  --model /models/gpt-oss-20b-q4.gguf \
  -ngl 99 -nkvo 0 -ncmoe 28 \
  -ctk q4_0 -ctv q8_0 \
  -b 2048 -ub 512 \
  -t 10 \
  --ctx-size 4096

# Performance attendue : ~35-45 t/s
```

## 🔬 Pour Aller Plus Loin

### Test de Stabilité

```bash
# Test sur 100 itérations pour vérifier la stabilité
docker compose run --rm llama-cpp \
  llama-bench \
    -m /models/gpt-oss-20b.gguf \
    -ngl 99 -nkvo 0 -ncmoe 22 \
    -ctk q8_0 -ctv q8_0 \
    -b 4096 -ub 1024 -t 8 \
    -p 512 -n 128 -r 100 -o csv > stability.csv
```

### Profiling Détaillé

```bash
# Avec verbose pour voir exactement ce qui se passe
docker compose run --rm llama-cpp \
  llama-bench \
    -m /models/gpt-oss-20b.gguf \
    -ngl 99 -nkvo 0 -ncmoe 22 \
    -v \
    -p 512 -n 128
```

## 📚 Ressources

- **Discussion GitHub sur GPT-OSS** : https://github.com/ggml-org/llama.cpp/discussions/15396
- **Script d'optimisation** : `./optimize-moe-advanced.sh`
- **Documentation MoE** : `LLAMA-OPTIMUS-MOE-NCMOE.md`

---

**TL;DR** : Pour les MoE, gardez **TOUT sur GPU sauf les experts**. N'utilisez PAS llama-optimus standard, utilisez `optimize-moe-advanced.sh`.
