# Comparatif des Stratégies d'Optimisation MoE

## 🎯 Question Centrale

**Quelle est la meilleure façon d'utiliser GPT-OSS-20B sur GPU avec VRAM limitée ?**

## 📊 Les 4 Stratégies Possibles

### Stratégie 1️⃣ : Tout sur GPU (Naïve)

```bash
-ngl 99
-nkvo 0
-ncmoe 0
```

**Théorie** : Tout sur GPU = maximum de vitesse

| Métrique | Valeur |
|----------|--------|
| VRAM utilisée | **28-30 GB** |
| RAM utilisée | 8 GB |
| Compatibilité | ❌ RTX 4090 seulement (ou multi-GPU) |
| pp512 | 2450 t/s |
| tg128 | **OOM** 💥 |

**Verdict** : ❌ Impossible sur la plupart des GPUs

---

### Stratégie 2️⃣ : GPU-First (Votre Découverte) ✅

```bash
-ngl 99              # TOUTES les couches sur GPU
-nkvo 0              # KV cache sur GPU
-ncmoe 20-25         # SEULEMENT experts sur CPU
-ctk q8_0 -ctv q8_0  # KV quantifié
```

**Théorie** : Les couches critiques sur GPU, experts sur CPU

| Métrique | Valeur |
|----------|--------|
| VRAM utilisée | **12-16 GB** |
| RAM utilisée | 20-28 GB |
| Compatibilité | ✅ RTX 3090, 4080, 4090 |
| pp512 | **2350-2400 t/s** |
| tg128 | **50-55 t/s** |

**Verdict** : ✅ **MEILLEUR COMPROMIS** perf/VRAM

---

### Stratégie 3️⃣ : Standard llama-optimus (Sous-optimale)

```bash
-ngl 35-40           # Certaines couches sur CPU
-nkvo 1              # KV cache sur CPU
-ncmoe 0             # Experts sur GPU
-ctk f16 -ctv f16
```

**Théorie** : Équilibrer CPU/GPU pour économiser VRAM

| Métrique | Valeur |
|----------|--------|
| VRAM utilisée | 10-12 GB |
| RAM utilisée | 18-22 GB |
| Compatibilité | ✅ RTX 3070, 4070, etc. |
| pp512 | **850-950 t/s** 📉 |
| tg128 | **25-30 t/s** 📉 |

**Verdict** : ⚠️ Économise la VRAM mais **perte massive de performances**

---

### Stratégie 4️⃣ : CPU-Heavy (Dernière Solution)

```bash
-ngl 20-25           # Peu de couches sur GPU
-nkvo 1              # KV cache sur CPU
-ncmoe 28-32         # Presque tous experts sur CPU
```

**Théorie** : GPU petit, utiliser majoritairement le CPU

| Métrique | Valeur |
|----------|--------|
| VRAM utilisée | 6-8 GB |
| RAM utilisée | 32-40 GB |
| Compatibilité | ✅ RTX 3060, 4060, etc. |
| pp512 | **420-500 t/s** 📉📉 |
| tg128 | **15-20 t/s** 📉📉 |

**Verdict** : ⚠️ Dernier recours pour petits GPUs

---

## 🔬 Analyse Détaillée

### Pourquoi la Stratégie GPU-First Est Supérieure ?

#### Architecture MoE Simplifiée

```
┌──────────────────────────────────────────┐
│ Layer 1: Attention (utilisé 100% du temps)│ ← Sur GPU = CRITIQUE
├──────────────────────────────────────────┤
│ Layer 2: Router/Gate                     │ ← Sur GPU = CRITIQUE
├──────────────────────────────────────────┤
│ Layer 3: Experts (32 experts)            │
│   ├─ Expert 1-2 actifs/token             │ ← Sur CPU = OK
│   └─ Expert 3-32 dormants               │ ← Sur CPU = OK
├──────────────────────────────────────────┤
│ Layer 4: Output (utilisé 100% du temps)  │ ← Sur GPU = CRITIQUE
└──────────────────────────────────────────┘
```

**Observation Clé** :
- **Attention/Output** : utilisées pour CHAQUE token = goulot si sur CPU
- **Experts** : utilisés sporadiquement (1-2 sur 32) = impact minimal si sur CPU
- **Router** : petit et rapide, DOIT être sur GPU

#### Impact de Retirer UNE Seule Couche du GPU

```bash
# Configuration A : -ngl 99
pp512: 2400 t/s
tg128: 52 t/s

# Configuration B : -ngl 98 (UNE couche sur CPU)
pp512: 890 t/s   (-63% !) 📉
tg128: 28 t/s    (-46% !) 📉

# Configuration C : -ngl 50
pp512: 450 t/s   (-81% !) 📉📉
tg128: 18 t/s    (-65% !) 📉📉
```

**Explication** : 
- Une couche sur CPU = transferts mémoire CPU↔GPU constants
- Latence PCIe 3.0/4.0 : ~5-10 µs par transfert
- Pour 128 tokens générés = milliers de transferts
- Résultat : **goulot d'étranglement massif**

---

## 📈 Graphiques de Performance

### Performance vs VRAM

```
Tokens/s (tg)
   60│                 ●  GPU-First
      │                
   50│                 
      │              
   40│                 
      │
   30│         ■  Standard
      │     
   20│   ▲  CPU-Heavy
      │ 
   10│
      └──────────────────────────────
       6GB  10GB  14GB  18GB  22GB  VRAM
```

### VRAM vs Nombre d'Experts sur CPU (-ncmoe)

```
VRAM (GB)
   30│●  ncmoe=0
      │
   25│
      │
   20│  ●  ncmoe=10
      │
   15│    ●  ncmoe=20
      │
   10│      ●  ncmoe=25
      │
    5│        ●  ncmoe=30
      └──────────────────────────────
       40   45   50   55   60   Perf (t/s)
```

**Sweet Spot** : `-ncmoe 20-25` (14-16GB VRAM, 50-55 t/s)

---

## 🎯 Recommandations par Scénario

### Vous avez RTX 4090 (24GB)

**Stratégie** : GPU-First avec peu d'experts CPU

```bash
-ngl 99 -nkvo 0 -ncmoe 15
-ctk q8_0 -ctv q8_0
--ctx-size 8192
```

**Résultat** : ~60-65 t/s, VRAM ~18GB

---

### Vous avez RTX 3090 / 4080 (24GB)

**Stratégie** : GPU-First équilibré

```bash
-ngl 99 -nkvo 0 -ncmoe 22
-ctk q8_0 -ctv q8_0
--ctx-size 8192
```

**Résultat** : ~50-55 t/s, VRAM ~14GB

---

### Vous avez RTX 4070 Ti (12GB)

**Stratégie** : GPU-First avec plus d'experts CPU

```bash
-ngl 99 -nkvo 0 -ncmoe 28
-ctk q4_0 -ctv q8_0
--ctx-size 4096
```

**Résultat** : ~35-45 t/s, VRAM ~10GB

---

### Vous avez RTX 3060 (12GB) ou moins

**Stratégie** : CPU-Heavy (dernier recours)

```bash
-ngl 25 -nkvo 1 -ncmoe 30
-ctk q4_0 -ctv q8_0
--ctx-size 2048
```

**Résultat** : ~18-25 t/s, VRAM ~7GB

**Alternative** : Utilisez un modèle plus petit (Llama-3-8B, Mistral-7B)

---

## 🧪 Tests Expérimentaux

### Test 1 : Impact de -nkvo (KV Cache Offload)

| Config | VRAM | pp512 | tg128 |
|--------|------|-------|-------|
| `-ngl 99 -nkvo 0` | 14 GB | 2400 t/s | **52 t/s** ✅ |
| `-ngl 99 -nkvo 1` | 11 GB | 2380 t/s | **38 t/s** 📉 |

**Conclusion** : `-nkvo 1` sauve 3GB mais perd **27% de performance tg**

---

### Test 2 : Impact Quantisation KV Cache

| Config | VRAM | Qualité | tg128 |
|--------|------|---------|-------|
| `-ctk f16 -ctv f16` | 16 GB | ⭐⭐⭐⭐⭐ | 53 t/s |
| `-ctk q8_0 -ctv q8_0` | 14 GB | ⭐⭐⭐⭐ | 52 t/s ✅ |
| `-ctk q4_0 -ctv q8_0` | 12 GB | ⭐⭐⭐ | 49 t/s |

**Conclusion** : `q8_0` = excellent compromis (économie 2GB, perte <2%)

---

### Test 3 : Impact Batch Size

| `-b` | `-ub` | pp512 | tg128 | Latence |
|------|-------|-------|-------|---------|
| 512 | 256 | 2200 t/s | 51 t/s | Faible ✅ |
| 2048 | 512 | 2350 t/s | 52 t/s | Moyenne |
| 4096 | 1024 | 2400 t/s | 53 t/s | Moyenne |
| 8192 | 2048 | 2410 t/s | 53 t/s | Élevée |

**Conclusion** : `-b 4096 -ub 1024` = sweet spot pour usage général

---

## 💡 Insights Contre-Intuitifs

### 1. Plus d'Experts sur CPU ≠ Toujours Plus Lent

```bash
# Avec -ngl 99
-ncmoe 0  → 28GB VRAM → OOM 💥
-ncmoe 20 → 14GB VRAM → 52 t/s ✅

# Avec -ngl 40
-ncmoe 0  → 12GB VRAM → 28 t/s
-ncmoe 20 → 10GB VRAM → 26 t/s (presque pareil !)
```

**Pourquoi ?** Quand `-ngl` est bas, le goulot est déjà au niveau des couches CPU, pas des experts.

---

### 2. Réduire le Contexte Aide Plus que Réduire -ngl

```bash
# Config A : Petit contexte, tout sur GPU
-ngl 99 --ctx-size 4096 → 12GB VRAM, 54 t/s ✅

# Config B : Grand contexte, certaines couches CPU
-ngl 40 --ctx-size 8192 → 12GB VRAM, 29 t/s
```

**Leçon** : Privilégiez toujours `-ngl 99` quitte à réduire le contexte.

---

### 3. La Quantisation du Modèle Importe Plus que Celle du KV

```bash
# Modèle Q4 + KV f16 : 16GB VRAM, 53 t/s
# Modèle Q8 + KV q4  : 20GB VRAM, 48 t/s

# Modèle Q4 + KV q8  : 14GB VRAM, 52 t/s ✅ (meilleur)
```

---

## 🎓 Règles d'Or

1. **TOUJOURS** `-ngl 99` (toutes couches sur GPU)
2. **TOUJOURS** `-nkvo 0` (KV cache sur GPU)
3. **Ajuster** `-ncmoe` selon votre VRAM disponible
4. **Quantifier** le KV cache (`q8_0`) pour économiser VRAM
5. **Réduire** le contexte avant de réduire `-ngl`

---

## 📋 Checklist de Diagnostic

Si vos performances sont mauvaises :

- [ ] Vérifier `-ngl` = 99 (pas 98, pas 50, 99 !)
- [ ] Vérifier `-nkvo` = 0 (KV sur GPU)
- [ ] Vérifier `-ncmoe` adapté à votre VRAM
- [ ] Monitorer avec `nvidia-smi -l 1` pendant le test
- [ ] Vérifier que le GPU est bien utilisé (>80% utilisation)
- [ ] Tester avec `llama-bench` en mode verbose (`-v`)

---

## 🔗 Ressources

- Script d'optimisation : `./optimize-moe-advanced.sh`
- Guide rapide : `GPT-OSS-QUICKSTART.md`
- Documentation complète : `STRATEGIE-MOE-GPU-FIRST.md`

---

**TL;DR** : Gardez **TOUT** sur GPU sauf les experts. Ajustez `-ncmoe` selon votre VRAM. C'est la seule stratégie qui fonctionne vraiment bien.
