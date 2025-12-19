# llama-optimus et les Modèles MoE : Le Flag -ncmoe

## ⚠️ Réponse Directe

**NON**, llama-optimus **n'optimise PAS** le paramètre `-ncmoe` (ou `--n-cpu-moe`).

### Paramètres Optimisés par llama-optimus

| Paramètre | Optimisé ? |
|-----------|-----------|
| `-t, --threads` | ✅ OUI |
| `-b, --batch-size` | ✅ OUI |
| `-ub, --ubatch-size` | ✅ OUI |
| `-ngl, --n-gpu-layers` | ✅ OUI |
| `-fa, --flash-attn` | ✅ OUI |
| `--override-tensor` | ✅ OUI (avec `--test-override-tensor`) |
| **`-ncmoe, --n-cpu-moe`** | ❌ **NON** |

---

## 🤔 C'est Quoi `-ncmoe` ?

### Définition

`-ncmoe` (ou `--n-cpu-moe`) est un paramètre **spécifique aux modèles MoE** (Mixture of Experts) qui spécifie **combien d'experts doivent tourner sur CPU** au lieu du GPU.

### Modèles MoE Concernés

| Modèle | Paramètres Totaux | Paramètres Actifs | Experts |
|--------|-------------------|-------------------|---------|
| **GPT-OSS-20B** | ~20B | ~3B | 32 experts |
| **GPT-OSS-120B** | ~120B | ~7B | 32 experts |
| **DeepSeek-V3** | 671B | ~37B | 256 experts |
| **Mixtral 8x7B** | 47B | ~13B | 8 experts |
| **Qwen3-235B-A22B** | 235B | ~22B | Multiple experts |

### Architecture MoE Expliquée

```
Input Token
     ↓
Router/Gate (décide quel expert utiliser)
     ↓
┌─────────────────────────────────┐
│ Expert 1 │ Expert 2 │ ... │ Expert N │
└─────────────────────────────────┘
     ↓
Output (combinaison des experts activés)
```

**Point Clé** : Seuls quelques experts sont activés par token, pas tous !

---

## 🎯 Utilisation de `-ncmoe`

### Exemple avec GPT-OSS-20B

```bash
# SANS -ncmoe : tous les experts sur GPU (nécessite ~30GB VRAM)
llama-server -m gpt-oss-20b.gguf -ngl 99

# AVEC -ncmoe : 22 experts sur CPU (nécessite seulement ~13GB VRAM)
llama-server -m gpt-oss-20b.gguf -ngl 99 -ncmoe 22
```

### Exemple avec GPT-OSS-120B

```bash
# Avec seulement 8GB VRAM !
llama-server -m gpt-oss-120b.gguf \
  --ctx-size 32768 \
  -b 2048 -ub 2048 \
  -ngl 99 \
  -ncmoe 35  # 35 experts sur CPU
```

### Comment Choisir la Valeur ?

**Formule approximative** :
```
VRAM_disponible = VRAM_totale - (Modèle_taille / Nb_experts × Nb_experts_GPU)
```

**Approche Empirique** (Recommandé) :

1. Commence avec tous les experts sur GPU (`-ncmoe 0`)
2. Si OOM (Out of Memory), augmente `-ncmoe` progressivement
3. Teste la vitesse à chaque étape

```bash
# Test progressif
llama-bench -m gpt-oss-20b.gguf -ngl 99 -ncmoe 0   # Test 1
llama-bench -m gpt-oss-20b.gguf -ngl 99 -ncmoe 5   # Test 2
llama-bench -m gpt-oss-20b.gguf -ngl 99 -ncmoe 10  # Test 3
llama-bench -m gpt-oss-20b.gguf -ngl 99 -ncmoe 15  # Test 4
# Continue jusqu'à trouver le sweet spot
```

---

## 🔧 Workflow : llama-optimus + Modèles MoE

### Étape 1 : Trouver `-ncmoe` Optimal (Manuel)

```bash
# Test rapide pour trouver le bon -ncmoe
for ncmoe in 0 5 10 15 20 25 30; do
  echo "Testing -ncmoe $ncmoe"
  llama-bench -m gpt-oss-20b.gguf -ngl 99 -ncmoe $ncmoe -n 128 -p 512 -r 2
done
```

**Sauvegarde la meilleure valeur**, par exemple : `-ncmoe 22`

### Étape 2 : Optimiser les Autres Paramètres avec llama-optimus

```bash
# Lance llama-optimus (il optimisera threads, batch, etc.)
llama-optimus \
  --model gpt-oss-20b.gguf \
  --trials 30 \
  --metric both

# Résultat exemple :
# Best config: batch=4096, ubatch=1024, threads=6, ngl=99, flash=1
```

### Étape 3 : Combiner les Deux

```bash
# Commande finale : llama-optimus + ton -ncmoe optimal
llama-server \
  --model gpt-oss-20b.gguf \
  -t 6 \                    # ← De llama-optimus
  --batch-size 4096 \       # ← De llama-optimus
  --ubatch-size 1024 \      # ← De llama-optimus
  -ngl 99 \                 # ← De llama-optimus
  --flash-attn \            # ← De llama-optimus
  -ncmoe 22                 # ← Trouvé manuellement !
```

---

## 📊 Impact de `-ncmoe` sur les Perfs

### Exemple : GPT-OSS-20B sur RTX 3070 Ti (8GB)

| Config | VRAM | RAM | pp512 (t/s) | tg128 (t/s) |
|--------|------|-----|-------------|-------------|
| `-ncmoe 0` | OOM 💥 | - | - | - |
| `-ncmoe 16` | 7.8 GB | 15 GB | 850 | 38 |
| `-ncmoe 22` | 6.2 GB | 22 GB | 720 | 32 |
| `-ncmoe 28` | 4.5 GB | 30 GB | 580 | 25 |

**Observation** :
- Plus de CPU experts = moins de VRAM = plus lent
- Il y a un **sweet spot** où tu maximises l'usage GPU sans OOM

---

## 🆚 `-ncmoe` vs `--override-tensor`

### Deux Approches pour les MoE

| Méthode | Niveau | Flexibilité | Facilité |
|---------|--------|-------------|----------|
| **`-ncmoe`** | Couches entières | Basique | ✅ Facile |
| **`--override-tensor`** | Tensors individuels | Avancée | ⚠️ Complexe |

### Exemple avec `--override-tensor` (Avancé)

```bash
# Offload TOUS les experts FFN sur CPU, garde attention sur GPU
llama-server -m qwen3-235b.gguf \
  --override-tensor "\.ffn_.*_exps\.weight=CPU" \
  -ngl 99
```

**Avantage** : Contrôle fin sur quels tensors sont où.

**Inconvénient** : Nécessite de comprendre l'architecture du modèle.

---

## 💡 Pourquoi llama-optimus Ne Supporte Pas `-ncmoe` ?

### Raisons Probables

1. **Complexité** : Chaque modèle MoE a une structure différente
2. **Spécificité** : C'est très hardware-dépendant (VRAM vs RAM)
3. **Scope** : llama-optimus se concentre sur les paramètres "universels"

### L'Approche Actuelle

llama-optimus peut optimiser `--override-tensor` (avec le flag `--test-override-tensor`), ce qui est une approche plus générique et flexible pour les MoE.

```bash
# Optimisation avec override-tensor (expérimental)
llama-optimus \
  --model deepseek-v3.gguf \
  --test-override-tensor \
  --trials 50
```

---

## 🛠️ Script d'Optimisation Complète pour MoE

Voici un script qui combine tout :

```bash
#!/bin/bash

MODEL="gpt-oss-20b.gguf"
LLAMA_BIN="$HOME/llama.cpp/build/bin"

echo "=== Phase 1 : Trouver -ncmoe optimal ==="
best_ncmoe=0
best_tg=0

for ncmoe in 0 5 10 15 20 25 30; do
  echo "Testing -ncmoe $ncmoe..."
  
  result=$($LLAMA_BIN/llama-bench \
    -m "$MODEL" \
    -ngl 99 \
    -ncmoe $ncmoe \
    -n 128 -p 512 -r 2 \
    -o csv 2>/dev/null | grep "tg128" | cut -d',' -f20)
  
  if (( $(echo "$result > $best_tg" | bc -l) )); then
    best_tg=$result
    best_ncmoe=$ncmoe
  fi
  
  echo "  → tg: $result t/s"
done

echo ""
echo "✅ Best -ncmoe: $best_ncmoe (tg: $best_tg t/s)"
echo ""

echo "=== Phase 2 : Optimisation llama-optimus ==="
llama-optimus \
  --llama-bin "$LLAMA_BIN" \
  --model "$MODEL" \
  --trials 30 \
  --metric both

echo ""
echo "=== Phase 3 : Commande Finale ==="
echo "N'oublie pas d'ajouter -ncmoe $best_ncmoe à ta commande !"
```

---

## 📋 Checklist pour MoE

Quand tu travailles avec un modèle MoE :

- [ ] Identifie combien d'experts a ton modèle
- [ ] Trouve `-ncmoe` optimal manuellement
- [ ] (Optionnel) Teste `--override-tensor` si VRAM très limitée
- [ ] Lance llama-optimus pour optimiser les autres params
- [ ] Combine les résultats
- [ ] Benchmark final avec `llama-bench`
- [ ] Vérifie l'utilisation VRAM/RAM avec `nvidia-smi` ou `rocm-smi`

---

## 🎓 Ressources pour MoE

### Documentation llama.cpp

- **Discussion GPT-OSS** : https://github.com/ggml-org/llama.cpp/discussions/15396
- Guide complet sur l'utilisation de `-ncmoe` avec exemples

### Articles Techniques

- **HOBBIT Paper** : Optimisation MoE inference sur hardware limité
- **Qwen3-235B Guide** : Utilisation de `--override-tensor` pour gros MoE

---

## 🎯 Conclusion

### TL;DR

| Question | Réponse |
|----------|---------|
| llama-optimus optimise `-ncmoe` ? | ❌ NON |
| Comment optimiser un MoE alors ? | 1. Trouve `-ncmoe` manuellement<br>2. Lance llama-optimus pour le reste<br>3. Combine les deux |
| Combien de temps ça prend ? | ~15 min pour `-ncmoe` + 30-60 min pour llama-optimus |

### Workflow Recommandé

```bash
# 1. Test rapide -ncmoe
for n in 0 10 20 30; do
  llama-bench -m model.gguf -ncmoe $n -n 128 -r 1
done

# 2. Optimisation automatique
llama-optimus --model model.gguf

# 3. Combine !
llama-server [params_optimisés] -ncmoe [valeur_optimale]
```

C'est un peu plus de travail manuel, mais tu obtiens les **meilleures perfs possibles** pour ton hardware ! 🚀
