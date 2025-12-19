#!/usr/bin/env bash
set -euo pipefail

# Script d'optimisation pour modèles MoE avec stratégie :
# - TOUTES les couches sur GPU (-ngl 99)
# - KV cache sur GPU (-nkvo 0)
# - SEULEMENT les experts sur CPU (-ncmoe)
# - Optimisation fine : batch sizes, quantisation KV, threads

MODEL="${1:-}"
if [[ -z "$MODEL" ]]; then
  echo "Usage: $0 <model.gguf> [best_ncmoe]"
  echo "Example: $0 models/gpt-oss-20b.gguf 22"
  exit 1
fi

BEST_NCMOE="${2:-}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Résoudre le chemin du modèle
if [[ "$MODEL" != /* ]]; then
  MODEL="$SCRIPT_DIR/$MODEL"
fi

if [[ ! -f "$MODEL" ]]; then
  echo "❌ Model not found: $MODEL"
  exit 1
fi

# Chemin dans le conteneur
CONTAINER_MODEL="/models/$(basename "$MODEL")"

echo "╔══════════════════════════════════════════════════════════╗"
echo "║   Optimisation MoE Avancée (GPU-first strategy)          ║"
echo "╚══════════════════════════════════════════════════════════╝"
echo ""
echo "📁 Model: $MODEL"
echo "🎯 Strategy: ALL layers on GPU + experts on CPU"
echo ""

# Étape 1 : Trouver le meilleur -ncmoe si non fourni
if [[ -z "$BEST_NCMOE" ]]; then
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
  echo "🔍 Phase 1/4 : Scan -ncmoe optimal"
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
  
  NCMOE_VALUES="0 5 10 15 20 25 30"
  best_ncmoe=0
  best_tg=0
  
  for ncmoe in $NCMOE_VALUES; do
    echo "Testing -ncmoe $ncmoe (ngl=99, nkvo=0)..."
    
    result=$(docker compose run --rm llama-cpp \
      llama-bench \
        -m "$CONTAINER_MODEL" \
        -ngl 99 -nkvo 0 -ncmoe "$ncmoe" \
        -p 512 -n 128 -r 2 \
        -o csv 2>/dev/null | grep "tg128" | awk -F',' '{print $(NF-1)}' || echo "0")
    
    echo "  → tg: $result t/s"
    
    if (( $(echo "$result > $best_tg" | bc -l 2>/dev/null || echo 0) )); then
      best_tg=$result
      best_ncmoe=$ncmoe
    fi
  done
  
  echo ""
  echo "✅ Best -ncmoe: $best_ncmoe (tg: $best_tg t/s)"
  BEST_NCMOE=$best_ncmoe
else
  echo "ℹ️  Using provided -ncmoe: $BEST_NCMOE"
fi

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🧪 Phase 2/4 : Optimisation quantisation KV cache"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

best_ctk="f16"
best_ctv="f16"
best_kv_tg=0

for ctk in f16 q8_0 q4_0; do
  for ctv in f16 q8_0; do
    echo "Testing KV cache: ctk=$ctk ctv=$ctv..."
    
    result=$(docker compose run --rm llama-cpp \
      llama-bench \
        -m "$CONTAINER_MODEL" \
        -ngl 99 -nkvo 0 -ncmoe "$BEST_NCMOE" \
        -ctk "$ctk" -ctv "$ctv" \
        -p 512 -n 128 -r 2 \
        -o csv 2>/dev/null | grep "tg128" | awk -F',' '{print $(NF-1)}' || echo "0")
    
    echo "  → tg: $result t/s"
    
    if (( $(echo "$result > $best_kv_tg" | bc -l 2>/dev/null || echo 0) )); then
      best_kv_tg=$result
      best_ctk=$ctk
      best_ctv=$ctv
    fi
  done
done

echo ""
echo "✅ Best KV cache: -ctk $best_ctk -ctv $best_ctv (tg: $best_kv_tg t/s)"

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "📦 Phase 3/4 : Optimisation batch sizes"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

best_batch=2048
best_ubatch=512
best_batch_tg=0

for batch in 512 1024 2048 4096 8192; do
  for ubatch in 256 512 1024; do
    if (( ubatch > batch )); then
      continue
    fi
    
    echo "Testing batch=$batch ubatch=$ubatch..."
    
    result=$(docker compose run --rm llama-cpp \
      llama-bench \
        -m "$CONTAINER_MODEL" \
        -ngl 99 -nkvo 0 -ncmoe "$BEST_NCMOE" \
        -ctk "$best_ctk" -ctv "$best_ctv" \
        -b "$batch" -ub "$ubatch" \
        -p 512 -n 128 -r 2 \
        -o csv 2>/dev/null | grep "tg128" | awk -F',' '{print $(NF-1)}' || echo "0")
    
    echo "  → tg: $result t/s"
    
    if (( $(echo "$result > $best_batch_tg" | bc -l 2>/dev/null || echo 0) )); then
      best_batch_tg=$result
      best_batch=$batch
      best_ubatch=$ubatch
    fi
  done
done

echo ""
echo "✅ Best batch sizes: -b $best_batch -ub $best_ubatch (tg: $best_batch_tg t/s)"

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "⚙️  Phase 4/4 : Optimisation threads CPU (pour experts)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

best_threads=8
best_threads_tg=0

for threads in 4 6 8 10 12 16; do
  echo "Testing threads=$threads..."
  
  result=$(docker compose run --rm llama-cpp \
    llama-bench \
      -m "$CONTAINER_MODEL" \
      -ngl 99 -nkvo 0 -ncmoe "$BEST_NCMOE" \
      -ctk "$best_ctk" -ctv "$best_ctv" \
      -b "$best_batch" -ub "$best_ubatch" \
      -t "$threads" \
      -p 512 -n 128 -r 3 \
      -o csv 2>/dev/null | grep "tg128" | awk -F',' '{print $(NF-1)}' || echo "0")
  
  echo "  → tg: $result t/s"
  
  if (( $(echo "$result > $best_threads_tg" | bc -l 2>/dev/null || echo 0) )); then
    best_threads_tg=$result
    best_threads=$threads
  fi
done

echo ""
echo "✅ Best threads: -t $best_threads (tg: $best_threads_tg t/s)"

# Résumé final
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "✅ OPTIMISATION TERMINÉE"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "Configuration optimale trouvée :"
echo "┌─────────────────────┬──────────┐"
echo "│ GPU Layers          │ 99       │ (TOUT sur GPU)"
echo "│ KV Offload          │ 0        │ (KV sur GPU)"
echo "│ CPU MoE Experts     │ $best_ncmoe       │"
echo "│ KV Cache Type K     │ $best_ctk    │"
echo "│ KV Cache Type V     │ $best_ctv    │"
echo "│ Batch Size          │ $best_batch     │"
echo "│ Micro-batch Size    │ $best_ubatch      │"
echo "│ CPU Threads         │ $best_threads       │"
echo "└─────────────────────┴──────────┘"
echo ""
echo "Performance finale : ~$best_threads_tg t/s"
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "📋 COMMANDES PRÊTES À L'EMPLOI"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "Pour lancer le serveur optimisé :"
echo "─────────────────────────────────"
echo "./llama.sh server \\"
echo "  --model $CONTAINER_MODEL \\"
echo "  -ngl 99 \\"
echo "  -nkvo 0 \\"
echo "  -ncmoe $BEST_NCMOE \\"
echo "  -ctk $best_ctk \\"
echo "  -ctv $best_ctv \\"
echo "  -b $best_batch \\"
echo "  -ub $best_ubatch \\"
echo "  -t $best_threads \\"
echo "  --ctx-size 8192"
echo ""
echo "Pour benchmarker :"
echo "──────────────────"
echo "llama-bench \\"
echo "  -m $CONTAINER_MODEL \\"
echo "  -ngl 99 -nkvo 0 -ncmoe $BEST_NCMOE \\"
echo "  -ctk $best_ctk -ctv $best_ctv \\"
echo "  -b $best_batch -ub $best_ubatch \\"
echo "  -t $best_threads \\"
echo "  -p 512 -n 128 -r 5 -o csv"
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
