#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

SERVICE="llama-cpp"
LLAMA_BIN_DIR="${LLAMA_BIN_DIR:-/app}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m' # No Color

usage() {
  cat <<'EOF'
Usage: ./llama.sh <command> [options]

QUICK START COMMANDS:
  dashboard                        Launch Streamlit dashboard (web UI)
  optimize <model> [--preset P]    Run HyperOptimus optimization (CLI)
  
LLAMA.CPP COMMANDS:
  server [args...]                 Run llama-server
  bench [args...]                  Run llama-bench
  shell [args...]                  Interactive shell in container

LEGACY PRESETS:
  optimus-fast|mid|high <model>    Run llama-optimus (old optimizer)
  moe-scan <model>                 Sweep -ncmoe values for MoE models

EXAMPLES:
  # Launch web dashboard
  ./llama.sh dashboard

  # Optimize a model (fastest way)
  ./llama.sh optimize my-model.gguf --preset fast

  # Run server with optimized config
  ./llama.sh server --model models/my-model.gguf -ngl 99 -c 16384

OPTIONS for 'optimize':
  --preset fast|mid|high    Optimization intensity (default: fast)
  --trials N                Number of trials
  --ctx-max N               Maximum context size
  --help                    Show all options

For full documentation: https://github.com/BerthalonLucas/llama-cpp-optimus-bench
EOF
}

# ─────────────────────────────────────────────────────────────────────────────
# Dashboard command
# ─────────────────────────────────────────────────────────────────────────────
run_dashboard() {
  local port="${DASHBOARD_PORT:-8510}"
  local host="${DASHBOARD_HOST:-0.0.0.0}"
  
  # Parse args
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --port|-p)
        port="${2:-8510}"
        shift 2
        ;;
      --host|-h)
        host="${2:-0.0.0.0}"
        shift 2
        ;;
      *)
        shift
        ;;
    esac
  done
  
  echo -e "${CYAN}${BOLD}🚀 Launching Dashboard${NC}"
  echo -e "   URL: ${GREEN}http://localhost:${port}${NC}"
  echo -e "   Host: ${host}"
  echo ""
  echo -e "   Press ${YELLOW}Ctrl+C${NC} to stop"
  echo ""
  
  # Activate venv if exists
  if [[ -f "$SCRIPT_DIR/.venv/bin/activate" ]]; then
    source "$SCRIPT_DIR/.venv/bin/activate"
  fi
  
  exec python3 -m streamlit run \
    "$SCRIPT_DIR/streamlit_dashboard/streamlit_app.py" \
    --server.port "$port" \
    --server.address "$host" \
    --server.headless true
}

# ─────────────────────────────────────────────────────────────────────────────
# Optimize command (HyperOptimus CLI)
# ─────────────────────────────────────────────────────────────────────────────
run_optimize() {
  # Activate venv if exists
  if [[ -f "$SCRIPT_DIR/.venv/bin/activate" ]]; then
    source "$SCRIPT_DIR/.venv/bin/activate"
  fi
  
  exec python3 "$SCRIPT_DIR/cli.py" optimize "$@"
}

validate_port() {
  local p="${1:-}"
  if [[ -z "$p" || ! "$p" =~ ^[0-9]+$ ]]; then
    echo "Invalid port: '${p}'" >&2
    exit 2
  fi
  if (( p < 1 || p > 65535 )); then
    echo "Invalid port (out of range 1-65535): '${p}'" >&2
    exit 2
  fi
}

extract_llama_server_port() {
  # Extract `--port N` or `--port=N` from llama-server args (best-effort).
  local args=("$@")
  local port=""
  for ((i=0; i<${#args[@]}; i++)); do
    case "${args[$i]}" in
      --port=*)
        port="${args[$i]#--port=}"
        ;;
      --port)
        port="${args[$((i+1))]:-}"
        ;;
    esac
    if [[ -n "$port" ]]; then
      echo "$port"
      return 0
    fi
  done
  return 1
}

normalize_llama_server_args() {
  # llama-server expects some boolean flags as "flag-present", while our dashboard
  # (and llama-bench) uses `-fa 0|1` and `-nkvo 0|1`. Accept both for convenience.
  local in=("$@")
  local out=()
  for ((i=0; i<${#in[@]}; i++)); do
    case "${in[$i]}" in
      -fa|--flash-attn)
        # If next token is 0/1, consume it and keep/remove the flag accordingly.
        if (( i+1 < ${#in[@]} )); then
          case "${in[$((i+1))]}" in
            0)
              i=$((i+1))
              continue
              ;;
            1)
              out+=("--flash-attn")
              i=$((i+1))
              continue
              ;;
          esac
        fi
        out+=("--flash-attn")
        ;;
      --flash-attn=0)
        ;;
      --flash-attn=1)
        out+=("--flash-attn")
        ;;
      -nkvo|--no-kv-offload)
        # If next token is 0/1, consume it and keep/remove the flag accordingly.
        if (( i+1 < ${#in[@]} )); then
          case "${in[$((i+1))]}" in
            0)
              i=$((i+1))
              continue
              ;;
            1)
              out+=("--no-kv-offload")
              i=$((i+1))
              continue
              ;;
          esac
        fi
        out+=("--no-kv-offload")
        ;;
      --no-kv-offload=0)
        ;;
      --no-kv-offload=1)
        out+=("--no-kv-offload")
        ;;
      *)
        out+=("${in[$i]}")
        ;;
    esac
  done

  printf '%s\n' "${out[@]}"
}

ensure_image() {
  mkdir -p models
  echo "Building/updating image for ${SERVICE} (pulling latest base)..."
  docker compose build --pull "$SERVICE"
}

download_hf_model() {
  local repo_id="$1"
  local target_file="${2:-}"
  local env_passthrough=()
  for var in HF_TOKEN HUGGINGFACEHUB_API_TOKEN HUGGINGFACEHUB_TOKEN; do
    if [[ -n "${!var:-}" ]]; then
      env_passthrough+=(-e "$var=${!var}")
    fi
  done

  echo "Downloading Hugging Face repo '${repo_id}' into ./models (cache: ./models/hf-cache)..." >&2
  local path
  path="$(docker compose run --rm \
    -e HF_HOME=/models \
    -e HF_HUB_CACHE=/models/hf-cache \
    -e HUGGINGFACE_HUB_CACHE=/models/hf-cache \
    "${env_passthrough[@]}" \
    "$SERVICE" \
    python3 - "$repo_id" "$target_file" <<'PY'
import sys
from huggingface_hub import HfApi, hf_hub_download

repo_arg = sys.argv[1]
target_file = sys.argv[2] if len(sys.argv) > 2 and sys.argv[2] else None

# Support: hf.co/<org>/<repo>:<quant> (quant optional)
quant = None
repo = repo_arg.strip()
if repo.startswith("hf.co/"):
    repo = repo[len("hf.co/"): ]
for prefix in ("https://huggingface.co/", "http://huggingface.co/", "https://hf.co/", "http://hf.co/"):
    if repo.startswith(prefix):
        repo = repo[len(prefix): ]
if ':' in repo:
    repo, quant = repo.rsplit(':', 1)

api = HfApi()

if target_file is None:
    files = api.list_repo_files(repo_id=repo, repo_type="model")
    ggufs = [f for f in files if f.lower().endswith(".gguf")]
    if not ggufs:
        print(f"No .gguf files found in repo '{repo}'. Please specify --hf-file.", file=sys.stderr)
        sys.exit(2)

    chosen = None
    if quant:
        q = quant.lower()
        # Prefer exact suffix match first
        exact = [f for f in ggufs if f.lower().endswith(f".{q}.gguf") or f.lower().endswith(f"-{q}.gguf")]
        if exact:
            exact.sort(key=lambda x: (len(x), x.lower()))
            chosen = exact[0]
        else:
            contains = [f for f in ggufs if q in f.lower()]
            contains.sort(key=lambda x: (len(x), x.lower()))
            if contains:
                chosen = contains[0]

    target_file = chosen or sorted(ggufs, key=str.lower)[0]

kwargs = dict(
    repo_id=repo,
    filename=target_file,
    repo_type="model",
    local_dir="/models",
)
# huggingface_hub < 1.0 supports local_dir_use_symlinks. In >= 1.0 it's removed.
try:
    path = hf_hub_download(**kwargs, local_dir_use_symlinks=False)
except TypeError:
    path = hf_hub_download(**kwargs)
print(path)
PY
  )"
  # Trim to last line in case warnings are emitted
  echo "$path" | tail -n 1 | tr -d '\r'
}

resolve_model_path() {
  local model="$1"
  if [[ -z "${model:-}" ]]; then
    echo "Model path is required." >&2
    exit 1
  fi

  if [[ "$model" != /* ]]; then
    model="$SCRIPT_DIR/models/$model"
  fi
  if [[ ! -f "$model" ]]; then
    echo "Model not found: $model" >&2
    exit 1
  fi

  local abs_model abs_models
  abs_model="$(realpath "$model")"
  abs_models="$(realpath "$SCRIPT_DIR/models")"
  if [[ "$abs_model" == "$abs_models"* ]]; then
    echo "/models/${abs_model#$abs_models/}"
  else
    echo "$abs_model"
  fi
}

run_optimus_preset() {
  local preset="$1"; shift

  local preset_args=()
  local preset_label=""
  case "$preset" in
    optimus-fast)
      preset_label="fast (smoke)"
      preset_args=(--trials 12 --repeat 1 --no-warmup --n-tokens 192)
      ;;
    optimus-mid)
      preset_label="mid (balanced)"
      preset_args=(--trials 40 --repeat 3 --n-tokens 512)
      ;;
    optimus-high)
      preset_label="high (thorough)"
      preset_args=(--trials 100 --repeat 5 --n-tokens 1024)
      ;;
    *)
      usage
      exit 1
      ;;
  esac

  local repo_id="" hf_file="" model="" extra=()
  while [[ $# -gt 0 ]]; do
    case "$1" in
      -hf|--hf)
        repo_id="${2:-}"
        shift 2
        ;;
      --hf-file)
        hf_file="${2:-}"
        shift 2
        ;;
      -m|--model)
        model="${2:-}"
        shift 2
        ;;
      *)
        extra+=("$1")
        shift
        ;;
    esac
  done

  local container_model=""
  if [[ -n "$repo_id" ]]; then
    container_model="$(download_hf_model "$repo_id" "$hf_file")"
  else
    if [[ -z "$model" && ${#extra[@]} -ge 1 ]]; then
      model="${extra[0]}"
      extra=("${extra[@]:1}")
    fi
    if [[ -z "$model" ]]; then
      echo "Model path is required (pass a local file or -hf <repo>)." >&2
      exit 1
    fi
    container_model="$(resolve_model_path "$model")"
  fi

  echo "Running llama-optimus preset '${preset_label}' on ${container_model}"
  exec docker compose run --rm "$SERVICE" \
    llama-optimus \
      --llama-bin "$LLAMA_BIN_DIR" \
      --model "$container_model" \
      "${preset_args[@]}" \
      "${extra[@]}"
}

# Quick MoE helper: sweeps -ncmoe with llama-bench to keep VRAM in check.
run_moe_scan() {
  local repo_id="" hf_file="" model=""
  local ncmoe_values_str="0 5 10 15 20 25 30"
  local ngl="${MOE_NGL:-99}" n_prompt="512" n_gen="128" repeat="2"
  local extra=()

  while [[ $# -gt 0 ]]; do
    case "$1" in
      -hf|--hf)
        repo_id="${2:-}"
        shift 2
        ;;
      --hf-file)
        hf_file="${2:-}"
        shift 2
        ;;
      -m|--model)
        model="${2:-}"
        shift 2
        ;;
      --ncmoe-values)
        ncmoe_values_str="${2:-}"
        shift 2
        ;;
      --ngl)
        ngl="${2:-}"
        shift 2
        ;;
      -p|--n-prompt)
        n_prompt="${2:-}"
        shift 2
        ;;
      -n|--n-gen)
        n_gen="${2:-}"
        shift 2
        ;;
      -r|--repeat)
        repeat="${2:-}"
        shift 2
        ;;
      --)
        shift
        extra+=("$@")
        break
        ;;
      *)
        extra+=("$1")
        shift
        ;;
    esac
  done

  if [[ -z "$repo_id" && -z "$model" && ${#extra[@]} -ge 1 ]]; then
    model="${extra[0]}"
    extra=("${extra[@]:1}")
  fi

  if [[ -z "$repo_id" && -z "$model" ]]; then
    echo "Model path is required for moe-scan/moe-sweep" >&2
    exit 1
  fi

  local container_model=""
  if [[ -n "$repo_id" ]]; then
    container_model="$(download_hf_model "$repo_id" "$hf_file")"
  else
    container_model="$(resolve_model_path "$model")"
  fi

  read -ra ncmoe_values <<<"$ncmoe_values_str"
  if [[ ${#ncmoe_values[@]} -eq 0 ]]; then
    echo "At least one value is required for --ncmoe-values." >&2
    exit 1
  fi

  echo "Sweeping -ncmoe values (${ncmoe_values[*]})"
  echo "  model: ${container_model}"
  echo "  prompt tokens: ${n_prompt}, gen tokens: ${n_gen}, repeat: ${repeat}, ngl: ${ngl}"

  docker compose run --rm "$SERVICE" \
    python3 - "$container_model" "$ngl" "$n_prompt" "$n_gen" "$repeat" "${ncmoe_values[@]}" -- "${extra[@]}" <<'PY'
import csv
import subprocess
import sys

model, ngl, n_prompt, n_gen, repeat, *rest = sys.argv[1:]
if "--" in rest:
    split_index = rest.index("--")
    values = rest[:split_index]
    bench_args = rest[split_index + 1 :]
else:
    values = rest
    bench_args = []

if not values:
    print("No -ncmoe values supplied", file=sys.stderr)
    sys.exit(2)

def run_value(val: str) -> float:
    cmd = [
        "llama-bench",
        "--model",
        model,
        "-ngl",
        ngl,
        "-ncmoe",
        val,
        "-p",
        n_prompt,
        "-n",
        n_gen,
        "-r",
        repeat,
        "-o",
        "csv",
    ]
    cmd.extend(bench_args)
    print(f"Testing -ncmoe {val}...", flush=True)
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        print(proc.stdout)
        print(proc.stderr, file=sys.stderr)
        raise SystemExit(proc.returncode)

    rows = list(csv.DictReader(proc.stdout.splitlines()))
    tg_vals = []
    for row in rows:
        test = (row.get("test") or "").strip()
        if test.startswith("tg"):
            val_str = row.get("avg_ts") or row.get("t/s")
            try:
                tg_vals.append(float(val_str))
            except (TypeError, ValueError):
                continue
    if not tg_vals:
        print("No tg* rows parsed for -ncmoe", val, file=sys.stderr)
        print(proc.stdout, file=sys.stderr)
        raise SystemExit(3)

    best_tg = max(tg_vals)
    print(f"  → tg: {best_tg:.2f} t/s", flush=True)
    return best_tg

best_val, best_tg = None, None
for ncmoe in values:
    tg = run_value(ncmoe)
    if best_tg is None or tg > best_tg:
        best_val, best_tg = ncmoe, tg

print()
print(f"Best -ncmoe: {best_val} (tg: {best_tg:.2f} t/s)")
print("Add '-ncmoe", best_val, "' to llama-server/llama-bench, then run llama-optimus to tune threads/batch/flash.")
PY
}

cmd="${1:-}"
if [[ -z "$cmd" || "$cmd" == "-h" || "$cmd" == "--help" ]]; then
  usage
  exit 0
fi
shift || true

	case "$cmd" in
  dashboard)
    run_dashboard "$@"
    ;;
  optimize)
    if [[ $# -lt 1 ]]; then
      echo -e "${RED}Model path is required for optimize${NC}" >&2
      echo "Usage: ./llama.sh optimize <model> [--preset fast|mid|high]"
      exit 1
    fi
    run_optimize "$@"
    ;;
  server)
	    ensure_image
	    # Publish port dynamically:
	    # - default: 8080:8080
	    # - if llama-server args include `--port N`, publish N:N
	    # - if LLAMA_SERVER_PORT is set and args do not include --port, the script
	    #   injects `--port $LLAMA_SERVER_PORT` so the server actually listens there
	    # - allow overriding host port without changing server port via env var LLAMA_SERVER_HOST_PORT
	    args=("$@")
	    server_port="8080"
	    port_in_args="0"
	    if port_from_args="$(extract_llama_server_port "${args[@]}")"; then
	      server_port="$port_from_args"
	      port_in_args="1"
	    elif [[ -n "${LLAMA_SERVER_PORT:-}" ]]; then
	      server_port="${LLAMA_SERVER_PORT}"
	    fi
	    host_port="${LLAMA_SERVER_HOST_PORT:-$server_port}"
	    validate_port "$server_port"
	    validate_port "$host_port"
	    if [[ "$port_in_args" != "1" && -n "${LLAMA_SERVER_PORT:-}" ]]; then
	      args+=("--port" "$server_port")
	    fi
	    # Make copy/paste from bench results work: normalize -fa 0|1 to server-style flag.
	    mapfile -t args < <(normalize_llama_server_args "${args[@]}")

	    # NOTE: the service uses an entrypoint `zsh -c`, so we must override the
	    # entrypoint to preserve args (otherwise only the first word is executed).
	    exec docker compose run --rm -p "${host_port}:${server_port}" \
	      --entrypoint "${LLAMA_BIN_DIR}/llama-server" \
	      "$SERVICE" \
	      "${args[@]}"
	    ;;
	  bench)
	    ensure_image
	    exec docker compose run --rm "$SERVICE" llama-bench "$@"
	    ;;
  shell)
    ensure_image
    exec docker compose run --rm "$SERVICE" bash -l "$@"
    ;;
  optimus-fast|optimus-mid|optimus-high)
    if [[ $# -lt 1 ]]; then
      echo "Model path is required for $cmd" >&2
      usage
      exit 1
    fi
    ensure_image
    run_optimus_preset "$cmd" "$@"
    ;;
  moe-scan|moe-sweep)
    ensure_image
    run_moe_scan "$@"
    ;;
  *)
    usage
    exit 1
    ;;
esac
