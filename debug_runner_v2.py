
import sys
import time
import threading
from pathlib import Path

# Add project root to path
sys.path.insert(0, "/home/lucas/Projects/Temp/test_llama_cpp")

from streamlit_dashboard.core.hyperoptimus import HyperOptimusRunner, HyperOptConfig
from streamlit_dashboard.core.paths import RepoPaths
from streamlit_dashboard.core.settings import AppSettings

def test_runner():
    print("Initializing runner...")
    runner = HyperOptimusRunner()
    
    paths = RepoPaths.detect()
    settings = AppSettings()
    
    # Use a real model
    model_path = paths.models_dir / "Qwen3-0.6B-UD-Q4_K_XL.gguf"
    if not model_path.exists():
        print(f"Model not found: {model_path}")
        # Try to find any gguf
        ggufs = list(paths.models_dir.glob("*.gguf"))
        if ggufs:
            model_path = ggufs[0]
            print(f"Using alternative model: {model_path}")
        else:
            print("No GGUF models found!")
            return

    print(f"Testing with model: {model_path}")

    cfg = HyperOptConfig(
        model_path_host=model_path,
        is_moe=False,
        ctx_targets=[512],
        trials=1,
        repeats=1,
        n_tokens=16,
        timeout_sec=60,
        batch_range=(128, 128, 1), # Fixed small batch
        ubatch_range=(64, 64, 1),
        threads_range=(4, 4, 1)
    )
    
    print("Starting runner...")
    runner.start(cfg, paths, settings)
    
    # Poll
    start_time = time.time()
    while time.time() - start_time < 60:
        time.sleep(1)
        
        while not runner.log_queue.empty():
            print(f"LOG: {runner.log_queue.get()}")
            
        while not runner.result_queue.empty():
            res = runner.result_queue.get()
            print(f"RESULT: {res.get('type')} - {res}")
            if res.get('type') == 'complete' or res.get('type') == 'error':
                print("Finished.")
                return

    print("Timeout waiting for runner.")

if __name__ == "__main__":
    test_runner()
