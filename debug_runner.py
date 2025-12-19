
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
    
    cfg = HyperOptConfig(
        model_path_host=Path("/home/lucas/Projects/Temp/test_llama_cpp/models/test.gguf"),
        is_moe=False,
        ctx_targets=[512],
        trials=1,
        repeats=1,
        n_tokens=16,
        timeout_sec=10
    )
    
    print("Starting runner...")
    runner.start(cfg, paths, settings)
    
    print(f"Runner started. is_running={runner.is_running}")
    
    # Poll for a few seconds
    for i in range(10):
        time.sleep(0.5)
        print(f"Poll {i}: is_running={runner.is_running}")
        
        while not runner.log_queue.empty():
            print(f"LOG: {runner.log_queue.get()}")
            
        while not runner.result_queue.empty():
            res = runner.result_queue.get()
            print(f"RESULT: {res.get('type')} - {res}")
            
        if not runner.is_running:
            print("Runner stopped.")
            break

if __name__ == "__main__":
    test_runner()
