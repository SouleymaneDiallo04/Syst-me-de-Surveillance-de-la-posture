import os
import subprocess
import sys
import time
from pathlib import Path

# Configuration
CONFIG = {
    "data": "fall.yaml",
    "weights": "models/yolov5s.pt",
    "cfg": "models/yolov5s.yaml",
    "epochs": 100,
    "batch_size": 16,
    "img_size": 640,
    "device": "0",
    "project": "custom_yolo",
    "name": "exp1",
    "workers": 4,
    "timeout_hours": 24  # Timeout après 24h
}

def check_environment():
    """Vérifie les prérequis"""
    required_files = {
        "Data config": CONFIG['data'],
        "Model config": CONFIG['cfg'],
        "Weights": CONFIG['weights']
    }
    
    for name, path in required_files.items():
        if not Path(path).exists():
            raise FileNotFoundError(f"{name} not found: {path}")

    # Détection GPU
    try:
        subprocess.run(["nvidia-smi"], check=True, 
                      stdout=subprocess.DEVNULL, 
                      stderr=subprocess.DEVNULL)
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("⚠️ Using CPU (GPU not detected)")
        CONFIG['device'] = "cpu"
        CONFIG['batch_size'] = 8
        CONFIG['workers'] = 0

def run_training():
    """Exécute l'entraînement avec timeout"""
    cmd = [
        sys.executable, "train.py",
        f"--img={CONFIG['img_size']}",
        f"--batch={CONFIG['batch_size']}",
        f"--epochs={CONFIG['epochs']}",
        f"--data={CONFIG['data']}",
        f"--cfg={CONFIG['cfg']}",
        f"--weights={CONFIG['weights']}",
        f"--project={CONFIG['project']}",
        f"--name={CONFIG['name']}",
        f"--device={CONFIG['device']}",
        f"--workers={CONFIG['workers']}"
    ]

    print("⚙️ Configuration:")
    for k, v in CONFIG.items():
        print(f"  {k:>12}: {v}")

    print("\n🚀 Starting training...")
    print(" ".join(cmd) + "\n")

    start_time = time.time()
    timeout = CONFIG['timeout_hours'] * 3600  # Convert to seconds

    try:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True,
            encoding='utf-8',
            bufsize=1
        )

        # Monitoring en temps réel
        while True:
            output = process.stdout.readline()
            
            # Vérification timeout
            if time.time() - start_time > timeout:
                process.terminate()
                raise TimeoutError(f"Training exceeded {CONFIG['timeout_hours']} hour limit")
            
            # Vérification fin du processus
            if process.poll() is not None:
                if output == '':
                    break
            
            if output:
                print(output.strip(), flush=True)
                # Détection des signaux de progression
                if "epochs completed" in output.lower():
                    print(f"⏳ Progress: {output.strip()}")

        # Vérification du code de sortie
        if process.returncode != 0:
            error_msg = process.stderr.read()
            raise subprocess.CalledProcessError(
                process.returncode, cmd, error_msg
            )

        print("\n✅ Training completed successfully!")

    except TimeoutError as e:
        print(f"\n❌ {str(e)}")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Training failed: {str(e)}")
        if hasattr(e, 'stderr') and e.stderr:
            print("\nError details:")
            print(e.stderr)
        sys.exit(1)

if __name__ == "__main__":
    try:
        check_environment()
        run_training()
    except Exception as e:
        print(f"\n❌ Setup failed: {str(e)}")
        sys.exit(1)