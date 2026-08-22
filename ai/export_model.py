import os
import subprocess
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PIPELINE_CONFIG_PATH = os.path.join(BASE_DIR, 'ai', 'config', 'pipeline.config')
CHECKPOINT_DIR = os.path.join(BASE_DIR, 'ai', 'models', 'ssd_mobilenet')
OUTPUT_DIR = os.path.join(BASE_DIR, 'ai', 'models', 'ssd_mobilenet', 'saved_model')

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Export command using TFOD exporter_main_v2
cmd = [
    sys.executable,
    "-m", "object_detection.exporter_main_v2",
    "--input_type=image_tensor",
    f"--pipeline_config_path={PIPELINE_CONFIG_PATH}",
    f"--trained_checkpoint_dir={CHECKPOINT_DIR}",
    f"--output_directory={OUTPUT_DIR}"
]

print("==================================================")
print("[INFO] Exporting trained checkpoint to SavedModel...")
print(f"[INFO] Pipeline Config: {PIPELINE_CONFIG_PATH}")
print(f"[INFO] Output Directory: {OUTPUT_DIR}")
print("==================================================")

try:
    subprocess.run(cmd, check=True)
    print("\n[SUCCESS] Model exported successfully to 'ai/models/ssd_mobilenet/saved_model/'!")
except subprocess.CalledProcessError as e:
    print(f"\n[ERROR] Export failed with return code {e.returncode}")