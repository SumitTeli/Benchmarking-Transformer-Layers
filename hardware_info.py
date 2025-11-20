import platform
import psutil
import torch
import subprocess
import json
import sys

print("===== SYSTEM INFORMATION =====")
print(f"OS: {platform.system()} {platform.release()}")
print(f"Machine: {platform.machine()}\n")

# ---------------------------------------------------
# CPU INFO
# ---------------------------------------------------
print("===== CPU =====")
try:
    if platform.system() == "Windows":
        cpu = platform.processor()
    elif platform.system() == "Darwin":
        cpu = subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"]).strip().decode()
    else:
        cpu = subprocess.check_output("cat /proc/cpuinfo | grep 'model name' | uniq", shell=True).decode().split(":")[1].strip()
except:
    cpu = "Unknown"

print(f"CPU Model: {cpu}")
print(f"Cores: {psutil.cpu_count(logical=False)} physical / {psutil.cpu_count(logical=True)} logical\n")

# ---------------------------------------------------
# RAM INFO
# ---------------------------------------------------
ram_gb = round(psutil.virtual_memory().total / (1024**3), 2)
print("===== MEMORY =====")
print(f"Total RAM: {ram_gb} GB\n")

# ---------------------------------------------------
# GPU INFO (NVIDIA)
# ---------------------------------------------------
print("===== GPU =====")

if torch.cuda.is_available():
    num_gpus = torch.cuda.device_count()
    print(f"CUDA Available: YES ({num_gpus} GPU(s))\n")

    for i in range(num_gpus):
        print(f"GPU #{i}: {torch.cuda.get_device_name(i)}")
        print(f"  CUDA Capability: {torch.cuda.get_device_capability(i)}")
        props = torch.cuda.get_device_properties(i)
        print(f"  VRAM: {round(props.total_memory / (1024**3), 2)} GB")
        print(f"  Multiprocessors: {props.multi_processor_count}")
        print()
else:
    print("CUDA Available: NO\n")

# ---------------------------------------------------
# APPLE SILICON (MPS BACKEND)
# ---------------------------------------------------
if torch.backends.mps.is_available():
    print("===== APPLE SILICON (MPS) =====")
    print("MPS Available: YES")
    print("Running on: Apple Silicon GPU (M1/M2/M3)")
    print()

# ---------------------------------------------------
# FALLBACK — CHECK METAL GPU (macOS)
# ---------------------------------------------------
if platform.system() == "Darwin" and not torch.backends.mps.is_available():
    try:
        out = subprocess.check_output(["system_profiler", "SPDisplaysDataType", "-json"]).decode()
        gpu_info = json.loads(out)
        print("===== macOS GPU =====")
        for gpu in gpu_info["SPDisplaysDataType"]:
            print(f"GPU: {gpu.get('sppci_model', 'Unknown')}")
            print(f"VRAM: {gpu.get('spdisplays_vram', 'Unknown')}")
        print()
    except:
        pass
