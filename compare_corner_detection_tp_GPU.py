import os
import subprocess
import re
import matplotlib.pyplot as plt
import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    "-gpu_exec",
    help="Path to the GPU executable",
    type=str
)
parser.add_argument(
    "-cpu_exec",
    help="Path to the CPU thread pool executable",
    type=str
)

args = parser.parse_args()
cpu_exec: str = args.cpu_exec
gpu_exec: str = args.gpu_exec

# Paths to executables
# gpu_exec = "/home/julfy/Documents/ACS/ACS_Visual_Odometry/cmake-build-release/GPU_parallel_feature_extraction_with_matching"
# cpu_exec = "/home/julfy/Documents/ACS/ACS_Visual_Odometry/cmake-build-release/parallel_feature_extraction_with_matching"

# Path to images
image_folder = "data/images"
images = sorted([img for img in os.listdir(image_folder) if img.endswith('.png')])

# Store times
gpu_times = []
cpu_times = []
image_names = []

# Regex patterns
gpu_pattern = re.compile(r"GPU response calculations:\s+(\d+)\s+ms")
cpu_pattern = re.compile(r"Threadpool gradient calculations:\s+(\d+)\s+ms")

print("Running benchmarks on images...")

for image in images:
    img_path = os.path.join(image_folder, image)
    image_names.append(image)

    # Run GPU executable
    gpu_proc = subprocess.run([gpu_exec, img_path, "/home/julfy1/Documents/4th_term/ACS/ACS_Visual_Odometry/kernels/feature_extraction_kernel_functions.bin"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    gpu_match = gpu_pattern.search(gpu_proc.stdout)
    gpu_time = int(gpu_match.group(1)) if gpu_match else -1
    gpu_times.append(gpu_time)
    # print(f"{gpu_proc.stdout = }")

    # Run CPU executable
    cpu_proc = subprocess.run([cpu_exec, img_path, img_path, '16', '150'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    cpu_match = cpu_pattern.search(cpu_proc.stdout)
    cpu_time = int(cpu_match.group(1)) if cpu_match else -1
    cpu_times.append(cpu_time)
    # print(f"{cpu_proc.stdout = }")

    print(f"{image}: GPU={gpu_time}ms, CPU={cpu_time}ms")

# Plotting
plt.figure(figsize=(14, 6))
plt.plot(image_names, gpu_times, label='GPU', marker='o')
plt.plot(image_names, cpu_times, label='Threadpool CPU', marker='s')
plt.xticks(rotation=90)
plt.xlabel("Image")
plt.ylabel("Execution Time (ms)")
plt.title("GPU vs Threadpool Execution Time per Image")
plt.legend()
plt.tight_layout()
plt.grid(True)
plt.show()

