import subprocess
import re
import argparse
import matplotlib.pyplot as plt
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument(
    "-gpu_exec",
    help="Path to the GPU executable",
    type=str,
    required=True
)
parser.add_argument(
    "-image",
    help="Path to the input image",
    type=str,
    required=True
)
parser.add_argument(
    "-kernel_bin",
    help="Path to the kernel binary",
    type=str,
    required=True
)
parser.add_argument(
    "-n",
    help="Number of times to run the benchmark",
    type=int,
    default=1
)

args = parser.parse_args()

# Regular expressions
conv_pattern = re.compile(r"GPU kernel convolution function time:\s+(\d+)\s+ms")
shi_pattern = re.compile(r"GPU shi-tomasi kernel corner detection calculations:\s+(\d+)\s+ms")

conv_times = []
shi_times = []

for i in range(args.n):
    print(f"Run {i+1}:")
    result = subprocess.run(
        [args.gpu_exec, args.image, args.kernel_bin],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )

    conv_match = conv_pattern.search(result.stdout)
    shi_match = shi_pattern.search(result.stdout)

    if conv_match and shi_match:
        conv_time = int(conv_match.group(1))
        shi_time = int(shi_match.group(1))
        conv_times.append(conv_time)
        shi_times.append(shi_time)
        print(f"  Convolution: {conv_time} ms")
        print(f"  Shi-Tomasi: {shi_time} ms")
    else:
        print("Failed to parse output.")
        continue

# Statistics
def print_stats(name, data):
    print(f"\n{name} Statistics:")
    print(f"  Mean: {np.mean(data):.2f} ms")
    print(f"  Median: {np.median(data):.2f} ms")
    print(f"  Std Dev: {np.std(data):.2f} ms")
    print(f"  Average: {np.average(data):.2f} ms")

if conv_times:
    print_stats("Convolution Kernel Time", conv_times)
if shi_times:
    print_stats("Shi-Tomasi Kernel Time", shi_times)

# Plotting
if conv_times and shi_times:
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, args.n + 1), conv_times, label="Convolution", marker='o')
    plt.plot(range(1, args.n + 1), shi_times, label="Shi-Tomasi", marker='s')
    plt.xlabel("Run")
    plt.ylabel("Execution Time (ms)")
    plt.title("GPU Kernel Timing Over Multiple Runs")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
