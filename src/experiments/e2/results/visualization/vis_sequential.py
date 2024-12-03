import json
import matplotlib.pyplot as plt
import numpy as np
from numpy import nan
import os

# Load JSON data
def load_json(file_path):
    with open(file_path, 'r') as file:
        return json.load(file)

# Load all data files
avg_mean_sequential = load_json('../average_mean_sequential.json')
times_sequential = load_json('../times_sequential.json')

# Extract the composite numbers from all files
odd_composite_numbers_sequential = set(map(int, avg_mean_sequential['odd_composite_numbers'].keys()))
odd_composite_numbers_times_sequential = set(map(int, times_sequential['odd_composite_numbers'].keys()))

# Find the union of all composite numbers across all files
all_odd_composite_numbers = sorted(odd_composite_numbers_sequential | odd_composite_numbers_times_sequential)

# Prepare data for plotting (handle missing values as NaN)
avg_sequential = [avg_mean_sequential['odd_composite_numbers'].get(str(num), [nan, nan])[0] for num in all_odd_composite_numbers]
mean_sequential = [avg_mean_sequential['odd_composite_numbers'].get(str(num), [nan, nan])[1] for num in all_odd_composite_numbers]

# Prepare runtime data for the composite numbers (handle missing values as NaN)
times_sequential_values = [times_sequential['odd_composite_numbers'].get(str(num), []) for num in all_odd_composite_numbers]

# Create the plot
plt.figure(figsize=(10, 6))

# Plot individual runtimes as scatter points
for i, num in enumerate(all_odd_composite_numbers):
    # Scatter for sequential mode
    if times_sequential_values[i]:
        plt.scatter([num] * len(times_sequential_values[i]), times_sequential_values[i], color='red', alpha=0.5, label=f'Sequential QFT {num}' if i == 0 else "")

# Plot average and mean trend lines with interpolation (skip missing values)
plt.plot(all_odd_composite_numbers, avg_sequential, color='red', label='Avg Sequential QFT', linestyle='-', marker='o', markersize=5)
plt.plot(all_odd_composite_numbers, mean_sequential, color='red', linestyle='--', label='Mean Sequential QFT', marker='x', markersize=5)

# Set dynamic y-axis limits
all_times = [time for sublist in times_sequential_values for time in sublist if time is not None]
y_min = min(all_times) * 0.95 if all_times else 0  # Add 5% padding for y-axis lower limit
y_max = max(all_times) * 1.05 if all_times else 1  # Add 5% padding for y-axis upper limit

# Set the limits for the plot
plt.ylim(y_min, y_max)

# Labels and title
plt.xlabel('Composite Numbers')
plt.ylabel('Time (s)')

# Add legend and display
plt.legend()

# Set the x-ticks to display every 5th number
tick_step = 1
x_ticks = all_odd_composite_numbers[::tick_step]  # Get every 5th number
plt.xticks(x_ticks)

plt.tight_layout()

# Get the script name
script_name = os.path.splitext(os.path.basename(__file__))[0]
output_file = f"{script_name}.svg"

# Save the plot
plt.savefig(output_file, format='svg')
