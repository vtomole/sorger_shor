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
avg_mean_normal = load_json('../average_mean_no_lucky_guesses.json')
avg_mean_optimized = load_json('../../../e1/results/average_mean_no_lucky_guesses.json')
times_normal = load_json('../times_no_lucky_guesses.json')
times_optimized = load_json('../../../e1/results/times_no_lucky_guesses.json')

# Extract the composite numbers from all files
odd_composite_numbers_normal = set(map(int, avg_mean_normal['odd_composite_numbers'].keys()))
odd_composite_numbers_optimized = set(map(int, avg_mean_optimized['odd_composite_numbers'].keys()))
odd_composite_numbers_times_normal = set(map(int, times_normal['odd_composite_numbers'].keys()))
odd_composite_numbers_times_optimized = set(map(int, times_optimized['odd_composite_numbers'].keys()))

# Find the union of all composite numbers across all files
all_odd_composite_numbers = sorted(odd_composite_numbers_normal | odd_composite_numbers_optimized | odd_composite_numbers_times_normal | odd_composite_numbers_times_optimized)

# Prepare data for plotting (handle missing values as NaN)
avg_normal = [avg_mean_normal['odd_composite_numbers'].get(str(num), [nan, nan])[0] for num in all_odd_composite_numbers]
mean_normal = [avg_mean_normal['odd_composite_numbers'].get(str(num), [nan, nan])[1] for num in all_odd_composite_numbers]
avg_optimized = [avg_mean_optimized['odd_composite_numbers'].get(str(num), [nan, nan])[0] for num in all_odd_composite_numbers]
mean_optimized = [avg_mean_optimized['odd_composite_numbers'].get(str(num), [nan, nan])[1] for num in all_odd_composite_numbers]

# Prepare runtime data for the composite numbers (handle missing values as NaN)
times_normal_values = [times_normal['odd_composite_numbers'].get(str(num), []) for num in all_odd_composite_numbers]
times_optimized_values = [times_optimized['odd_composite_numbers'].get(str(num), []) for num in all_odd_composite_numbers]

# Create the plot
plt.figure(figsize=(10, 6))

# Plot individual runtimes as scatter points
for i, num in enumerate(all_odd_composite_numbers):
     # Scatter for Optimized mode
    if times_optimized_values[i]:
        plt.scatter([num] * len(times_optimized_values[i]), times_optimized_values[i], color='green', alpha=0.5, label=f'Our Implementation Optimized No Lucky Guesses {num}' if i == 0 else "")
    # Scatter for Not Optimized mode
    if times_normal_values[i]:
        plt.scatter([num] * len(times_normal_values[i]), times_normal_values[i], color='purple', alpha=0.5, label=f'Our Implementation Not Optimized No Lucky Guesses {num}' if i == 0 else "")
   
# Plot average and mean trend lines with interpolation (skip missing values)
plt.plot(all_odd_composite_numbers, avg_optimized, color='green', label='Avg Our Implementation Optimized No Lucky Guesses', linestyle='-', marker='o', markersize=5)
plt.plot(all_odd_composite_numbers, avg_normal, color='purple', label='Avg Our Implementation Not Optimized No Lucky Guesses', linestyle='-', marker='o', markersize=5)
plt.plot(all_odd_composite_numbers, mean_optimized, color='green', linestyle='--', label='Mean Our Implementation Optimized No Lucky Guesses', marker='x', markersize=5)
plt.plot(all_odd_composite_numbers, mean_normal, color='purple', linestyle='--', label='Mean Our Implementation Not Optimized No Lucky Guesses', marker='x', markersize=5)

# Set dynamic y-axis limits
all_times = [time for sublist in times_normal_values + times_optimized_values for time in sublist if time is not None]
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
tick_step = 5
x_ticks = all_odd_composite_numbers[::tick_step]  # Get every 5th number
plt.xticks(x_ticks)

plt.tight_layout()

# Get the script name
script_name = os.path.splitext(os.path.basename(__file__))[0]
output_file = f"{script_name}.svg"

# Save the plot
plt.savefig(output_file, format='svg')
