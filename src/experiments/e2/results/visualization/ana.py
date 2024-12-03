import json
import os
import numpy as np

def analyze(name):
    # Read the input JSON file
    input_file_path = f"../times_{name}.json"
    output_file_path = f"../average_mean_{name}.json"


    with open(input_file_path, "r") as file:
        data = json.load(file)

    # Function to calculate average and median
    def calculate_average_mean(values):
        avg = np.mean(values)  # Calculate average
        med = np.median(values)  # Calculate median
        return [avg, med]

    # Process each category and update the results
    result = {}
    for category, numbers in data.items():
        result[category] = {}
        for key, values in numbers.items():
            result[category][key] = calculate_average_mean(values)
    
    # Write the processed results to the output JSON file
    with open(output_file_path, "w") as f:
        json.dump(result, f, indent=4)


if __name__ == "__main__":
    analyze("normal")
    analyze("sequential")