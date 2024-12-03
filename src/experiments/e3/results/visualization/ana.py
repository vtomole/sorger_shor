import json
import os
import numpy as np

def analyze(isNoLuckyGuesses=False):
    # Read the input JSON file
    if isNoLuckyGuesses:
        input_file_path = f"../times_no_lucky_guesses.json"
        output_file_path = f"../average_mean_no_lucky_guesses.json"
    else:
        input_file_path = f"../times.json"
        output_file_path = f"../average_mean.json"

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


def remove_lucky_guesses():
    # File paths
    input_file_path = "../times.json"
    output_file_path = "../times_no_lucky_guesses.json"

    # Read the input JSON file
    with open(input_file_path, "r") as file:
        data = json.load(file)

    # Function to filter values based on the condition
    def filter_values(values):
        if all(val < 1 for val in values):
            # If all values are less than 1, keep them all
            return values
        else:
            # Otherwise, filter out values less than 1
            return [val for val in values if val >= 1]

    # Process each category and update the results
    result = {}
    for category, numbers in data.items():
        result[category] = {}
        for key, values in numbers.items():
            result[category][key] = filter_values(values)
    
    # Write the processed results to the output JSON file
    with open(output_file_path, "w") as f:
        json.dump(result, f, indent=4)


if __name__ == "__main__":
    analyze()
    remove_lucky_guesses()
    analyze(isNoLuckyGuesses=True)