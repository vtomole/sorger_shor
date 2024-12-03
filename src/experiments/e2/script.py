import json
import os
import numpy as np
from e2.RMTL_Shor_Normal_QFT import run as run_normal
from e2.RMTL_Shor_Sequential_QFT import run as run_sequential

def run(TimeTracker):
    print("Running Experiment 2")
    tracker, results = configure(TimeTracker)
    
    # Read the input JSON file
    data = get_data()

    # Run the experiment: normal
    name = "normal"
    # run_set(run_normal, "composite_numbers", data['composite_numbers'], tracker, results, name)
    
    # Analyze the results
    analyze(name)


    # Run the experiment: normal
    name = "sequential"
    # run_set(run_sequential, "composite_numbers", data['composite_numbers'], tracker, results, name)

    # Analyze the results
    analyze(name)

    print("Experiment 2 completed.")


def configure(TimeTracker):
    tracker = TimeTracker()
    # Initialize the result dictionary
    results = {
        # "primes": {},
        # "power_primes": {},
        # "non_primes_non_power_primes": {},
        "composite_numbers": {}
    }
    return tracker, results


def get_data(path='./e2/data.json'):
    try:
        with open(path, 'r') as config_file:
            config = json.load(config_file)
            return config
    except FileNotFoundError:
        print("Config file not found.")
    except json.JSONDecodeError:
        print("Error parsing the JSON file.")


def run_set(run_shor, category_name, data_set, tracker, results, name):
    # How often the test is run
    for j in range(50):
        # The number it is run for   
        for i in data_set:
            print(f"Running Shor's algorithm for {i}, for the {j+1} time.")
            tracker.start()
            # Run shor in a simulation
            run_shor(i)
            tracker.stop()
            write_result(i, category_name, tracker.total_time, name)
            # times.append(tracker.total_time)
            tracker.reset()


def write_result(number, category_name, time, name):
    file_path = f"./e2/results/times_{name}.json"
    
    # Check if the file exists and load its content, otherwise start with an empty dict
    if os.path.exists(file_path):
        with open(file_path, "r") as f:
            results = json.load(f)
    else:
        results = {
            # "primes": {},
            # "power_primes": {},
            # "non_primes_non_power_primes": {},
            "composite_numbers": {}
        }

    # Ensure the category and number exist in the results structure
    if category_name not in results:
        results[category_name] = {}
    if str(number) not in results[category_name]:
        results[category_name][str(number)] = []
    
    # Append the time to the correct list
    results[category_name][str(number)].append(time)

    # Write the updated results back to the file
    with open(file_path, "w") as f:
        json.dump(results, f, indent=4)


def analyze(name):
    # Read the input JSON file
    input_file_path = f"./e2/results/times_{name}.json"
    output_file_path = f"./e2/results/average_mean_{name}.json"


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
