import json
import numpy as np
import os

def run(run_shor, TimeTracker):
    print("Running Experiment 1")
    tracker, results = configure(TimeTracker)
    
    # Read the input JSON file
    data = get_data()

    # Run the experiment
    # run_set(run_shor, "composite_numbers", data['composite_numbers'], tracker, results)
    run_set(run_shor, "odd_composite_numbers", data['odd_composite_numbers'], tracker, results)
    
    # Save results to JSON
    with open("./e1/results/times.json", "w") as f:
        json.dump(results, f, indent=4)

    # Analyze the results
    analyze()
    print("Experiment 1 completed.")



def configure(TimeTracker):
    tracker = TimeTracker()
    # Initialize the result dictionary
    results = {
        # "primes": {},
        # "power_primes": {},
        # "non_primes_non_power_primes": {},
        # "composite_numbers": {},
        "odd_composite_numbers": {}
    }
    return tracker, results


def get_data(path='./e1/data.json'):
    try:
        with open(path, 'r') as config_file:
            config = json.load(config_file)
            return config
    except FileNotFoundError:
        print("Config file not found.")
    except json.JSONDecodeError:
        print("Error parsing the JSON file.")


def run_set(run_shor, category_name, data_set, tracker, results):
    for i in data_set:
        times = []
        # Run the experiment 100 times
        for j in range(5):
            print(f"Running Shor's algorithm for {i}, for the {j+1} time.")
            tracker.start()
            # Run shor in a simulation
            run_shor(i, isSimulator=True, isCircuitOptimized=True, isPrinting=False)
            tracker.stop()
            write_result(i, category_name, tracker.total_time)
            # times.append(tracker.total_time)
            tracker.reset()
        results[category_name][str(i)] = times


def write_result(number, category_name, time):
    file_path = f"./e1/results/times.json"
    
    # Check if the file exists and load its content, otherwise start with an empty dict
    if os.path.exists(file_path):
        with open(file_path, "r") as f:
            results = json.load(f)
    else:
        results = {
            # "primes": {},
            # "power_primes": {},
            # "non_primes_non_power_primes": {},
            # "composite_numbers": {},
            "odd_composite_numbers": {}
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


def analyze():
    # Read the input JSON file
    with open("./e1/results/times.json", "r") as file:
        data = json.load(file)

    # Function to calculate average and mean
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

    # Write the results to the output JSON file
    with open("./e1/results/average_mean.json", "w") as outfile:
        json.dump(result, outfile, indent=4)
