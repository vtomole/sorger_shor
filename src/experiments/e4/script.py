import json
import numpy as np
import os

def run(run_shor):
    print("Running Experiment 4")
    
    # Read the input JSON file
    data = get_data()

    # Run the experiment
    run_set(run_shor, "odd_composite_numbers", data['odd_composite_numbers'])

    print("Experiment 4 completed.")


def get_data(path='./e4/data.json'):
    try:
        with open(path, 'r') as config_file:
            config = json.load(config_file)
            return config
    except FileNotFoundError:
        print("Config file not found.")
    except json.JSONDecodeError:
        print("Error parsing the JSON file.")


def run_set(run_shor, category_name, data_set):
    for i in data_set:
        # Run the algorithm 10 times
        for j in range(10):
            print(f"Running Shor's algorithm for {i}, for the {j+1} time.")
            # Run shor in a simulation
            isSimulator = False
            r, p, q = run_shor(i, isSimulator=isSimulator, isCircuitOptimized=True, isPrinting=False)
            # Analyze results
            if r:
                if p * q == i:
                    shor_result = "correct"
                else:
                    shor_result = "incorrect"
            else:
                shor_result = "dnf"
            print(f"Number {i} run #{j} result: {shor_result}")
            write_result(i, category_name, shor_result, isSimulator)


def write_result(number, category_name, shor_result, isSimulator):
    file_path = f"./e4/results/results.json"

    if isSimulator:
        experiment = "Simulator"
    elif isSimulator == False:
        experiment = "QHW"
    else:
        raise ValueError("isSimulator must be a boolean value")
    
    # Check if the file exists and load its content, otherwise start with an empty dict
    if os.path.exists(file_path):
        with open(file_path, "r") as f:
            results = json.load(f)
    else:
        results = {
            "Simulator":{
                "odd_composite_numbers": {}
            },
            "QHW":{
                "odd_composite_numbers": {}
            }
        }

    # Ensure the category and number exist in the results structure
    if category_name not in results[experiment]:
        results[experiment][category_name] = {}
        
    if str(number) not in results[experiment][category_name]:
        results[experiment][category_name][str(number)] = {"correct":0,"incorrect":0,"dnf":0}
    
    results[experiment][category_name][str(number)][shor_result] = results[experiment][category_name][str(number)][shor_result] + 1

    # Write the updated results back to the file
    with open(file_path, "w") as f:
        json.dump(results, f, indent=4)
