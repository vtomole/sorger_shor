import json
import os

def get_config(path):
    try:
        with open(path, 'r') as config_file:
            config = json.load(config_file)
            return config
    except FileNotFoundError:
        print("Config file not found.")
    except json.JSONDecodeError:
        print("Error parsing the JSON file.")


def configure_qiskit_runtime(config):
    from qiskit_ibm_runtime import QiskitRuntimeService

    # Save an IBM Quantum account and set it as your default account.
    QiskitRuntimeService.save_account(
        channel = "ibm_quantum",
        token = config['ibm_qp_token'],
        set_as_default = True,
        overwrite = True,
    )

    # Load saved credentials
    service = QiskitRuntimeService()
    return service


def configure(path='./config/config.json', isPrinting=True):
    if isPrinting:
        print("Configuring...")
    config = get_config(path)
    service = configure_qiskit_runtime(config)
    if isPrinting:
        print("Configuration complete.")
    return service
