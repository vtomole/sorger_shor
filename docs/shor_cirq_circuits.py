import cirq

from sorger_shor.shor import factoring_circuit

import supermarq as sm

def main(n):
    while True:
        try:
            qc = factoring_circuit(n)
            break
        except:
            continue
    cirq_circuit = sm.converters.qiskit_to_cirq(qc)
    print(cirq.num_qubits(cirq_circuit))

if __name__ == '__main__':
    main(n=100)