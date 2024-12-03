# Experiment 2 -- Execution Time Comparison
In this experiment we compare our implementation with the implementation of Rui Maia and Tiago Leão \[1\].

## External Implementations
We decided to compare our software to the project developed by Rui Maia and Tiago Leão because their IBM adapted from this implementation for their own implementation of Shor in an older version of Qiskit \[2\].

The implementation of Rui Maia and Tiago Leão  contains two different variations:
- **Normal QFT** and
- **Sequential QFT**

### Normal QFT
This version implements Shor's Algorithm using a quantum circuit based on the first simplification from \[3\].
It employs the standard Quantum Fourier Transform (QFT) and features a top register with $2n$ qubits.

### Sequential QFT
This version implements Shor's Algorithm using a quantum circuit based on the second simplification from \[3\].
It uses the "sequential" Quantum Fourier Transform (QFT) with a top register of just 1 qubit, performing $2n$ measurements throughout the circuit.
These measurements directly yield $x_{final}$, eliminating the need for a QFT at the circuit's end.

## Methodology
### Our Implementation
We used the execution times that were generated using [experiment one](../e1/).

### External Implementation
We implemented an automatic script running both implementations - normal QFT and sequential QFT - the same way as we tested our implementation in the [first experiment](../e1/).

To run this code we have to use an old version of Qiskit; qiskit v0.x to be precise.
We therefore created a new `requirements.txt` for this experiment.

In case you want to create this environment yourself, e.g. by using `conda` you can use the following commands:
```Bash
conda create --name qiskit python=3.7
conda activate qiskit
pip install qiskit==0.33.0
```

### Used Numbers
We decided to use all odd composite numbers between 1-1000.
We decided to eliminate all even numbers since they are a trivial case for this algorithm.

References
---
\[1\] https://github.com/tiagomsleao/ShorAlgQiskit/  
\[2\] https://docs.quantum.ibm.com/api/qiskit/0.25/qiskit.algorithms.Shor  
\[3\] Stephane Beauregard (2003), Circuit for Shor's algorithm using 2n+3 qubits, Quantum Information and Computation, Vol. 3, No. 2 (2003) pp. 175-185. Also on quant-ph/0205095  
