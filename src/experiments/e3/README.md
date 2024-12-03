# Experiment 3 -- Quantum Circuit Optimization
In this experiment we analyze the average runtime needed for this implemenation of the Shor's Algorithm to find the solution.
We run the circuit in an unoptimized form here, so we can later compare it's results with the optimized one in first experiment.

## Methodology
We developed an automatic script running our implementation of Shor's Algorithm with different values.

We run and time the execution time for every input multiple times to generate an average runtime per input.

### Used Numbers
We decided to use all odd composite numbers between 1-1000.
We decided to eliminate all even numbers, since they are a trivial case for this algorithm.
