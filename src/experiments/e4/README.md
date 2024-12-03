# Experiment 4 -- Shor Hardware Accuracy
In this experiment we analyze the accuracy of our implementation when run on real-life quantum hardware.

## Methodology
We developed an automatic script running our implementation of Shor's Algorithm with different values.

We run Shor's algorithm for every input 10 times and record the number of correct and incorrect answers, and the number of times the algorithm does not finish.

### Used Numbers
We decided to use all odd composite numbers up to 81. The current hardware does not allow for larger numbers due to the assembly becoming too long.

We decided to eliminate all even numbers, since they are a trivial case for this algorithm.
