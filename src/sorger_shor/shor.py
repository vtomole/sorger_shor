#!/usr/bin/env python
# coding: utf-8

# In[1]:


import math
import random

import numpy as np
import pandas as pd
from numpy import pi
from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister
from qiskit.circuit.classical import expr
from qiskit.circuit.library import RZGate
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_aer import AerSimulator
from qiskit_ibm_runtime import SamplerV2 as Sampler

# from config.config import configure


# ## Condition to Expression
# 
# The function takes a classic register (reg) and a boolean values (value) and either sets the register to...
# - 1 if value == TRUE
# - 0 if value == FALSE

# In[2]:


def condition_to_expr(reg, value):
    return expr.lift(reg) if value else expr.logic_not(reg)


# ### Real Number to Continued Fraction (CF)
# 
# Computes the continued fraction representation of a real number.
# 
# $v = a_0 + \cfrac{1}{a_1 + \cfrac{1}{a_2 + \cfrac{1}{a_3 + \cdots}}}$
# 

# In[3]:


def continued_fraction(real_number, iter_num=5):
    terms = []

    # base case
    if iter_num <= 0:
        return terms
    a = 0
    x = real_number
    for i in range(iter_num):
        a = np.floor(x)
        terms.append(a)
        frac = x - a
        
        # if frac is close to zero then break
        if math.isclose(frac, 0, abs_tol=1e-09):
            break
        x = 1/frac
        
    return terms 


# ### Convert Continued Fraction (CF) to Fraction (F)
# 
# Useful in Shor's algorithm for deriving rational approximations, especially when trying to find the period of a function using the continued fraction expansion of a quantum measurement result.
# 
# Example input: [4, 2, 6, 7]
# 
# Example output: $\frac{415}{93}$

# In[4]:


def convCf2F(cf):
    r = False
    cf_size = len(cf)
    PArr = []
    QArr = []
    Pn = 0  # numerator
    Qn = 1  # denominator
    
    # Handling the empty continued fraction case
    if cf_size == 0:
        return r, Pn, Qn
    
    # Handling a single-element continued fraction
    elif cf_size == 1:
        Pn = cf[0]
        Qn = 1
        return True, Pn, Qn
    
    # Handling a two-element continued fraction
    elif cf_size == 2:
        Pn = cf[1] * (cf[0]) + 1
        Qn = cf[1] * 1 + 0
        return True, Pn, Qn

    # Initialize P0, P1, Q0, Q1
    PArr.append(cf[0])                        # P0
    PArr.append(cf[1] * cf[0] + 1)            # P1
    QArr.append(1)                            # Q0
    QArr.append(cf[1] * 1 + 0)                # Q1
    
    # Calculate Pn, Qn for n >= 2
    for idx in range(2, cf_size):
        Pn = cf[idx] * PArr[idx-1] + PArr[idx-2]
        Qn = cf[idx] * QArr[idx-1] + QArr[idx-2]
        PArr.append(Pn)
        QArr.append(Qn)
    
    return True, Pn, Qn


# In[5]:


# Example
def example():
    continued_fraction_example = [4, 2, 6, 7]
    fraction = convCf2F(continued_fraction_example)
    print(fraction)


# ### Greatest Common Divisor (GCD)
# 
# After finding the period $r$ of the function $f(x) = a^x \mod N$ we use the GCD to attempt factorization.
# 
# If $r$ is even and $a^{\frac{r}{2}} \not\equiv -1 (\mod N)$ then r is useful. We can calculate:
#     
# $$x = a^\frac{r}{2} \mod N$$
#     
# There are two potential factors of $N$:
# 
# $$factor_1=gcd⁡(x−1,N)$$
# 
# $$factor_2=gcd⁡(x+1,N)$$

# In[6]:


def gcd(a, b):
    # Base case: If one of the numbers is zero
    if a == 0:
        return b
    if b == 0:
        return a

    # Recursion
    return gcd(b, a%b)


# ### Inverse Quantum Fourier Transform (QFT)
# 
# The inverse QFT is crucial in Shor's algorithm because it enables the extraction of the periodicity of the function $f(x) = a^x \mod N$. This periodicity is the key to efficiently factoring the integer $N$, which is what makes Shor's algorithm exponentially faster than classical factoring algorithms.

# In[7]:


# apply inverse phase shift rotation gates in recursive 
def inv_qft_rotation(circuit, current_qubit, total_qubits, quantum_register):

    # If all qubits have been processed
    if current_qubit==0:
        return circuit
        
    # get theta index for rotation
    l = range(2, total_qubits-current_qubit+2)
    revl = list(l)[::-1]# Reverse the order of the list

    control_idx = 0
    current_qubit-=1

    # Controlled phase rotation
    for k in revl:
        # cp -> controlled phase gates
        circuit.cp(-pi/2**(k-1), quantum_register[control_idx], quantum_register[total_qubits-current_qubit-1])
        control_idx+=1
        
    circuit.h(quantum_register[total_qubits-current_qubit-1])
    
    return inv_qft_rotation(circuit, current_qubit, total_qubits, quantum_register)


# In[8]:


def inv_qft_rotation_simple(circuit, n, m, q, c):
    if n==0:
        return circuit
    # get theta index for rotation
    l = range(2, m-n+2)
    revl = list(l)[::-1]

    control_idx = 0
    n-=1

    angle = 0
    # calculate a sum of angles applied 
    for k in revl:
        #circuit.p(-pi/2**(k-1), q).c_if(c[control_idx], 1)

        # TODO comment back in when converter has been fixed to deal with this.
        # with circuit.if_test(condition_to_expr(c[control_idx], 1)):
        #     circuit.p(-pi/2**(k-1), q)
        control_idx+=1
    # apply hadamard gate
    circuit.h(q)


# In[9]:


# apply phase shift rotation gates in recursive 
def qft_rotation(circuit, n, q):
    if n==0:
        return circuit
    n-=1
    circuit.h(q[n])
    control_idx = n-1
    for k in range(2, n+2):
        circuit.cp(pi/2**(k-1), q[control_idx], q[n])
        control_idx-=1
    return qft_rotation(circuit, n, q)


# In[10]:


# apply qft transform
def qft(circuit, n, q, swap=True):
    qft_rotation(circuit, n, q)
    if swap:
        qft_swap(circuit, n, q)


# ### Swap functions
# 
# This function swaps qubits to reverse their order. It is often used to align qubit order after applying the QFT or Inverse QFT.

# In[11]:


def qft_swap(circuit, n, q):
    """Reverse the order of bits in a quantum register"""
    swap_num = (int)(n/2)
    m = n-1
    for i in range(swap_num):
        circuit.swap(q[i], q[m])
        m -= 1 


# In[12]:


def swap_classical_reg(c):
    """Reverse the order of bits in a classical register"""
    for i in range(len(c)//2):
        c[i], c[len(c)-1-i] = c[len(c)-1-i], c[i]


# In[13]:


def inv_qft(circuit, n, m, q, swap=True):
    """The function applies the inverse QFT and swaps the order of qubits, if swap=True"""
    if swap:
        qft_swap(circuit, n, q)
    inv_qft_rotation(circuit, n, m, q)


# In[14]:


def addr(circuit, target_qubits, num_qubits, phase_coeff, ctrl_qubit1=None, ctrl_qubit2=None):
    """
    Applies a series of controlled or uncontrolled phase rotations 
    to a set of target qubits, used to perform an addition.
    
    Parameters:
    - circuit (QuantumCircuit): The quantum circuit to which the gates will be applied.
    - target_qubits (list[int] or range): List or range of qubits to apply the rotations on.
    - num_qubits (int): The number of target qubits.
    - phase_coeff (float): The coefficient 'a' used to compute the rotation angles.
    - ctrl_qubit1 (int, optional): The control qubit for singly controlled rotations.
    - ctrl_qubit2 (int, optional): The second control qubit for doubly controlled rotations.
    
    Behavior:
    - If `ctrl_qubit1` and `ctrl_qubit2` are both None, applies standard `RZ` rotations to the target qubits.
    - If `ctrl_qubit1` is specified (but not `ctrl_qubit2`), applies a controlled phase (`CP`) rotation.
    - If both `ctrl_qubit1` and `ctrl_qubit2` are specified, applies a doubly controlled `RZ` gate.
    """
    
    # Case 1: Apply standard RZ rotations if no control qubits are provided
    if ctrl_qubit1 is None and ctrl_qubit2 is None:
        for i in range(num_qubits):
            angle = 2 * np.pi * phase_coeff / 2**(i + 1)
            circuit.rz(angle, target_qubits[i])
    
    # Case 2: Apply controlled phase rotations (CP) if one control qubit is provided
    elif ctrl_qubit1 is not None and ctrl_qubit2 is None:
        for i in range(num_qubits):
            angle = 2 * np.pi * phase_coeff / 2**(i + 1)
            circuit.cp(angle, ctrl_qubit1, target_qubits[i])
    
    # Case 3: Apply doubly controlled RZ rotations if two control qubits are provided
    elif ctrl_qubit1 is not None and ctrl_qubit2 is not None:
        for i in range(num_qubits):
            angle = 2 * np.pi * phase_coeff / 2**(i + 1)
            gate = RZGate(angle)
            # Apply a doubly controlled version of the RZ gate
            circuit.append(gate.control(2), [ctrl_qubit1, ctrl_qubit2, target_qubits[i]])
    
    # Case 4: Handle any invalid parameter combinations
    else:
        raise ValueError("Invalid combination of control qubits provided.")


# In[15]:


# The exact opposite of the above cell
def subtract(circuit, target_qubits, num_qubits, phase_coeff, ctrl_qubit1=None, ctrl_qubit2=None):
    """
    Applies a series of controlled or uncontrolled phase subtraction rotations to a set of target qubits.
    
    Parameters:
    - circuit (QuantumCircuit): The quantum circuit to which the gates will be applied.
    - target_qubits (list[int] or range): List or range of qubits to apply the rotations on.
    - num_qubits (int): The number of target qubits.
    - phase_coeff (float): The coefficient 'a' used to compute the rotation angles.
    - ctrl_qubit1 (int, optional): The control qubit for singly controlled rotations.
    - ctrl_qubit2 (int, optional): The second control qubit for doubly controlled rotations.
    
    Behavior:
    - If `ctrl_qubit1` and `ctrl_qubit2` are both None, applies standard `RZ` rotations with a negative angle to the target qubits.
    - If `ctrl_qubit1` is specified (but not `ctrl_qubit2`), applies a controlled phase (`CP`) rotation with a negative angle.
    - If both `ctrl_qubit1` and `ctrl_qubit2` are specified, applies a doubly controlled `RZ` gate with a negative angle.
    """
    
    # Case 1: Apply standard RZ rotations with negative angles if no control qubits are provided
    if ctrl_qubit1 is None and ctrl_qubit2 is None:
        for i in range(num_qubits):
            angle = -2 * np.pi * phase_coeff / 2**(i + 1)
            circuit.rz(angle, target_qubits[i])
    
    # Case 2: Apply controlled phase rotations (CP) with negative angles if one control qubit is provided
    elif ctrl_qubit1 is not None and ctrl_qubit2 is None:
        for i in range(num_qubits):
            angle = -2 * np.pi * phase_coeff / 2**(i + 1)
            circuit.cp(angle, ctrl_qubit1, target_qubits[i])
    
    # Case 3: Apply doubly controlled RZ rotations with negative angles if two control qubits are provided
    elif ctrl_qubit1 is not None and ctrl_qubit2 is not None:
        for i in range(num_qubits):
            angle = -2 * np.pi * phase_coeff / 2**(i + 1)
            gate = RZGate(angle)
            # Apply a doubly controlled version of the RZ gate
            circuit.append(gate.control(2), [ctrl_qubit1, ctrl_qubit2, target_qubits[i]])
    
    # Case 4: Handle any invalid parameter combinations
    else:
        raise ValueError("Invalid combination of control qubits provided.")


# ### Modular Addition and Substraction
# 
# The two following functions implement an addition/substraction of a value $a$ modulus $N$.
# 
# The modular addition of a value is an important step in computing the period of the function
# 
# $$f(x) = a^x \mod N$$
# 
# which we need to factor the integer N.

# In[16]:


def mod_addr(circuit, q, ctrl1, ctrl2, aux, a, N, n):
    """
    Implements a modular addition routine within the context of Shor's algorithm.
    This function performs a modular add operation controlled by two qubits, 
    along with modular subtraction and conditional logic to ensure results 
    stay within the modulus N.
    
    Parameters:
    - circuit (QuantumCircuit): The quantum circuit to apply the operations on.
    - q (list[int] or range): The quantum register (target qubits).
    - ctrl1 (int): First control qubit for doubly controlled operations.
    - ctrl2 (int): Second control qubit for doubly controlled operations.
    - aux (int): Auxiliary qubit used for overflow detection.
    - a (float): Value to be added in the modular addition.
    - N (float): Modulus value for modular arithmetic.
    - n (int): Number of qubits in the quantum register `q`.
    
    This function is part of implementing modular exponentiation, crucial for 
    finding the period in Shor's algorithm.
    """

    # Step 1: Doubly controlled addition of 'a' to the quantum register q
    addr(circuit, q, n, a, ctrl1, ctrl2)
    
    # Step 2: Subtract the modulus 'N' from the quantum register q
    subtract(circuit, q, n, N)
    
    # Step 3: Apply the Inverse Quantum Fourier Transform (QFT) to the register q
    # This helps in transforming the state to detect if an overflow has occurred
    inv_qft(circuit, n, n, q, swap=False)
    
    # Step 4: Check for overflow using the most significant bit of q with a CNOT gate
    circuit.cx(q[n-1], aux)
    
    # Step 5: Reapply the QFT to revert the register back to its original basis
    qft(circuit, n, q, swap=False)
    
    # Step 6: Conditionally add the modulus 'N' back if an overflow was detected
    # This addition is controlled by the auxiliary qubit
    addr(circuit, q, n, N, aux)
    
    # Step 7: Doubly controlled subtraction of 'a' to correct the result
    addr(circuit, q, n, -a, ctrl1, ctrl2)
    
    # Step 8: Apply the Inverse QFT again to prepare for the next operation
    inv_qft(circuit, n, n, q, swap=False)
    
    # Step 9: Flip the most significant bit to check the overflow flag
    circuit.x(q[n-1])
    
    # Step 10: Use the auxiliary qubit to reset the overflow flag if needed
    circuit.cx(q[n-1], aux)
    
    # Step 11: Restore the most significant bit to its original state
    circuit.x(q[n-1])
    
    # Step 12: Apply the QFT one last time to finalize the modular addition
    qft(circuit, n, q, swap=False)
    
    # Step 13: Reapply the doubly controlled addition of 'a' as the final step
    addr(circuit, q, n, a, ctrl1, ctrl2)


# In[17]:


def inv_mod_addr(circuit, q, ctrl1, ctrl2, aux, a, N, n):
    """
    Implements the inverse of the modular addition routine within the context of Shor's algorithm.
    This function undoes the effect of modular addition and subtraction of 'a' and 'N', and 
    applies quantum operations to reverse the modular arithmetic.

    Parameters:
    - circuit (QuantumCircuit): The quantum circuit to apply the operations on.
    - q (list[int] or range): The quantum register (target qubits).
    - ctrl1 (int): First control qubit for doubly controlled operations.
    - ctrl2 (int): Second control qubit for doubly controlled operations.
    - aux (int): Auxiliary qubit used for overflow detection.
    - a (float): Value to be subtracted in the modular subtraction.
    - N (float): Modulus value for modular arithmetic.
    - n (int): Number of qubits in the quantum register `q`.
    
    This function is used as part of the modular exponentiation process in Shor's algorithm.
    It reverses the effect of adding and subtracting 'a' and 'N' while applying the Quantum Fourier Transform (QFT) to ensure that the period-finding operation is correct.
    """
    
    # Step 13: Doubly controlled subtraction of 'a' to reverse previous modular addition
    subtract(circuit, q, n, a, ctrl1, ctrl2)
    
    # Step 12: Apply the Inverse QFT to prepare for the next operation
    inv_qft(circuit, n, n, q, False)
    
    # Step 11: Flip the most significant bit of the register q
    circuit.x(q[n-1])
    
    # Step 10: CNOT between the most significant bit and the auxiliary qubit
    circuit.cx(q[n-1], aux)
    
    # Step 9: Flip the most significant bit back to its original state
    circuit.x(q[n-1])
    
    # Step 8: Apply QFT to revert the quantum state back to the computational basis
    qft(circuit, n, q, False)
    
    # Step 7: Doubly controlled addition of 'a' (to reverse subtraction of 'a')
    addr(circuit, q, n, a, ctrl1, ctrl2)
    
    # Step 6: Singly controlled subtraction of 'N' to reverse the addition of 'N'
    subtract(circuit, q, n, N, aux, None)
    
    # Step 5: Apply the Inverse QFT again to transform back after subtraction
    inv_qft(circuit, n, n, q, False)
    
    # Step 4: CNOT between the most significant bit and the auxiliary qubit
    circuit.cx(q[n-1], aux)
    
    # Step 3: Apply QFT to revert to the computational basis
    qft(circuit, n, q, False)
    
    # Step 2: Add 'N' (reversing previous subtraction of 'N')
    addr(circuit, q, n, N, None, None)
    
    # Step 1: Doubly controlled subtraction of 'a' to reverse addition of 'a'
    subtract(circuit, q, n, a, ctrl1, ctrl2)


# In[18]:


def c_mult(circuit, ctrl, q, aux, a, N, n):
    """
    Implements the Controlled Multiplication (CMULT) gate for modular exponentiation.
    This gate performs a controlled modular multiplication of the quantum register 
    `q` with a constant `a`, modulo `N`, based on the control qubit `ctrl`. It uses 
    QFT and Inv QFT to facilitate the modular arithmetic.

    Parameters:
    - circuit (QuantumCircuit): The quantum circuit on which the operations are applied.
    - ctrl (int): The control qubit that determines whether the multiplication occurs.
    - q (list[int] or range): The quantum register containing the input values to be multiplied.
    - aux (int): The auxiliary qubits used for the QFT and overflow control.
    - a (int): The constant multiplier (the base value for multiplication).
    - N (int): The modulus for the modular arithmetic.
    - n (int): The number of qubits in the quantum register `q`.
    """
    
    # Step 1: Apply Quantum Fourier Transform (QFT) to the auxiliary qubits
    # This prepares the quantum register for modular arithmetic and multiplication.
    qft(circuit, n+1, aux, False)
    
    # Step 2: Apply n modular addition gates (modular multiplications) controlled by `ctrl`
    for i in range(0, n):
        # For each qubit in the quantum register `q`, perform a modular addition
        # based on the power of `a` corresponding to each qubit (i.e., a^i mod N).
        # This simulates the multiplication of each power of `a` modulo N.
        mod_addr(circuit, aux, q[i], ctrl, aux[n+1], ((2**i) * a) % N, N, n+1)
    
    # Step 3: Apply Inverse Quantum Fourier Transform (Inv QFT)
    # This undoes the QFT and returns the register to the computational basis.
    inv_qft(circuit, n+1, n+1, aux, False)


# In[19]:


# Inverse Controlled Multiplication(CMULT) gate
def inv_c_mult(circuit, ctrl, q, aux, a, N, n):
    # Step 1: Apply Quantum Fourier Transform (QFT) to the auxiliary qubits
    # The QFT prepares the quantum register to perform the inverse of the modular multiplication.
    qft(circuit, n+1, aux, False)
    
    # Step 2: Apply n modular subtraction gates (inverse modular multiplication)
    for i in range(0, n):
        idx = n-1-i  # Reverse the order for the inverse operation
        # For each qubit in the quantum register `q`, perform the inverse modular addition
        # by subtracting the corresponding value of `a` mod N.
        inv_mod_addr(circuit, aux, q[idx], ctrl, aux[n+1], ((2**idx) * a) % N, N, n+1)
    
    # Step 3: Apply Inverse Quantum Fourier Transform (Inv QFT)
    # This undoes the QFT and returns the register to the computational basis.
    inv_qft(circuit, n+1, n+1, aux, False)


# In[20]:


def gen_ua(circuit, ctrl, eigen_vec_reg, aux, a, N, n, isPrinting=True):
    """
    Implements a general unitary operator (Ua) used in Shor's algorithm for modular exponentiation.
    This operator applies a series of modular exponentiation gates, controlled swaps, and the
    modular inverse of the base `a`, in order to simulate a quantum unitary transformation
    that is essential for quantum phase estimation.

    Parameters:
    - circuit (QuantumCircuit): The quantum circuit on which the operations are applied.
    - ctrl (int): The control qubit that determines whether the modular exponentiation occurs.
    - eigen_vec_reg (list[int] or range): The quantum register containing the eigenvector.
    - aux (int): The auxiliary qubits used for performing the QFT and modular arithmetic.
    - a (int): The base value for the modular exponentiation.
    - N (int): The modulus for the modular arithmetic.
    - n (int): The number of qubits in the quantum register `eigen_vec_reg`.
    """
    
    # Step 1: Apply Controlled Modular Multiplication (CMULT) with base `a` mod `N`
    c_mult(circuit, ctrl, eigen_vec_reg, aux, a, N, n)
    
    # Step 2: Apply controlled swaps between the eigenvector register and auxiliary qubits
    # This is a key step in Shor's algorithm that facilitates the entanglement of the quantum states.
    for i in range(0, n):
        circuit.cswap(ctrl, eigen_vec_reg[i], aux[i])

    # Step 3: Calculate the modular inverse of `a` modulo `N`
    a_mod_inv = mod_inverse(a, N, isPrinting)
    
    # Step 4: Apply the inverse controlled modular multiplication (inv_c_mult) with `a^-1` mod `N`
    inv_c_mult(circuit, ctrl, eigen_vec_reg, aux, a_mod_inv, N, n)


# ### Extended GCD
# 
# The function extended_gcd(a, b) implements the Extended Euclidean Algorithm, which computes the greatest common divisor (GCD) of two integers $a$ and $b$, and also finds the coefficients $x$ and $y$ such that:
# 
# $$a*x+b*y = gcd(a,b)$$

# In[21]:


def extended_gcd(a, b):
    """
    Returns:
    - gcd (int): The greatest common divisor of a and b.
    - x (int): The coefficient for a in the equation.
    - y (int): The coefficient for b in the equation.
    """
    if a == 0:
        return b, 0, 1
    else:
        gcd, x, y = extended_gcd(b % a, a)
        return gcd, y - b//a * x, x


# ### Modular Inverse
# 
# The function mod_inverse(a, m) computes the modular inverse of $a$ modulo $m$. The modular inverse of $a$ modulo $m$ is the integer $x$ such that:
# 
# $$a*x \equiv 1 \mod m$$
# 
# This function uses the Extended Euclidean Algorithm to compute the modular inverse, relying on the extended_gcd function.

# In[22]:


def mod_inverse(a, m, isPrinting=True):
    gcd, x, _ = extended_gcd(a, m)
    if gcd != 1:
            raise ValueError(f"The modular inverse does not exist for {a} modulo {m}.")
    else:
        return x % m


# In[23]:


# choose a randomly 2 <= a <= N-1
def choose_random_a(N):
    return random.randint(2, N - 1)


# In[24]:

def factoring_circuit(N, isCircuitOptimized=True, isPrinting=False):
    # Step 2: Calculate the number of qubits needed
    n = math.ceil(math.log(N, 2))
    # Step 3: Perform various iterations to find the factors
    # Step 4: Set up quantum registers
    aux_reg = QuantumRegister(n + 2, name="aux")
    eigen_vec_reg = QuantumRegister(n, name="eigen vec")
    counting_reg = None
    counting_reg_1qubit = None
    # Choose counting register based on optimization flag
    if isCircuitOptimized:
        counting_reg_1qubit = QuantumRegister(1, name="counting_1_qubit")
    else:
        counting_reg = QuantumRegister(2 * n, name="counting_2n_qubit")
    classic_reg = ClassicalRegister(2 * n, name="classic")
    # Step 5: Create the Quantum Circuit
    if isCircuitOptimized:
        circuit = QuantumCircuit(eigen_vec_reg, aux_reg, counting_reg_1qubit, classic_reg)
    else:
        circuit = QuantumCircuit(counting_reg, eigen_vec_reg, aux_reg, classic_reg)
        circuit.h(counting_reg)
    # Step 6: Initialize eigenvector to |00..01>
    circuit.x(eigen_vec_reg[0])
    # Step 7: Performing a lucky guess by calculating the gcd for a random 'a' auch that 1 < a < N
    a = choose_random_a(N)

    # Step 8: Run the order-finding quantum computing subroutine
    if isCircuitOptimized:
        # Step 9: Serialize the phase estimation process.
        # Instead of having 2n qubits in superposition all at once,
        # the circuit reuses a single qubit, performing the controlled operations iteratively
        serialize_2n(circuit, counting_reg_1qubit, eigen_vec_reg, aux_reg, classic_reg, a, N, n)
    else:
        # Step 9: Apply unitary operator (gen_ua) controlled by each counting qubit
        # Using the property : ((a^n) * x) mod N = (a..(a(ax)mod N)mod N)..)mod N)
        # we don't need to apply the Ua gate n-times to get (Ua)^n because we can directly
        # run Ua^n (where a^n mod N in computed classically) which is the same as (Ua)^n
        for i in range(2 * n):
            # Compute classically: a^(2^i)
            gen_ua(circuit, counting_reg[i], eigen_vec_reg, aux_reg, int(pow(a, pow(2, i))), N, n, isPrinting)

        inv_qft(circuit, 2 * n, 2 * n, counting_reg, True)

        circuit.measure(counting_reg, classic_reg)
    return circuit



def factor_integer(N, isSimulator=True, isCircuitOptimized=True, isPrinting=False, device_backend = None):
    """
    Implementation of Shor's Algorithm to factorize a composite number N using Quantum Computing.
    
    Parameters:
    - N (int): The composite number to be factorized.
    - isSimulator (bool): Whether to run on a simulator (default: True).
    - isCircuitOptimized (bool): Use optimized circuit with 1 qubit counting register (default: False).
    
    Returns:
    - Tuple: (success (bool), p (int), q (int)), where p and q are the factors of N if found.
    """
    
    MAX_ITER_REAL_DEVICE = 1
    MAX_ITER_SIM = 10
    n = math.ceil(math.log(N,2))
    max_iter = MAX_ITER_SIM if isSimulator else MAX_ITER_REAL_DEVICE
    a = 2 #default a
    #
    # # Step 3: Perform various iterations to find the factors
    for count in range(max_iter):
        circuit = factoring_circuit(N, isCircuitOptimized, isPrinting)
        
        # Step 10: Set up backend
        shots = 10
        backend = AerSimulator() if isSimulator else device_backend
        if isPrinting:
            print(f"Step 10: Setting up '{backend.name}' as a backend")

        pm = generate_preset_pass_manager(backend=backend, optimization_level=3)
        isa_qc = pm.run(circuit)

        sampler = Sampler(backend)
        job = sampler.run([isa_qc], shots=shots)

        if isPrinting:
            print(f"Job ID: {job.job_id()}")

        # Step 11: Analyzing the results
        pub_result = job.result()[0]
        counts = pub_result.data.classic.get_counts()

        # Convert measurement results to phases
        rows, measured_phases = [], []
        for output in counts:
            decimal = int(output, 2)  # Convert (base 2) string to decimal
            phase = decimal/(2**(2*n))  # Find corresponding eigenvalue
            measured_phases.append(phase)
            rows.append([f"{output}(bin) = {decimal:>3}(dec)", f"{decimal}/{2**(2*n)} = {phase:.2f}"])

        df = pd.DataFrame(rows, columns=["Register Output", "Phase"])
        if isPrinting:
            print("Step 11: Analyzing the results\n")
            print(df)
            print('\n')

        # Step 12: Convert phases to continued fractions to find guesses for r
        rows = []
        for phase in measured_phases:
            terms = continued_fraction(phase)
            result = convCf2F(terms)
            if(result):
                rows.append([phase, f"{result[1]}/{result[2]}", result[2]])

        df = pd.DataFrame(rows, columns=["Phase", "Fraction", "Guess for r"])
        if isPrinting:
            print("Step 12: Convert phases to continued fractions to find guesses for r\n")
            print(df)
            print('\n')

        # Step 13: Check if a valid period r was found and use it to find factors
        if isPrinting:
            print(f"Step 13: Evaluating the {len(rows)} possible candidates for the period r")
        for i, row in enumerate(rows):
            r = row[2]
            if isPrinting:
                print(f"Candidate {i}: Guessed period r = {r}")

            if r % 2 == 0: # r must be even
                exponent = int(r // 2)
                x = pow(a, exponent, N)

                # Check (a^(r/2) ± 1) mod N
                factor1 = gcd(x - 1, N)
                factor2 = gcd(x + 1, N)

                if factor1 != 1 and factor1 != N:
                    if isPrinting:
                        print(f"RESULT: Found the factor {factor1} with {N} = {factor1} x {N // factor1}")
                    return True, factor1, N // factor1, isa_qc
                if factor2 != 1 and factor2 != N:
                    if isPrinting:
                        print(f"RESULT: Found the factor {factor2} with {N} = {factor2} x {N // factor2}")
                    return True, factor2, N // factor2, isa_qc
    if isPrinting:                
        print("FAILURE: Factorization failed")
    return False, 0, 0


# In[25]:


def check_for_simple_factors(N, isPrinting=True):
    """
    Checks for validity and simple factors of N such as even numbers, powers of 2, and prime numbers.
    """
    if N <= 2: 
        # invalid N
        if isPrinting:
            print(f"N should be greater than 2; not {N}")
        return True, 0, 0

    if (N%2)==0:
        # N is even
        p = 2
        q = N / p
        if isPrinting:
            print("N is even, factoring is done.")
        return True, p, q

    # check if N is a prime or prime power
    k = 4; # Accuracy level of the primality test (number of iterations)
    if isPrime(N, k):
        if isPrinting:
            print(f"{N} is a prime")
        return True, N, 1
    else:
        i, power = is_prime_power(N)
        if i is not None: # N is a prime power
            if isPrinting:
                print(f"{N} is a prime power: {i} ^ {power}")
            return True, i, (int)(N/i)
    return False, 0, 0

# In[26]:


# Utility function to do modular exponentiation. It returns (x^y) % p
def power(x, y, p):
	
	# Initialize result
	res = 1 
	
	# Update x if it is more than or equal to p
	x = x % p 
	while (y > 0):
		
		# If y is odd, multiply x with result
		if (y & 1):
			res = (res * x) % p

		# y must be even now
		y = y>>1 # y = y/2
		x = (x * x) % p
	
	return res


# This function is called for all k trials.
# It returns false if n is composite and returns false if n is probably prime.
# d is an odd number such that d*2<sup>r</sup> = n-1 for some r >= 1
def miillerTest(d, n):
	
	# Pick a random number in [2..n-2]
	# Corner cases make sure that n > 4
	a = 2 + random.randint(1, n - 4)

	# Compute a^d % n
	x = power(a, d, n)

	if (x == 1 or x == n - 1):
		return True

	# Keep squaring x while one of the following doesn't happen
	# (i)   d does not reach n-1
	# (ii)  (x^2) % n is not 1
	# (iii) (x^2) % n is not n-1
	while (d != n - 1):
		x = (x * x) % n
		d *= 2

		if (x == 1):
			return False
		if (x == n - 1):
			return True

	# Return composite
	return False


# It returns false if n is composite and returns true if n is probably prime.
# k is an input parameter that determines accuracy level. Higher value of k indicates more accuracy.
def isPrime(n, k):
	
	# Corner cases
	if (n <= 1 or n == 4):
		return False
	if (n <= 3):
		return True

	# Find r such that n = 2^d * r + 1 for some r >= 1
	d = n - 1
	while (d % 2 == 0):
		d //= 2

	# Iterate given number of 'k' times
	for i in range(k):
		if (miillerTest(d, n) == False):
			return False

	return True


# In[27]:


# Example:
def example2():
	k = 4 

	print("All primes smaller than 100: ")
	for n in range(1,100):
		if (isPrime(n, k)):
			print(n , end=" ")


# In[28]:


def sieve_of_eratosthenes(n):
    """
    Returns a list of prime numbers up to n using the Sieve of Eratosthenes algorithm.
    """
    # List of numbers from 0 to n
    numbers = list(range(n+1))
    # Mark non-prime numbers as 0
    numbers[0] = numbers[1] = 0 # Mark 0 and 1 as non-prime (0)

    # First prime is 2
    p = 2

    while p < n:
        if p:
            index = p + p
            while index <= n:
                numbers[index] = 0
                index += p
        p += 1

    # Return primes (all numbers that are not marked as 0)
    return [i for i in numbers if i != 0]


# In[29]:


def is_prime_power(N):
    """
    Check if N is prime power.
    """
    prime_list = sieve_of_eratosthenes(N)

    for i in prime_list:
        m = N
        power = 0
        while m % i == 0:
            m = math.floor(m / i)
            power += 1
        if m == 1:
            return i, power
    return None, None


# In[30]:


def serialize_2n(circuit, counting_reg_1qubit, eigen_vec_reg, aux_reg, classic_reg, a, N, n):
    """
    Serialize the phase estimation process for optimization purposes.
    Make a one-qubit control circuit in which a series of inverse-QFT are applied.
    """
    m = 2 * n
    for i in range(m):
        # Apply a hadamard gate
        circuit.h(counting_reg_1qubit)
        # Apply controlled Ua gate
        gen_ua(circuit, counting_reg_1qubit, eigen_vec_reg, aux_reg, int(pow(a, pow(2, 2*n-1-i))), N, n)
        
        # Apply serialized inverse QFT on 1 qubit
        inv_qft_rotation_simple(circuit, m-i, m, counting_reg_1qubit, classic_reg)

        # Measure
        circuit.measure(counting_reg_1qubit, classic_reg[i])
        
        # Apply X if the measured qubit is 1
        # TODO comment back in when converter has been modified to handle this.
        # with circuit.if_test(condition_to_expr(classic_reg[i],1)):
        #    circuit.x(counting_reg_1qubit)


# In[31]:


def configure_service(isSimulator, isPrinting=True):
    # service = configure(isPrinting=isPrinting)
    # if isSimulator == False:
    #     device_backend = service.least_busy(operational=True, dynamic_circuits=True)
    # else:
    device_backend = None

    if isPrinting:
        print("backend: ", device_backend.name)
        print("Number of qubits:", device_backend.configuration().n_qubits)

    return device_backend

# In[32]:


def run_shor(N, isSimulator, isCircuitOptimized, isPrinting):
    r, p, q = factor_integer(N, isSimulator, isCircuitOptimized, isPrinting)
    return r, p, q


# In[33]:


if __name__ == "__main__":
    # Declare the number to factor: N
    N = 33

    # Run on a simulator or real quantum hardware 
    isSimulator = False

    # Optimize circuit to use only 2n+3 qubits
    isCircuitOptimized = True

    # Decide if steps should be printed
    isPrinting = True

    # Configure the service
    backend = configure_service(isSimulator, isPrinting)

    print(f"backend_service: {backend}")
    
    # Run Shor's Algorithm to factorize the number N
    r, p, q = factor_integer(N, isSimulator, isCircuitOptimized, isPrinting, backend)

    # Print the results
    if r:
        print("The factoring has succeeded.")
        print(f"{N} = {p} x {q}")
    else:
        print("The factoring failed")


