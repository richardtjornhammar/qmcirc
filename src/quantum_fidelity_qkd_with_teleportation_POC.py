from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit_aer import Aer
from qiskit.quantum_info import Statevector, DensityMatrix, partial_trace, state_fidelity, Operator
import numpy as np

'''
This quantum circuit simulates a double quantum teleportation protocol with an intermediary node (Charlie) between Alice and Bob. Here's a breakdown of the system and quantum properties involved:
System Modeled

    Quantum communication network with entanglement swapping:
        The system involves three parties: Alice (qubit 0), Charlie (qubits 2 & 3), and Bob (qubit 4).
        Charlie acts as a relay, performing an entanglement swapping step to enable teleportation between Alice and Bob indirectly.

    Quantum state transfer across multiple nodes:
        Alice's unknown quantum state (prepared on qubit 0) is teleported through Charlie to Bob.
        The use of two Bell pairs (qubits 1-2 and 3-4) simulates distributed entanglement resources between the parties.

Quantum Properties Simulated

    Entanglement creation:
        Creation of two Bell pairs (maximally entangled states) between qubits (1,2) and (3,4).

    Entanglement swapping:
        Charlie performs a Bell measurement on his two qubits (2,3), which effectively "swaps" entanglement, linking Alice’s qubit (0) and Bob’s qubit (4).

    Bell measurements:
        Both Charlie and Alice perform Bell basis measurements on their qubit pairs.

    Conditional unitary corrections:
        Bob applies X and Z Pauli corrections on his qubit depending on classical measurement outcomes received from Charlie and Alice.

    Quantum state teleportation:
        The protocol transfers an unknown quantum state from Alice to Bob, using classical communication and pre-shared entanglement.

    Quantum fidelity calculation:
        The fidelity metric compares the final corrected state at Bob with the original Alice state, quantifying the teleportation quality.

Physical/Operational Context

    Quantum repeaters / long-distance communication:
        Charlie acts like a repeater node performing entanglement swapping to extend the range of teleportation.

    Noisy or ideal quantum channel simulation:
        In this ideal version, noise is absent, but the setup can be extended to include noise models representing decoherence and operational errors.

    Measurement and classical communication:
        Measurement collapses qubits and the classical bits store results for conditional corrections.

Summary

This system is a quantum teleportation network with entanglement swapping via an intermediate node, simulating:
    Preparation and distribution of entangled states,
    Local Bell-state measurements,
    Classical communication of measurement outcomes,
    Conditional corrections to recover the original quantum state remotely,
    Evaluation of teleportation fidelity.

It captures fundamental processes underlying quantum communication protocols, quantum repeaters, and quantum networks used to overcome distance and noise limitations in quantum state transfer.
'''


# Alice's initial state preparation circuit (1 qubit)
theta = np.pi / 4
qc_1q = QuantumCircuit(1)
qc_1q.ry(2 * theta, 0)
op_1q = Operator(qc_1q)
alice_sv = Statevector.from_instruction(qc_1q)
rho_alice = DensityMatrix(alice_sv)

# Registers: 5 qubits and 4 classical bits
q = QuantumRegister(5)
c = ClassicalRegister(4)
qc = QuantumCircuit(q, c)

# Step 0: Prepare Alice's qubit
qc.append(op_1q, [q[0]])

# Step 1: Create Bell pairs
qc.h(q[1])
qc.cx(q[1], q[2])
qc.h(q[3])
qc.cx(q[3], q[4])

# Step 2: Charlie's Bell measurement on q[2], q[3]
qc.cx(q[2], q[3])
qc.h(q[2])
qc.measure(q[2], c[0])  # Charlie bit 1
qc.measure(q[3], c[1])  # Charlie bit 2

# Step 3: Alice's Bell measurement on q[0], q[1]
qc.cx(q[0], q[1])
qc.h(q[0])
qc.measure(q[0], c[2])  # Alice bit 1
qc.measure(q[1], c[3])  # Alice bit 2

# Step 4: Bob applies corrections conditionally
from qiskit.circuit.library import XGate, ZGate
'''
# First Z for Charlie's first bit
qc.append(ZGate().to_mutable().c_if(c, 1 << 0), [q[4]])
same as: (OLD QISKIT VERSION FIX)
qc.append(ZGate().to_mutable(), [q[4]])
qc.data[-1].operation.condition = (c, 1 << 0)
'''

qc.append(ZGate().to_mutable(), [q[4]])
qc.data[-1].operation.condition = (c, 1 << 0)  # Charlie bit 1
qc.append(ZGate().to_mutable(), [q[4]])
qc.data[-1].operation.condition = (c, 1 << 2)  # Alice bit 1
qc.append(XGate().to_mutable(), [q[4]])
qc.data[-1].operation.condition = (c, 1 << 1)  # Charlie bit 2
qc.append(XGate().to_mutable(), [q[4]])
qc.data[-1].operation.condition = (c, 1 << 3)  # Alice bit 2

# Run on qasm_simulator with shots
sim = Aer.get_backend('qasm_simulator')
shots = 8192
result = sim.run(qc, shots=shots).result()
counts = result.get_counts()

print(f"Measurement counts:\n{counts}")

# Now reconstruct Bob's corrected states for each measurement outcome
# and calculate weighted fidelity with Alice's initial state

# We'll iterate through all measurement keys (bitstrings of length 4, c[3]c[2]c[1]c[0])
# Correction bits are mapped as:
# c[0] = Charlie bit 1 (Z)
# c[1] = Charlie bit 2 (X)
# c[2] = Alice bit 1   (Z)
# c[3] = Alice bit 2   (X)

# Function to apply corrections to Bob's qubit statevector
def apply_corrections(bob_sv, bits):
    # bits is string like '0110' c[3]c[2]c[1]c[0]
    # convert bits to int
    b3, b2, b1, b0 = [int(b) for b in bits]
    # Create a 1-qubit circuit for corrections
    corr_circ = QuantumCircuit(1)
    if b0 == 1:  # Charlie bit 1 -> Z
        corr_circ.z(0)
    if b2 == 1:  # Alice bit 1 -> Z
        corr_circ.z(0)
    if b1 == 1:  # Charlie bit 2 -> X
        corr_circ.x(0)
    if b3 == 1:  # Alice bit 2 -> X
        corr_circ.x(0)
    corr_op = Operator(corr_circ)
    return bob_sv.evolve(corr_op)

def project_statevector_on_bit(sv: Statevector, qubit: int, bit: str) -> Statevector:
    """Project a statevector on a measurement outcome bit (0 or 1) of a given qubit."""
    dim = 2 ** sv.num_qubits
    new_amplitudes = np.zeros(dim, dtype=complex)
    #
    for i in range(dim):
        # Check if qubit-th bit of basis state i matches 'bit'
        # Bit ordering: least significant bit is qubit 0
        if ((i >> qubit) & 1) == int(bit):
            new_amplitudes[i] = sv.data[i]
    #
    # Normalize the projected statevector
    norm = np.linalg.norm(new_amplitudes)
    if norm == 0:
        raise ValueError(f"Projection gave zero norm for qubit {qubit} bit {bit}")
    new_amplitudes /= norm
    #
    return Statevector(new_amplitudes)

# We need to get Bob's qubit reduced statevector for each measurement outcome
# Since we only have measurement counts (no final statevectors per shot),
# we have to simulate each outcome manually.

# Prepare to calculate total fidelity weighted by counts
total_fidelity = 0
total_counts = sum(counts.values())

for outcome, count in counts.items():
    # Build statevector conditioned on measurement outcome:
    # Start with full circuit but fix measurement results
    # We'll simulate the initial circuit but measure all, then project Bob's qubit accordingly
    # For simplicity, create a statevector circuit:
    qc_sv = QuantumCircuit(5)
    qc_sv.append(op_1q, [0])
    qc_sv.h(1)
    qc_sv.cx(1, 2)
    qc_sv.h(3)
    qc_sv.cx(3, 4)
    #
    qc_sv.cx(2, 3)
    qc_sv.h(2)
    qc_sv.cx(0, 1)
    qc_sv.h(0)
    #
    # Now project qubits [2,3,0,1] to measurement outcomes in 'outcome'
    # outcome bits correspond to c[3] c[2] c[1] c[0]
    # but c = [c0,c1,c2,c3], so reverse order
    m_bits = outcome[::-1]  # reverse to c[0],c[1],c[2],c[3]
    #
    # Project qubits to the measurement bits
    from qiskit.quantum_info import DensityMatrix
    sv = Statevector(qc_sv)
    for idx, bit in zip([2,3,0,1], m_bits):
        sv = project_statevector_on_bit(sv, idx, bit) # OLD QISKIT VERSION FIX
        #proj0 = DensityMatrix.from_label('0')
        #proj1 = DensityMatrix.from_label('1')
        #if bit == '0':
        #    sv = sv.projector(proj0, [idx])
        #else:
        #    sv = sv.projector(proj1, [idx])
    # Partial trace out all except Bob's qubit (4)
    rho_bob_cond = partial_trace(sv, [0,1,2,3])
    #
    # Convert reduced state to Statevector (if pure)
    # We'll try to get pure state vector for fidelity:
    # If rho is pure, it has rank 1 and can be converted
    # For mixed states, fidelity still works
    # Let's convert to Statevector if possible
    #
    try:
        bob_sv_cond = Statevector(rho_bob_cond)
    except:
        bob_sv_cond = rho_bob_cond
    #
    # Apply Bob's corrections based on measurement bits
    bob_sv_corr = apply_corrections(bob_sv_cond, outcome)
    #
    # Calculate fidelity
    fid = state_fidelity(rho_alice, bob_sv_corr)
    #
    # Weight by count
    total_fidelity += fid * count

print(f"Average teleportation fidelity over {shots} shots: {total_fidelity/total_counts:.4f}")
