from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit_aer import Aer
from qiskit.providers.aer.noise import NoiseModel, depolarizing_error
from qiskit.quantum_info import Statevector, partial_trace, DensityMatrix, state_fidelity
from qiskit.quantum_info import DensityMatrix, Operator
from qiskit_aer.noise import NoiseModel, thermal_relaxation_error, depolarizing_error
import numpy as np
from numpy import pi
from typing import Union

def create_bell_pair_circuit():
    """Return a QuantumCircuit that prepares a Bell state."""
    circ = QuantumCircuit(2)
    circ.h(0)
    circ.cx(0, 1)
    return circ

def create_noise_model(hop_distance: float, t1_base=50e3, t2_base=70e3, gate_time=100):
    """Create a thermal noise model for a given hop distance (in km)."""
    t1 = t1_base / (1 + hop_distance)
    t2 = t2_base / (1 + hop_distance)
    thermal_noise = thermal_relaxation_error(t1, t2, gate_time)
    noise_model = NoiseModel()
    noise_model.add_all_qubit_quantum_error(thermal_noise, ["cx", "id"])
    return noise_model

# Helper: create distance-dependent noise model
def create_simple_noise_model(distance):
    error_prob = min(0.1, 0.01 * distance)  # adjust scaling as needed
    noise_model = NoiseModel()
    error_1q = depolarizing_error(error_prob, 1)
    error_2q = depolarizing_error(error_prob * 1.5, 2)
    #
    for q in range(5):
        noise_model.add_quantum_error(error_1q, ['x', 'h', 'ry', 'id'], [q])
    noise_model.add_quantum_error(error_2q, ['cx'], [1, 2])
    noise_model.add_quantum_error(error_2q, ['cx'], [3, 4])
    return noise_model

def create_noise_model_dep(distance):
    noise_model = NoiseModel()
    #
    # Distance-dependent relaxation times (ns)
    t1 = max(5e4 - distance * 2e3, 1e3)  # Clamp at 1 µs (1000 ns)
    t2 = max(7e4 - distance * 3e3, 1e3)
    gate_time_1q = 100  # ns typical 1-qubit gate duration
    gate_time_2q = 300  # ns typical 2-qubit gate duration
    #
    # 1-qubit thermal relaxation error
    error_1q = thermal_relaxation_error(t1, t2, gate_time_1q)
    #
    # 2-qubit thermal relaxation error (tensor product of two 1-qubit errors)
    error_2q = thermal_relaxation_error(t1, t2, gate_time_2q).tensor(
        thermal_relaxation_error(t1, t2, gate_time_2q)
    )
    #
    # Apply errors to all relevant gates and qubits
    for q in range(5):
        noise_model.add_quantum_error(error_1q, ['x', 'h', 'ry'], [q])
    #
    noise_model.add_quantum_error(error_2q, ['cx'], [1, 2])
    noise_model.add_quantum_error(error_2q, ['cx'], [3, 4])
    #
    return noise_model

def dm_to_sv_if_pure(dm: DensityMatrix):
    """Convert DensityMatrix to Statevector if pure, else return DensityMatrix."""
    import numpy as np
    eigvals, eigvecs = np.linalg.eigh(dm.data)
    max_eigval = np.max(eigvals)
    if np.isclose(max_eigval, 1.0, atol=1e-6):
        max_eigvec = eigvecs[:, np.argmax(eigvals)]
        return Statevector(max_eigvec)
    else:
        return dm

def apply_corrections(state: Union[Statevector, DensityMatrix], outcome: str):
    """Apply conditional X/Z gates to Bob's qubit based on measurement outcome."""
    bits = [int(b) for b in outcome]
    circuit = QuantumCircuit(1)
    if bits[0]: circuit.z(0)  # Charlie Z
    if bits[1]: circuit.x(0)  # Charlie X
    if bits[2]: circuit.z(0)  # Alice Z
    if bits[3]: circuit.x(0)  # Alice X
    return state.evolve(Operator(circuit))


def project_statevector_on_bit(sv: Statevector, qubit: int, bit: str) -> Statevector:
    """Project a statevector on a measurement outcome bit (0 or 1) of a given qubit."""
    dim = 2 ** sv.num_qubits
    new_amplitudes = np.zeros(dim, dtype=complex)
    for i in range(dim):
        if ((i >> qubit) & 1) == int(bit):
            new_amplitudes[i] = sv.data[i]
    norm = np.linalg.norm(new_amplitudes)
    if norm == 0:
        raise ValueError(f"Projection gave zero norm for qubit {qubit} bit {bit}")
    new_amplitudes /= norm
    return Statevector(new_amplitudes)
