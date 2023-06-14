import numpy as np
from scipy.spatial.distance import hamming
from rank_decomposition.utils.binary_string_utils import strings_with_weight, read_binary_array, change_bits

global even_weight_bin, even_weight


def _coefficient_formula(label_index, n):
    """Returns a function which computes the amplitude for a component of a Gaussian state given the amplitudes of the
    components with lower weight.
    :param label_index: The index of the desired component in the list even_weight_bin
    :param n: the number of qubits of the system
    """
    label_bin = even_weight_bin[label_index]
    weight = int(hamming(label_bin, [0] * n) * n)
    if weight <= 2:
        def direct_formula(_parameters: np.ndarray, _label_values: np.ndarray):
            _label_values[label_index, :] = np.dot(_parameters[:, label_index, :], np.array([1, 1j]))
            return _label_values
        return direct_formula
    else:
        indices = np.where(label_bin == 1)[0]

        def recursive_formula(_parameters: np.ndarray, _label_values: np.ndarray):
            _label_values[label_index:label_index+1, :] = sum(
                (-1) ** (i + 1)
                * _label_values[
                    np.where(even_weight == read_binary_array(change_bits([0] * n, [indices[0], indices[i]])))[0],
                    :
                ]
                * _label_values[
                    np.where(even_weight == read_binary_array(change_bits(label_bin, [indices[0], indices[i]])))[0],
                    :
                ]
                for i in range(1, weight)
            ) / _label_values[0:1, :]
            return _label_values
        return recursive_formula


def _coefficient_formulae(n):
    """Returns a list of all coefficient formulae for a Gaussian state."""
    return np.array([_coefficient_formula(idx, n) for idx in range(len(even_weight_bin))])


def _grad_formula_zero(label_index, n):
    """Returns a function which computes the grad with respect to the zero component for a component of a Gaussian state
    given the amplitudes of all components and gradients of components with lower weight.
    :param label_index: The index of the desired component in the list even_weight_bin
    :param n: the number of qubits of the system
    """
    label_bin = even_weight_bin[label_index]
    weight = int(hamming(label_bin, [0] * n) * n)
    if weight == 0:
        def weight_0_zero_grad_formula(_label_values, _grad_values):
            chi = _grad_values.shape[-1]
            _grad_values[label_index, 0, :, :] = np.full((2, chi), np.array([[1], [1j]]))
            return _grad_values
        return weight_0_zero_grad_formula

    elif weight == 2:
        def weight_2_zero_grad_formula(_label_values, _grad_values):
            _grad_values[label_index, 0, ...] = 0
            return _grad_values
        return weight_2_zero_grad_formula

    else:
        indices = np.where(label_bin == 1)[0]

        def recursive_zero_grad_formula(_label_values, _grad_values):
            _grad_values[
                label_index, 0, ...
            ] = np.outer(
                np.array([[1], [1j]]),
                1 / _label_values[0:1, :] * (
                    - _label_values[label_index:label_index + 1, :]
                    + sum(
                        (-1) ** (i + 1)
                        * _label_values[
                            np.where(
                                even_weight == read_binary_array(change_bits([0] * n, [indices[0], indices[i]]))
                            )[0],
                            :
                        ]
                        * _grad_values[
                            np.where(
                                even_weight == read_binary_array(change_bits(label_bin, [indices[0], indices[i]]))
                            )[0],
                            0,
                            0,
                            :
                        ]
                        for i in range(1, weight)
                    )
                )
            )
            return _grad_values
        return recursive_zero_grad_formula


def _grad_formula_non_zero(label_index, grad_label_index, n):
    """Returns a function which computes the grad with respect to a non-zero component for a component of a Gaussian
    state given the amplitudes of all components.
    :param label_index: The index of the desired component in the list even_weight_bin
    :param grad_label_index: The index of the component to differentiate wrt in the list even_weight_bin
    :param n: the number of qubits of the system
    """
    label_bin = even_weight_bin[label_index]
    weight = int(hamming(label_bin, [0] * n) * n)
    if weight == 0:
        def weight_0_grad_formula(_label_values, _grad_values):
            _grad_values[label_index, grad_label_index, ...] = 0
            return _grad_values
        return weight_0_grad_formula

    elif weight == 2:
        def weight_2_grad_formula(_label_values, _grad_values: np.ndarray):
            chi = _grad_values.shape[-1]
            _grad_values[
                label_index,
                grad_label_index,
                ...
            ] = np.full((2, chi), np.array([[1], [1j]])) if label_index == grad_label_index else np.zeros((2, chi))
            return _grad_values
        return weight_2_grad_formula

    else:
        indices = np.where(label_bin == 1)[0]

        def recursive_grad_formula(_label_values, _grad_values):
            _grad_values[
                label_index,
                grad_label_index,
                ...
            ] = np.outer(
                np.array([[1], [1j]]),
                sum(
                    (-1) ** (i + 1)
                    * (
                        _grad_values[
                            np.where(
                                even_weight == read_binary_array(change_bits([0] * n, [indices[0], indices[i]]))
                            )[0],
                            grad_label_index,
                            0,
                            :
                        ]
                        * _label_values[
                            np.where(
                                even_weight == read_binary_array(change_bits(label_bin, [indices[0], indices[i]]))
                            )[0],
                            :
                        ]
                        + _label_values[
                            np.where(
                                even_weight == read_binary_array(change_bits([0] * n, [indices[0], indices[i]]))
                            )[0],
                            :
                        ]
                        * _grad_values[
                            np.where(
                                even_weight == read_binary_array(change_bits(label_bin, [indices[0], indices[i]]))
                            )[0],
                            grad_label_index,
                            0,
                            :
                        ]
                    )
                    for i in range(1, weight)
                ) / _label_values[0:1, :]
            )
            return _grad_values
        return recursive_grad_formula


def _grad_formula(label_index, grad_label_index, n):
    """Returns the appropriate grad formula"""
    grad_label_bin = even_weight_bin[grad_label_index]
    if not int(hamming(grad_label_bin, [0] * n) * n) <= 2:
        raise Exception('Invalid label to diff w.r.t in grad formula')
    if grad_label_index == 0:
        return _grad_formula_zero(label_index, n)
    else:
        return _grad_formula_non_zero(label_index, grad_label_index, n)


def _grad_formulae(n):
    """Returns an array of all grad formulae"""
    grad_indexes = [idx for idx in range(len(even_weight_bin)) if hamming(even_weight_bin[idx], [0] * n) * n <= 2]
    return np.array([[
        _grad_formula(label_idx, grad_label_idx, n)
        for label_idx in range(len(even_weight_bin))]
        for grad_label_idx in grad_indexes
    ])


def make_formulae(n):
    """Returns two arrays, containing formulae to compute the amplitudes and gradients of a Gaussian state
    given the amplitudes of the weight 0 and 2 components"""
    global even_weight_bin, even_weight
    even_weight_bin = np.array([
        item for sublist in
        [strings_with_weight(n, k) for k in range(0, n + 1, 2)]
        for item in sublist
    ])
    even_weight = np.array([read_binary_array(b) for b in even_weight_bin])
    coefficients = _coefficient_formulae(n)
    grads = _grad_formulae(n)
    return coefficients, grads


if __name__ == '__main__':
    c, g = make_formulae(8)
