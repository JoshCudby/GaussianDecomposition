import numpy as np
from operator import mul
from functools import reduce
from itertools import product

from rank_decomposition.utils.binary_string_utils import read_binary_array, strings_with_weight


def _make_magic_from_labels_and_amplitudes(n, mlb, ma):
    """
    Given a list of indexes (in binary) and corresponding amplitudes, return an array representing the
    magic state
    """
    even_weight_bin = np.array([
        item for sublist in
        [strings_with_weight(n, k) for k in range(0, n + 1, 2)]
        for item in sublist
    ])
    even_weight_labels = np.array([read_binary_array(b) for b in even_weight_bin])

    ml = np.array([read_binary_array(b) for b in mlb])
    mi = [np.where(even_weight_labels == label)[0] for label in ml]
    magic_amplitudes = np.zeros(2 ** (n - 1))
    for index, amplitude in zip(mi, ma):
        magic_amplitudes[index] = amplitude
    return magic_amplitudes


def new_magic(n):
    """The magic state ket{tilde{M}} in the paper"""
    labels_bin = [
        sum(x, [])
        for x in product([[0, 0, 0, 0], [0, 0, 1, 1], [1, 1, 0, 0], [1, 1, 1, 1]], repeat=int(n / 4))
    ]
    amplitudes = [
        reduce(mul, x) for x in
        product([(3 / 8) ** 0.5, (2 / 8) ** 0.5, (2 / 8) ** 0.5, (1 / 8) ** 0.5], repeat=int(n / 4))
    ]
    return _make_magic_from_labels_and_amplitudes(n, labels_bin, amplitudes)


def default_magic(n):
    """The magic state ket{M} in the paper"""
    labels_bin = [
        sum(x, [])
        for x in product([[0, 0, 0, 0], [1, 1, 1, 1]], repeat=int(n / 4))
    ]
    amplitudes = [
        reduce(mul, x) for x in
        product([(2 / 4) ** 0.5, (2 / 4) ** 0.5], repeat=int(n / 4))
    ]
    return _make_magic_from_labels_and_amplitudes(n, labels_bin, amplitudes)
