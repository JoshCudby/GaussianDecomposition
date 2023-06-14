import pickle

import numpy as np
from time import time
from scipy.optimize import dual_annealing, minimize
from scipy.special import comb

from rank_decomposition.utils.gaussian_formulae import make_formulae
from rank_decomposition.utils.logging_utils import get_formatted_logger
from rank_decomposition.utils.magic_states import new_magic, default_magic

logger = get_formatted_logger(__name__)
global n, chi, num_even_weight, num_weight_less_4, coefficient_formulae, grad_formulae, magic_amplitudes


def random_initial_guess():
    width = 1
    positive_bias = 0.25
    return np.random.default_rng().random(chi * num_weight_less_4 * 2) * width - (width / 2 - positive_bias)


def compute_label_values(params):
    """Compute the values of amplitudes given the amplitudes of weight 0 and 2 components. Due to the recursive nature
    of the formulae, computes one at a time."""
    label_values = np.zeros((num_even_weight, chi), dtype=complex)
    for i in range(len(coefficient_formulae)):
        label_values = coefficient_formulae[i](params, label_values)
    return label_values


def compute_grad_values(label_values):
    """Compute the gradient values given the amplitudes of all components. Due to the recursive nature
    of the formulae, computes one at a time."""
    grad_values = np.full((num_even_weight, num_weight_less_4, 2, chi), np.inf * (1 + 1j), dtype=complex)
    for d in range(num_weight_less_4):
        for index in range(num_even_weight):
            grad_values = grad_formulae[d, index](label_values, grad_values)
    return grad_values


def alternating_real_imag_parts(values):
    """Returns a new array with twice as many rows, which contains in alternating rows the real and imaginary parts of
    the original array"""
    ret = np.zeros(2 * values.shape[0]) if len(values.shape) == 1 \
        else np.zeros((2 * values.shape[0], values.shape[1]))
    ret[0::2, ...] = np.real(values)
    ret[1::2, ...] = np.imag(values)
    return ret


def residuals(params):
    """Compute the difference in each component of the trial decomposition and the target magic state"""
    label_values = compute_label_values(params.reshape((chi, num_weight_less_4, 2)))
    summed_labels = np.sum(label_values, axis=1)
    return alternating_real_imag_parts(summed_labels - magic_amplitudes)


def jacobian(params):
    """The Jacobian matrix for the residual function above"""
    params = params.reshape((chi, num_weight_less_4, 2))
    label_values = compute_label_values(params)
    grad_values = compute_grad_values(label_values)

    J = np.zeros((2 * num_even_weight, 2 * chi * num_weight_less_4))
    for i in range(chi):
        J[:, 2 * num_weight_less_4 * i: 2 * num_weight_less_4 * (i + 1)] = alternating_real_imag_parts(
            grad_values[..., i].reshape(num_even_weight, 2 * num_weight_less_4)
        )
    return J


def loss_function(params):
    """Norm squared of the residuals"""
    return np.linalg.norm(residuals(params)) ** 2


def loss_function_grad(params):
    """Gradient of the loss function"""
    return 2 * np.dot(residuals(params), jacobian(params))


def setup(copies, magic_state):
    """Initial code to set up various parameter values"""
    global n, chi, num_even_weight, num_weight_less_4, coefficient_formulae, grad_formulae, magic_amplitudes

    n = 4 * copies
    chi = (2 ** copies) - 1
    num_even_weight = 2 ** (n - 1)
    num_weight_less_4 = int(comb(n, 2)) + 1
    coefficient_formulae, grad_formulae = make_formulae(n)

    if magic_state == 'new':
        logger.info('New magic state')
        magic_amplitudes = new_magic(n)
    else:
        logger.info('Default magic state')
        magic_amplitudes = default_magic(n)


def local_minimize(copies, magic_state, save_file_name, local_search_iter):
    """Use the SciPy minimize function to run the L-BFGS-B algorithm, a local minimizer"""
    setup(copies, magic_state)
    x0 = random_initial_guess()
    logger.info('Local Minimize')
    bounds = list(zip([-10] * x0.size, [10] * x0.size))
    opt_result = minimize(
        loss_function,
        x0,
        method='L-BFGS-B',
        jac=loss_function_grad,
        options={'disp': True, 'maxiter': local_search_iter},
        bounds=bounds,
        tol=10**-10
    )
    opt_result['magic_state_type'] = magic_state
    opt_result['n'] = n
    with open(save_file_name, 'wb') as file:
        pickle.dump(opt_result, file, pickle.HIGHEST_PROTOCOL)
    return opt_result


def main(copies, magic_state, save_file_name, local_search_iter, global_search_iter):
    """Use the SciPy dual_annealing function to run a global minimizer"""
    setup(copies, magic_state)

    x0 = random_initial_guess()
    logger.info('Dual Annealing')
    bounds = list(zip([-10] * x0.size, [10] * x0.size))

    def annealing_callback(x, f, context):
        logger.info('Minimum found')
        logger.info(f'Function value: {f}')
        logger.info(f'Context: {context}')
        if (context == 1 or context == 0 and f < 0.5) and save_file_name is not None:
            with open(save_file_name, 'wb') as file:
                pickle.dump(x, file, pickle.HIGHEST_PROTOCOL)
        return False

    opt_result = dual_annealing(
        loss_function,
        bounds=bounds,
        minimizer_kwargs={
            'jac': loss_function_grad,
            'method': 'L-BFGS-B',
            'options': {'disp': True, 'maxiter': local_search_iter},
            'bounds': bounds
        },
        maxiter=global_search_iter,
        x0=x0,
        callback=annealing_callback,
        no_local_search=False,
        maxfun=200000
    )
    opt_result['magic_state_type'] = magic_state
    opt_result['n'] = n
    return opt_result


if __name__ == '__main__':
    out_global = main(
        copies=2,
        magic_state='default',
        save_file_name=f'data/global_minimize_{round(time())}.pkl',
        local_search_iter=400,
        global_search_iter=1000
    )
    out_local = local_minimize(
        copies=2,
        magic_state='default',
        save_file_name=f'data/local_minimize_{round(time())}.pkl',
        local_search_iter=750
    )
