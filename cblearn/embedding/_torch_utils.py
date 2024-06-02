from typing import Sequence, Callable, Optional

import numpy as np
import scipy


def torch_device(device: str) -> str:
    import torch

    if device == "auto":
        if torch.cuda.is_available():
            return "cuda"
        else:
            return "cpu"


def torch_minimize_kernel(method, objective, init, **kwargs):
    import torch

    kernel_init = init @ init.T * .1
    dim = init.shape[1]

    def kernel_objective(K, *args, **kwargs):
        with torch.no_grad():
            D, U = torch.linalg.eigh(K)
            D = D[-dim:].clamp(min=0)
            K[:] = U[:, -dim:].mm(D.diag()).mm(U[:, -dim:].transpose(0, 1))
        return objective(K, *args, **kwargs)

    result = torch_minimize(method, kernel_objective, kernel_init, **kwargs)
    U, s, _ = np.linalg.svd(result.x)

    result.x = U[:, :dim] @ np.sqrt(np.diag(s[:dim]))
    return result


def torch_minimize(method,
                   objective: Callable, init: np.ndarray, data: Sequence, args: Sequence = [],
                   seed: Optional[int] = None, device: str = 'auto', max_iter: int = 100,
                   batch_size=50_000, shuffle=True, **kwargs) -> 'scipy.optimize.OptimizeResult':
    """ Pytorch minimization routine using L-BFGS.

        This function is aims to be a pytorch version of :func:`scipy.optimize.minimize`.

        Args:
            objective:
                Loss function to minimize.
                Function argument is the current parameters and optional additional argument.
            init:
                The initial parameter values.
            data:
                Sequence of data arrays.
            args:
                Sequence of additional arguments, passed to the objective.
            seed:
                Manual seed of randomness in data sampling. Use to preserve reproducability.
            device:
                Device to run the minimization on, usually "cpu" or "cuda".
                "auto" uses "cuda", if available.
            max_iter:
                The maximum number of optimizer iteration (also called epochs).
        Returns:
            Dict-like object containing status and result information
            of the optimization.
    """
    import torch

    device = torch_device(device)

    data = [torch.tensor(d).to(device) for d in data]
    args = [torch.tensor(a).to(device) for a in args]
    factr = 1e7 * np.finfo(float).eps
    g = torch.Generator()
    if seed is not None:
        g.manual_seed(seed)
    dataloader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(*data),
                                             batch_size=batch_size, shuffle=shuffle, generator=g)

    method = method.lower()
    optimizers = {
        'l-bfgs-b': torch.optim.LBFGS,
        'adam': torch.optim.Adam,
    }
    X = torch.tensor(init, requires_grad=True, device=device)
    if method.lower() in optimizers:
        optimizer = optimizers[method.lower()]([X], **kwargs)
    else:
        raise ValueError(f"Expects method in {optimizers.keys()}, got {method}.")

    loss = float("inf")
    success, message = True, ""
    for n_iter in range(max_iter):
        prev_loss = loss
        loss = 0
        for batch in dataloader:
            def closure():
                optimizer.zero_grad()
                loss = objective(X, *batch, *args)
                loss.backward()
                return loss

            step_loss = optimizer.step(closure)
            loss += len(batch) * step_loss / len(dataloader.dataset)

        if abs(prev_loss - loss) / max(abs(loss), abs(prev_loss), 1) < factr:
            break
    else:
        success = False
        message = f"{method} did not converge."

    return scipy.optimize.OptimizeResult(
        x=X.cpu().detach().numpy(), fun=loss.cpu().detach().numpy(),
        nit=n_iter + 1, success=success, message=message)

def torch_hard_threshold_minimize(method,
                   objective: Callable, init: np.ndarray, data: Sequence, args: Sequence = [],
                   seed: Optional[int] = None, device: str = 'auto', max_iter: int = 100,
                   batch_size=50_000, shuffle=True, beta=None, **kwargs) -> 'scipy.optimize.OptimizeResult':
    """ Pytorch minimization routine using L-BFGS.

        This function is aims to be a pytorch version of :func:`scipy.optimize.minimize`.

        Args:
            objective:
                Loss function to minimize.
                Function argument is the current parameters and optional additional argument.
            init:
                The initial parameter values.
            data:
                Sequence of data arrays.
            args:
                Sequence of additional arguments, passed to the objective.
            seed:
                Manual seed of randomness in data sampling. Use to preserve reproducability.
            device:
                Device to run the minimization on, usually "cpu" or "cuda".
                "auto" uses "cuda", if available.
            max_iter:
                The maximum number of optimizer iteration (also called epochs).
        Returns:
            Dict-like object containing status and result information
            of the optimization.
    """
    import torch

    device = torch_device(device)

    data = [torch.tensor(d).to(device) for d in data]
    args = [torch.tensor(a).to(device) for a in args]
    factr = 1e7 * np.finfo(float).eps
    g = torch.Generator()
    if seed is not None:
        g.manual_seed(seed)
    dataloader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(*data),
                                             batch_size=batch_size, shuffle=shuffle, generator=g)

    method = method.lower()
    optimizers = {
        'l-bfgs-b': torch.optim.LBFGS,
        'adam': torch.optim.Adam,
    }
    X = torch.tensor(init, requires_grad=True, device=device)
    if method.lower() in optimizers:
        optimizer = optimizers[method.lower()]([X], **kwargs)
    else:
        raise ValueError(f"Expects method in {optimizers.keys()}, got {method}.")

    loss = float("inf")
    success, message = True, ""
    for n_iter in range(max_iter):
        prev_loss = loss
        loss = 0
        for batch in dataloader:
            def closure():
                optimizer.zero_grad()
                loss = objective(X, *batch, *args)
                loss.backward()
                return loss

            step_loss = optimizer.step(closure)
            loss += len(batch) * step_loss / len(dataloader.dataset)

        if abs(prev_loss - loss) / max(abs(loss), abs(prev_loss), 1) < factr:
            break
    else:
        success = False
        message = f"{method} did not converge."

    # Thresholding the matrix (not sure if it is already thresholded)
    #x = X.cpu().detach().numpy()
    def omega_approx(b):
        return 0.56*(b**3) - 0.95*(b**2) + 1.82*b + 1.43
    U, s, Vh = torch.linalg.svd(X.cuda(), driver='gesvd', full_matrices=False)
    omega = omega_approx(beta)
    svd_threshold = torch.median(s) * omega
    thresholded_s = torch.where(s > svd_threshold, s, torch.zeros_like(s))
    # add a small value to avoid numerical instability
    #thresholded_s += 1e-8
    low_rank_X = U @ torch.diag(thresholded_s) @ Vh
    low_rank_X = low_rank_X.cpu().detach().numpy()
    #s = s.cpu().detach().numpy()
    #print(f"s = {s}")
    #print(f"low_rank_X = {low_rank_X}")

    return scipy.optimize.OptimizeResult(
        x=low_rank_X, fun=loss.cpu().detach().numpy(),
        nit=n_iter + 1, success=success, message=message)

def torch_svt_minimize(method,
                   objective: Callable, init: np.ndarray, data: Sequence, args: Sequence = [],
                   seed: Optional[int] = None, device: str = 'auto', max_iter: int = 100,
                   batch_size=50_000, shuffle=True, **kwargs) -> 'scipy.optimize.OptimizeResult':
    """ Pytorch minimization routine using L-BFGS.

        This function is aims to be a pytorch version of :func:`scipy.optimize.minimize`.

        Args:
            objective:
                Loss function to minimize.
                Function argument is the current parameters and optional additional argument.
            init:
                The initial parameter values.
            data:
                Sequence of data arrays.
            args:
                Sequence of additional arguments, passed to the objective.
            seed:
                Manual seed of randomness in data sampling. Use to preserve reproducability.
            device:
                Device to run the minimization on, usually "cpu" or "cuda".
                "auto" uses "cuda", if available.
            max_iter:
                The maximum number of optimizer iteration (also called epochs).
        Returns:
            Dict-like object containing status and result information
            of the optimization.
    """
    import torch

    device = torch_device(device)

    data = [torch.tensor(d).to(device) for d in data]
    args = [torch.tensor(a).to(device) for a in args]
    factr = 1e7 * np.finfo(float).eps
    g = torch.Generator()
    if seed is not None:
        g.manual_seed(seed)
    dataloader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(*data),
                                             batch_size=batch_size, shuffle=shuffle, generator=g)

    method = method.lower()
    optimizers = {
        'l-bfgs-b': torch.optim.LBFGS,
        'adam': torch.optim.Adam,
    }
    X = torch.tensor(init, requires_grad=True, device=device)
    if method.lower() in optimizers:
        optimizer = optimizers[method.lower()]([X], **kwargs)
    else:
        raise ValueError(f"Expects method in {optimizers.keys()}, got {method}.")

    loss = float("inf")
    success, message = True, ""
    for n_iter in range(max_iter):
        prev_loss = loss
        loss = 0
        for batch in dataloader:
            def closure():
                optimizer.zero_grad()
                loss = objective(X, *batch, *args)
                loss.backward()
                return loss

            step_loss = optimizer.step(closure)
            loss += len(batch) * step_loss / len(dataloader.dataset)

        if abs(prev_loss - loss) / max(abs(loss), abs(prev_loss), 1) < factr:
            break
    else:
        success = False
        message = f"{method} did not converge."

    # Thresholding the matrix (not sure if it is already thresholded)
    #x = X.cpu().detach().numpy()
    U, s, Vh = torch.linalg.svd(X.cuda(), driver='gesvd', full_matrices=False)
    threshold = args[-1]
    thresholded_s = torch.where(s > threshold, s, torch.zeros_like(s))
    # add a small value to avoid numerical instability
    thresholded_s += 1e-8
    low_rank_X = U @ torch.diag(thresholded_s) @ Vh
    low_rank_X = low_rank_X.cpu().detach().numpy()
    s = s.cpu().detach().numpy()
    #print(f"s = {s}")
    #print(f"low_rank_X = {low_rank_X}")

    return scipy.optimize.OptimizeResult(
        x=low_rank_X, fun=loss.cpu().detach().numpy(),
        nit=n_iter + 1, success=success, message=message)

def torch_qr_minimize(method,
                   objective: Callable, init: np.ndarray, data: Sequence, args: Sequence = [],
                   seed: Optional[int] = None, device: str = 'auto', max_iter: int = 100,
                   batch_size=50_000, shuffle=True, **kwargs) -> 'scipy.optimize.OptimizeResult':
    """ Pytorch minimization routine using L-BFGS.

        This function is aims to be a pytorch version of :func:`scipy.optimize.minimize`.

        Args:
            objective:
                Loss function to minimize.
                Function argument is the current parameters and optional additional argument.
            init:
                The initial parameter values.
            data:
                Sequence of data arrays.
            args:
                Sequence of additional arguments, passed to the objective.
            seed:
                Manual seed of randomness in data sampling. Use to preserve reproducability.
            device:
                Device to run the minimization on, usually "cpu" or "cuda".
                "auto" uses "cuda", if available.
            max_iter:
                The maximum number of optimizer iteration (also called epochs).
        Returns:
            Dict-like object containing status and result information
            of the optimization.
    """
    import torch

    device = torch_device(device)

    data = [torch.tensor(d).to(device) for d in data]
    args = [torch.tensor(a).to(device) for a in args]
    factr = 1e7 * np.finfo(float).eps
    g = torch.Generator()
    if seed is not None:
        g.manual_seed(seed)
    dataloader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(*data),
                                             batch_size=batch_size, shuffle=shuffle, generator=g)

    method = method.lower()
    optimizers = {
        'l-bfgs-b': torch.optim.LBFGS,
        'adam': torch.optim.Adam,
    }
    X = torch.tensor(init, requires_grad=True, device=device)
    if method.lower() in optimizers:
        optimizer = optimizers[method.lower()]([X], **kwargs)
    else:
        raise ValueError(f"Expects method in {optimizers.keys()}, got {method}.")

    loss = float("inf")
    success, message = True, ""
    for n_iter in range(max_iter):
        prev_loss = loss
        loss = 0
        for batch in dataloader:
            def closure():
                optimizer.zero_grad()
                loss = objective(X, *batch, *args)
                loss.backward()
                return loss

            step_loss = optimizer.step(closure)
            loss += len(batch) * step_loss / len(dataloader.dataset)

        if abs(prev_loss - loss) / max(abs(loss), abs(prev_loss), 1) < factr:
            break
    else:
        success = False
        message = f"{method} did not converge."

    # Thresholding the matrix (not sure if it is already thresholded)
    #X = X.cpu().detach().numpy()

    Q, R = torch.linalg.qr(X.cuda())
    # Calculate relative threshold
    R_norm = torch.linalg.norm(R, ord='fro')
    # t_val = torch.min(args[-1], torch.Tensor(0.2))
    # FIXME: This is a hack. Need to find a better way to ensure threshold doesnt break
    relative_threshold = args[-1] * R_norm
    diag_abs = torch.abs(torch.diagonal(R, 0))
    k = torch.sum(diag_abs > relative_threshold)
    if k>1:
        # Truncate Q and R
        Q_trunc = Q[:, :k]
        R_trunc = R[:k, :]
    else:
        # All elements become zero
        Q_trunc = Q[:, :1]
        R_trunc = R[:1, :]
    low_rank_X = Q_trunc @ R_trunc
    low_rank_X = low_rank_X.cpu().detach().numpy()

    return scipy.optimize.OptimizeResult(
        x=low_rank_X, fun=loss.cpu().detach().numpy(),
        nit=n_iter + 1, success=success, message=message)


import torch
import numpy as np
import scipy.optimize
from typing import Callable, Sequence, Optional

def torch_mf_minimize(method,
                      objective: Callable, init: np.ndarray, data: Sequence, args: Sequence = [],
                      seed: Optional[int] = None, device: str = 'auto', max_iter: int = 100,
                      batch_size=50_000, shuffle=True, **kwargs) -> 'scipy.optimize.OptimizeResult':
    """ PyTorch matrix factorization using alternating minimization.

    This function aims to minimize the objective function using alternating minimization,
    decomposing the initial parameter values (init) into LR^T.

    Args:
        method:
            Optimization method, 'l-bfgs-b' or 'adam'.
        objective:
            Loss function to minimize.
            Function arguments are the current parameters (L, R) and optional additional arguments.
        init:
            The initial parameter values as X, which will be decomposed into LR^T.
        data:
            Sequence of data arrays.
        args:
            Sequence of additional arguments, passed to the objective.
        seed:
            Manual seed of randomness in data sampling. Use to preserve reproducibility.
        device:
            Device to run the minimization on, usually "cpu" or "cuda".
            "auto" uses "cuda", if available.
        max_iter:
            The maximum number of optimizer iterations (epochs).
    Returns:
        Dict-like object containing status and result information
        of the optimization.
    """
    import torch

    device = torch_device(device)

    data = [torch.tensor(d).to(device) for d in data]
    args = [torch.tensor(a).to(device) for a in args]
    factr = 1e7 * np.finfo(float).eps
    g = torch.Generator()
    if seed is not None:
        g.manual_seed(seed)
    dataloader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(*data),
                                             batch_size=batch_size, shuffle=shuffle, generator=g)

    method = method.lower()
    optimizers = {
        'l-bfgs-b': torch.optim.LBFGS,
        'adam': torch.optim.Adam,
    }
    X = torch.tensor(init, requires_grad=False, device=device)
    
    # Decompose X into L and R using SVD
    U, S, V = torch.svd_lowrank(X, q=X.shape[1])
    L = torch.matmul(U, torch.diag(S.sqrt())).requires_grad_()
    R = torch.matmul(torch.diag(S.sqrt()), V.t()).T.requires_grad_()
    
    optimizer_L = optimizers[method.lower()]([L], **kwargs)
    optimizer_R = optimizers[method.lower()]([R], **kwargs)

    loss = float("inf")
    success, message = True, ""
    for n_iter in range(max_iter):
        prev_loss = loss
        loss = 0
        for batch in dataloader:
            def closure_L():
                optimizer_L.zero_grad()
                loss_L = objective(L, R, *batch, *args)
                loss_L.backward()
                return loss_L

            def closure_R():
                optimizer_R.zero_grad()
                loss_R = objective(L, R, *batch, *args)
                loss_R.backward()
                return loss_R

            loss += len(batch) * optimizer_L.step(closure_L) / len(dataloader.dataset)
            loss += len(batch) * optimizer_R.step(closure_R) / len(dataloader.dataset)

        if abs(prev_loss - loss) / max(abs(loss), abs(prev_loss), 1) < factr:
            break
    else:
        success = False
        message = "Alternating minimization did not converge."

    # Return LR^T only
    result_LR = torch.matmul(L, R.T)
    
    return scipy.optimize.OptimizeResult(
        x=result_LR.cpu().detach().numpy(), 
        fun=loss.cpu().detach().numpy(),
        nit=n_iter + 1, success=success, message=message)



def assert_torch_is_available() -> None:
    """ Check if the torch module is installed and raise error otherwise.

        The error message contains instructions how to install the pytorch package.

        Raises:
            ModuleNotFoundError: If torch package is not found.
    """
    try:
        import torch  # noqa: F401  We do not use torch here on purpose
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError(f"{e}.\n\n"
                                  "This function of cblearn requires the installation of the pytorch package.\n"
                                  "To install pytorch, visit https://pytorch.org/get-started/locally/\n"
                                  "or run either of the following commands:\n"
                                  "    pip install cblearn[torch]\n"
                                  "    pip install torch\n"
                                  "    conda install -c=conda-forge pytorch\n")
