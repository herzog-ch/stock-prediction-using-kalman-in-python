import numpy as np
from collections import namedtuple

State = namedtuple('State', 'X, P')


def predict(state, F, Q):
    """Perform the predict step

    x_pred = Fx
    P_pred = F P F^T + Q

    :param state: State namedtuple
    :param F: Transition matrix
    :param Q: Process Covariance
    :return: The prior as a State namedtuple
    """

    assert state.X.shape[0] == F.shape[1]
    assert state.X.shape[1] == 1
    assert F.shape[0] == F.shape[1]
    assert Q.shape[0] == Q.shape[1]
    assert Q.shape[0] == F.shape[0]

    x_pred = np.matmul(F, state.X)
    p_pred = np.matmul(F, np.matmul(state.P, F.T)) + Q
    return State(x_pred, p_pred)


def update(prior, z, R, H):
    """Perform update step

    S = H P_prior H^T + R
    K = P_prior H^T S^-1
    y = z - H x_prior
    x = x_prior + Ky
    P = (I - KH) P_prior

    :param prior: State namedtuple holding the prior mean and covariance
    :param z: measurement vector
    :param R: measurement covariance matrix
    :param H: measurement matrix
    :return: Returns the posterior mean and covariance as State namedtuple
    """

    assert prior.X.shape[1] == 1
    assert prior.X.shape[0] == H.shape[1]
    assert H.shape[0] == z.shape[0]
    assert z.shape[1] == 1
    assert prior.P.shape[0] == prior.P.shape[1]
    assert prior.P.shape[1] == H.shape[1]

    z_pred = np.matmul(H, prior.X)
    y = z - z_pred
    S = np.matmul(H, np.matmul(prior.P, H.T)) + R
    K = np.matmul(prior.P, np.matmul(H.T, np.linalg.inv(S)))
    x_posterior = prior.X + np.matmul(K, y)
    p_posterior = np.matmul((np.identity(prior.P.shape[0]) - np.matmul(K, H)), prior.P)
    return State(x_posterior, p_posterior)
