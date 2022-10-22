import numpy as np
from random import randint


def mean_squared_error_gd(y, tx, initial_w, max_iters, gamma):
    """
    gradient descent using mse error
    arguments :
    y -- the true assignment
    tx -- array of features for each datapoint
    initial_w -- initial weights from which we start the GD
    max_iters -- the amount of iterations we perform
    gamma -- the GD step size 
    """
    w = initial_w
    N = tx.shape[0]

    for _ in range(max_iters):
        e = y - (tx @ w)
        grad = (-1 / N) * (tx.T @ e)
        w = w - (gamma * grad)

    e = y - (tx @ w)
    loss = 1 / 2 * (e**2).mean()

    return (w, loss)


def mean_squared_error_sgd(y, tx, initial_w, max_iters, gamma):
    """
    stochastic gradient descent using mse error
    arguments :
    y -- the true assignment
    tx -- array of features for each datapoint
    initial_w -- initial weights from which we start the SGD
    max_iters -- the amount of iterations we perform
    gamma -- the GD step size 
    """
    w = initial_w
    N = tx.shape[0]

    for _ in range(max_iters):
        index = randint(0, N - 1)
        xi = tx[index].reshape(1, -1)
        yi = y[index].reshape(1, -1)
        ei = yi - (xi @ w)
        grad = (-ei) * xi.T
        w = w - (gamma * grad)

    e = y - (tx @ w)
    loss = 1 / 2 * (e**2).mean()

    return (w, loss)


def least_squares(y, tx):
    """
    least squares linear regression solved with equations
    arguments :
    y -- the true assignment
    tx -- array of features for each datapoint
    """
    weights = np.linalg.inv(tx.T @ tx) @ tx.T @ y

    e = y - (tx @ weights)
    loss = 1 / 2 * (e**2).mean()

    return (weights, loss)


def ridge_regression(y, tx, lambda_):
    """
    ridge regression solved with equations
    arguments :
    y -- the true assignment
    tx -- array of features for each datapoint
    lambda_ -- the penalization term
    """
    N, O = tx.shape
    ridge_matrix = (2 * N * lambda_) * np.eye(O)
    weights = np.linalg.inv(tx.T @ tx + ridge_matrix) @ tx.T @ y

    e = y - (tx @ weights)
    loss = 1 / 2 * (e**2).mean()

    return (weights, loss)


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def logistic_regression(y, tx, initial_w, max_iters, gamma):
    """
    logistic regression solved with GD
    arguments :
    y -- the true assignment
    tx -- array of features for each datapoint
    initial_w -- initial weights from which we start the GD
    max_iters -- the amount of iterations we perform
    gamma -- the step size
    """
    w = initial_w
    N = tx.shape[0]

    for _ in range(max_iters):
        e = y - sigmoid(tx @ w)
        grad = (-1 / N) * (tx.T @ e)
        w = w - (gamma * grad)

    pred = sigmoid(tx @ w)
    loss = (-y * np.log(pred) - (1 - y) * np.log(1 - pred)).mean()
    return (w, loss)


def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    """
    regularized logistic regression solved with GD
    arguments :
    y -- the true assignment
    tx -- array of features for each datapoint
    lambda_ -- the penalization term
    initial_w -- initial weights from which we start the GD
    max_iters -- the amount of iterations we perform
    gamma -- the step size
    """
    w = initial_w
    N = tx.shape[0]

    for _ in range(max_iters):
        e = y - sigmoid(tx @ w)
        grad = (-1 / N) * (tx.T @ e) + 2 * lambda_ * w
        w = w - (gamma * grad)

    pred = sigmoid(tx.dot(w))
    loss = (
        -y * np.log(pred) - (1 - y) * np.log(1 - pred)
    ).mean() + lambda_ * np.linalg.norm(w) ** 2
    return (w, loss)
