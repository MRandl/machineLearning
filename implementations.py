def mean_squared_error_gd(y, tx, initial_w, max_iters, gamma):
  w = initial_w
  N = tx.shape[0]

  for _ in range(max_iters):
    e = y - (tx @ w)
    grad = (-1/N) * (tx.T @ e)
    w = w - (gamma * grad)

  e = y - (tx @ w)
  loss = (e ** 2).mean()

  return (w, loss)
  

def mean_squared_error_sgd(y, tx, initial_w, max_iters, gamma):
  w = initial_w
  N = tx.shape[0]

  for _ in range(max_iters):
    index = randint(0, N - 1)
    xi = tx[index]
    yi = y[index]
    ei = yi - (xi @ w)
    grad = (-ei) * xi.T
    w = w - (gamma * grad)

  e = y - (tx @ w)
  loss = (e ** 2).mean()

  return (w, loss)


def least_squares(y, tx):
  weights = np.linalg.inv(tx.T @ tx) @ tx.T @ y
  
  e = y - (tx @ weights)
  loss = (e ** 2).mean()

  return (weights, loss)


def ridge_regression(y, tx, lambda_):
  N, O = tx.shape
  ridge_matrix = (2*N*lambda_) * np.eye(O)
  weights = np.linalg.inv(tx.T @ tx + ridge_matrix) @ tx.T @ y
  
  e = y - (tx @ weights)
  loss = (e ** 2).mean()

  return (weights, loss)