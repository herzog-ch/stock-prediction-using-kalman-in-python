import src.kalman_filter as kalman_filter
import numpy as np


def main():

    x = np.full((2, 1), 2)
    P = np.zeros((2, 2))
    Q = np.full((2, 2), 0)

    F = np.array([[2, 0],
                  [0, 1]])

    state = kalman_filter.State(x, P)
    prior = kalman_filter.predict(state, F, Q)


if __name__ == '__main__':
    main()
