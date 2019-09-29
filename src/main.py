import src.kalman_filter as kalman_filter
import src.yahoo_financedata as yahoo_financedata
import numpy as np
import matplotlib.pyplot as plt


def main():

    # read data
    data_filename = '../data/IFNNY.csv'
    dataReader = yahoo_financedata.YahooFinanceData()
    dataReader.open_data(data_filename)

    # init kalman filter
    x = np.full((2, 1), 2)
    P = np.full((2, 2), 2 ** 2)
    state = kalman_filter.State(x, P)

    Q = np.full((2, 2), 1 ** 2)
    F = np.array([[1, 1],
                  [0, 1]])
    R = np.array([[0.5 ** 2]])
    H = np.array([[1, 0]])

    # data for plotting
    gt = []
    result = []

    counter = 0

    while dataReader.has_more_data():
        z = dataReader.next_measurement()

        if counter == 0:
            x = np.array([[z[0][0]], [0]])
            state = kalman_filter.State(x, P)

        prior = kalman_filter.predict(state, F, Q)
        posterior = kalman_filter.update(prior, z, R, H)
        state = posterior

        gt.append(z[0][0])
        result.append(prior.X[0][0])

        counter += 1
        if counter > 20:
            break

    plt.plot(gt)
    plt.plot(result)

    plt.legend(['Actual stock price', 'predicted stock price'])

    plt.grid()
    plt.show()


if __name__ == '__main__':
    main()
