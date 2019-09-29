import csv
import numpy as np


class YahooFinanceData:

    def __init__(self):
        self.data = []
        self.index = 0

    def open_data(self, filename):
        with open(filename) as csv_file:
            csv_reader = csv.DictReader(csv_file, delimiter=',')
            for row in csv_reader:
                self.data.append(np.array([[float(row['Adj Close'])]]))
            self.index = 0

    def has_more_data(self):
        return self.index < len(self.data)

    def next_measurement(self):
        self.index += 1
        return self.data[self.index - 1]
