# Pattern Recognition
# Classification of Iris Dataset
import numpy as np
import matplotlib.pyplot as plt
import os
import urllib.request

# Load Iris
class Irisdataset:
    feature_names = ['Sepal length', 'Sepal width', 'Petal length', 'Petal width']
    set_names = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']

    # trn: number of traning examples
    # 150-trn: number of test examples
    def __init__(self, trn=120):
        # 0<=TR_data<=150
        N = trn
        # Testing samples 150-N
        self._AR = np.zeros((150, 5))

        if not os.path.isfile('iris.data'):
            print('Download IRIS dataset')
            urllib.request.urlretrieve(
                "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data",
                "iris.data"
            )
            print('Download complete')

        with open('iris.data', 'r') as f:
            i = 0
            for Line in f:
                line = Line[:-1].split(',')
                line[0] = float(line[0])
                line[1] = float(line[1])
                line[2] = float(line[2])
                line[3] = float(line[3])
                if(line[4] == 'Iris-setosa'):
                    line[4] = 0.0
                elif(line[4] == 'Iris-versicolor'):
                    line[4] = 1.0
                elif(line[4] == 'Iris-virginica'):
                    line[4] = 2.0

                self._AR[i] = np.array(line)
                i += 1
                if i == 150:
                    break
        self._AR = np.random.permutation(self._AR) # random sort
        self.TrainData = self._AR[:N]
        self.TestData = self._AR[N:]
