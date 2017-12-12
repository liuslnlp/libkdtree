from kdtree import KNeighborsRegressor
from sklearn.neighbors import KNeighborsRegressor as SKRegressor
import numpy as np
import time
import sys


def main():
    np.random.seed(19260817)
    X = np.random.rand(60000, 250)
    y = np.random.rand(60000)

    skstart = time.clock()
    skclf = SKRegressor(n_neighbors=5)
    skclf.fit(X, y)
    t = skclf.predict(X[:100])
    skend = time.clock()

    start = time.clock()
    clf = KNeighborsRegressor(k=5)
    clf.fit(X, y)
    t = clf.predict(X[:100])    
    end = time.clock()

    print("SKlearn: %.6f" % (skend - skstart))
    print("libkdtree: %.6f" %(end - start))

if __name__ == '__main__':
    main()