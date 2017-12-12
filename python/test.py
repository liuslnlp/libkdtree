import kdtree as kt
import numpy as np

def test_kd_tree(n_samples, n_features=5):
    X = np.random.rand(n_samples, n_features).astype(np.float64)
    y = np.arange(n_samples)
    X_test = np.random.rand(1, n_features)
    dists = np.sum((X - X_test)**2, axis=1)
    dists = np.sqrt(dists)
    arg = np.argmin(dists)
    clf = kt.KNeighborsRegressor(k=1)
    clf.fit(X, y)
    pred = clf.predict(X_test).astype(np.int)[0]
    assert arg == pred


if __name__ == '__main__':
    for _ in range(500):
        test_kd_tree(200, 10)
    print("Success!")