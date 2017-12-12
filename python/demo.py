import kdtree as kt
import matplotlib.pyplot as plt
import numpy as np

def plot_decision_boundary(pred_func, X, y, title=None):
    """Plot the descision boundary
    Parameters
    ----------
    pred_func: predict function
    X: train set, shape=(n_samples, 2)
    y: labels, shape=(n_samples, )
    """
    # Set min and max values and give it some padding
    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    h = 0.01
    # Generate a grid of points with distance h between them
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    # Predict the function value for the whole gid
    Z = pred_func(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    # Plot the contour and training examples
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)
    plt.scatter(X[:, 0], X[:, 1], s=40, c=y, cmap=plt.cm.Spectral)

    if title:
        plt.title(title)
    plt.show()

def main():
    import sklearn.datasets
    X, y = sklearn.datasets.make_moons(200, noise=0.20)
    y = 2 * y - 1
    clf = kt.KNeighborsClassifier(3)
    clf.fit(X, y)
    plot_decision_boundary(clf.predict, X, y, 'K Neighbors Classifier')

if __name__ == '__main__':
    main()