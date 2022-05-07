#################################
# Your name: Omri Elad
# ID: 204620702
# Python Version: 3.9.11
#################################

from __future__ import annotations
from typing import Union
import numpy as np
import numpy.random
from numpy.random import randint
from sklearn.datasets import fetch_openml
import sklearn.preprocessing
from dataclasses import dataclass
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)
"""
Please use the provided function signature for the SGD implementation.
Feel free to add functions and other code, and submit this file with the name sgd.py
"""
EtaChar = "\u03B7"
VECTOR_SIZE = 784


def helper():
    mnist = fetch_openml('mnist_784', as_frame=False)
    data = mnist['data']
    labels = mnist['target']

    neg, pos = "0", "8"
    train_idx = numpy.random.RandomState(0).permutation(
        np.where((labels[:60000] == neg) | (labels[:60000] == pos))[0])
    test_idx = numpy.random.RandomState(0).permutation(
        np.where((labels[60000:] == neg) | (labels[60000:] == pos))[0])

    train_data_unscaled = data[train_idx[:6000], :].astype(float)
    train_labels = (labels[train_idx[:6000]] == pos) * 2 - 1

    validation_data_unscaled = data[train_idx[6000:], :].astype(float)
    validation_labels = (labels[train_idx[6000:]] == pos) * 2 - 1

    test_data_unscaled = data[60000 + test_idx, :].astype(float)
    test_labels = (labels[60000 + test_idx] == pos) * 2 - 1

    # Preprocessing
    train_data = sklearn.preprocessing.scale(
        train_data_unscaled, axis=0, with_std=False)
    validation_data = sklearn.preprocessing.scale(
        validation_data_unscaled, axis=0, with_std=False)
    test_data = sklearn.preprocessing.scale(
        test_data_unscaled, axis=0, with_std=False)
    return train_data, train_labels, validation_data, validation_labels, test_data, test_labels


def SGD_hinge(data, labels, C, eta_0, T):
    """
    Implements SGD for hinge loss.
    """
    w = np.zeros(VECTOR_SIZE, dtype=np.longdouble)
    for i in range(T):
        idx = randint(0, len(data))
        w = hinge_w(w, data[idx], labels[idx], C, eta_0/(i+1))
    return w


def SGD_log(data, labels, eta_0, T, get_norms=False):
    """
    Implements SGD for log loss.
    """
    norms = []
    w = np.zeros(VECTOR_SIZE, dtype=np.longdouble)
    for i in range(T):
        idx = randint(0, len(data))
        w = logloss_w(w, data[idx], labels[idx], eta_0/(i+1))
        if get_norms:
            norms.append(np.linalg.norm(w))
    if get_norms:
        return w, norms
    return w

#################################


@dataclass
class BestResult:
    performance: float
    parameter: Union[float, int]
    name: str
    section: str

    def update(self, performance, parameter):
        if performance < self.performance:
            return
        self.performance = performance
        self.parameter = parameter

    def __str__(self) -> str:
        return f'In Section {self.section} the best {self.name} was {self.parameter} with {100*self.performance:.4f}% success on validation'


def hinge_w(w, x, y, C, eta):
    '''
    Given w_t, x_i, y_i, C and eta_t returns w_(t+1) using hinge loss
    '''
    if np.dot(y*w, x) < 1:
        return np.add(np.multiply(w, np.longdouble(1-eta), dtype=np.longdouble), np.multiply(x, np.longdouble(C*eta*y), dtype=np.longdouble))
    return np.multiply(np.longdouble(1-eta), w, dtype=np.longdouble)


def logloss_w(w, x, y, eta):
    '''
    Given w_t, x_i, y_i, C and eta_t returns w_(t+1) using Log loss
    '''
    def grad_log(x, y, w):
        return x*(1/(1 + np.exp(np.dot(x, w)))-y)

    return np.subtract(
        w,
        eta*grad_log(x, y, w))

    pass


def well_predicted_rate(w, xvalid, yvalid):
    '''
    given w and a validation set return the well-predicted rate of xi using w
    '''
    def predict(x, w):
        return 1 if np.dot(x, w) > 0 else -1

    predicted = [1 for x, y in zip(xvalid, yvalid) if predict(x, w) == y]
    return len(predicted)/len(xvalid)


def section1A(*helper_data) -> BestResult:
    '''
    Perform requirement if section A
    helper_data = return of helper() function
    '''
    def plot(etas, performances, runs):
        plt.plot(etas, performances, color='b')
        plt.xscale('log')
        plt.xlabel(EtaChar)
        plt.ylabel(f"average accuracy over {runs} runs")
        blue_patch = patches.Patch(
            color='blue', label=f'Average Prediction Accuracy on Validation set vs {EtaChar}')
        plt.legend(handles=[blue_patch])
        plt.show()

    C, runs, T = 1, 10, 1000
    best = BestResult(0.0, 0.0, EtaChar, 'A')
    xtrain, ytrain, xvalid, yvalid, _, _ = helper_data
    etas = [10**i for i in range(-5, 6)]
    print(f"\tSection A: Finding best {EtaChar} from {etas}")
    avg_accuracy = []
    for eta0 in etas:
        w = [SGD_hinge(data=xtrain, labels=ytrain, C=C, eta_0=eta0, T=T)
             for r in range(runs)]
        performance = np.average([
            well_predicted_rate(wi, xvalid, yvalid)
            for wi in w
        ])
        avg_accuracy.append(performance)
        best.update(performance, eta0)

    print(f"\t\t {best}")
    plot(etas, avg_accuracy, runs)
    return best


def Section1B(best_eta: float, *helper_data) -> BestResult:
    '''
    Perform requirement if section B
    helper_data = return of helper() function
    '''
    def plot(Cs, performances, eta, runs):
        plt.plot(Cs, performances, color='g')
        plt.xscale('log')
        plt.xlabel("C")
        plt.ylabel(f"average accuracy over {runs} runs")
        green_patch = patches.Patch(
            color='green', label=f'Average Prediction Accuracy on Validation set [using {EtaChar}={eta}] vs C')
        plt.legend(handles=[green_patch])
        plt.show()

    runs, T = 10, 1000
    best = BestResult(0.0, 0.0, "C", "B")
    xtrain, ytrain, xvalid, yvalid, _, _ = helper_data
    Cs = [10**i for i in range(-5, 6)]
    print(
        f"\tSection B: for {EtaChar}={best_eta:.2f} Finding best C from {Cs}")
    avg_accuracy = []
    for C in Cs:
        w = [SGD_hinge(data=xtrain, labels=ytrain, C=C, eta_0=best_eta, T=T)
             for r in range(runs)]
        performance = np.average([
            well_predicted_rate(wi, xvalid, yvalid)
            for wi in w
        ])
        avg_accuracy.append(performance)
        best.update(performance, C)

    print(f"\t\t {best}")
    plot(Cs, avg_accuracy, best_eta, runs)
    return best


def Section1CD(eta, C, *helper_data):
    print(f"\tSection C&D: Finding Accuracy of {EtaChar}={eta}, C={C}")
    best = BestResult(0.0, 0.0, "Accuracy", "C&D")
    xtrain, ytrain, _, _, xtest, ytest = helper_data
    w = SGD_hinge(data=xtrain, labels=ytrain, C=C, eta_0=eta, T=2000)
    plt.imshow(np.reshape(w, (28, 28)), interpolation='nearest')
    plt.show()
    performace = well_predicted_rate(w, xtest, ytest)
    best.update(performace, "achieved")
    print(f"\t\t {best}")
    return best


def Question1(h):
    print("Question 1:")
    a = section1A(*h)
    b = Section1B(a.parameter, *h)
    cd = Section1CD(a.parameter, b.parameter, *h)
    print()


def section2A(*helper_data) -> BestResult:
    '''
    Perform requirement if section A
    helper_data = return of helper() function
    '''
    def plot(etas, performances, runs):
        plt.plot(etas, performances, color='b')
        plt.xscale('log')
        plt.xlabel(EtaChar)
        plt.ylabel(f"average accuracy over {runs} runs")
        blue_patch = patches.Patch(
            color='blue', label=f'Average Prediction Accuracy on Validation set vs {EtaChar}')
        plt.legend(handles=[blue_patch])
        plt.show()

    runs, T = 10, 1000
    best = BestResult(0.0, 0.0, EtaChar, 'A')
    xtrain, ytrain, xvalid, yvalid, _, _ = helper_data
    etas = [10**i for i in range(-5, 6)]
    print(f"\tSection A: Finding best {EtaChar} from {etas}")
    avg_accuracy = []
    for eta0 in etas:
        w = [SGD_log(data=xtrain, labels=ytrain, eta_0=eta0, T=T)
             for r in range(runs)]
        performance = np.average([
            well_predicted_rate(wi, xvalid, yvalid)
            for wi in w
        ])
        avg_accuracy.append(performance)
        best.update(performance, eta0)

    print(f"\t\t {best}")
    plot(etas, avg_accuracy, runs)
    return best


def Section2B(best_eta: float, *helper_data) -> BestResult:
    '''
    Perform requirement if section B
    helper_data = return of helper() function
    '''
    runs, T = 10, 2000
    best = BestResult(0.0, 0.0, "w", "B")
    xtrain, ytrain, xvalid, yvalid, xtest, ytest = helper_data
    print(
        f"\tSection B: for {EtaChar}={best_eta:.2f} Finding best Classifier")
    w = SGD_log(data=xtrain, labels=ytrain, eta_0=best_eta, T=T)
    performance = well_predicted_rate(w, xtest, ytest)
    best.update(performance, f"[{w[0]}, ... , {w[-1]}]")
    print(f"\t\t {best}")
    plt.imshow(np.reshape(w, (28, 28)), interpolation='nearest')
    plt.show()
    return best


def Section2C(best_eta: float, *helper_data) -> BestResult:
    '''
    Perform requirement if section B
    helper_data = return of helper() function
    '''
    def plot(iters, norms):
        plt.plot(iters, norms, color='g')
        # plt.xscale('log')
        plt.xlabel("Iteration")
        plt.ylabel(f"norm of wi")
        green_patch = patches.Patch(
            color='green', label=f'Norm of w in iteration #i')
        plt.legend(handles=[green_patch])
        plt.show()

    runs, T = 10, 2000
    best = BestResult(0.0, 0.0, "w", "B")
    xtrain, ytrain, xvalid, yvalid, _, _ = helper_data
    print(
        f"\tSection C: for {EtaChar}={best_eta:.2f} Finding Classifier's norm per iteration")
    w, norms = SGD_log(data=xtrain, labels=ytrain,
                       eta_0=best_eta, T=T, get_norms=True)
    plot([i for i in range(T)], norms)
    return best


def Question2(h):
    print("Question 2:")
    a = section2A(*h)
    b = Section2B(a.parameter, *h)
    c = Section2C(a.parameter, *h)
    print()


#################################


if __name__ == "__main__":
    print(f'Programming Assignment #3: Getting Labeled-Data from mnist_784 using helper()')
    h = helper()
    Question1(h)
    Question2(h)
