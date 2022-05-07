'''
Omri Elad 204620702
Python 3.10
'''

import numpy as np
import math
import matplotlib.pyplot as plt
from sklearn import svm, datasets


def plot_results(models, titles, X, y, plot_sv=False):
    # Set-up 2x2 grid for plotting.
    if len(titles) <= 4:
        fig, sub = plt.subplots(1, len(titles))  # 1, len(list(models)))
    else:
         fig, sub = plt.subplots(2, len(titles)//2)  # 1, len(list(models)))

    X0, X1 = X[:, 0], X[:, 1]
    xx, yy = make_meshgrid(X0, X1)

    if len(titles) == 1:
        sub = [sub]
    else:
        sub = sub.flatten()
    for clf, title, ax in zip(models, titles, sub):
        # print(title)
        plot_contours(ax, clf, xx, yy, cmap=plt.cm.coolwarm, alpha=0.8)
        ax.scatter(X0, X1, c=y, cmap=plt.cm.coolwarm, s=20, edgecolors="k")
        if plot_sv:
            sv = clf.support_vectors_
            ax.scatter(sv[:, 0], sv[:, 1], c='k', s=60)

        ax.set_xticks(())
        ax.set_yticks(())
        ax.set_title(title)
        ax.set_aspect('equal', 'box')
    fig.tight_layout()
    plt.show()


def make_meshgrid(x, y, h=0.02):
    """Create a mesh of points to plot in

    Parameters
    ----------
    x: data to base x-axis meshgrid on
    y: data to base y-axis meshgrid on
    h: stepsize for meshgrid, optional

    Returns
    -------
    xx, yy : ndarray
    """
    x_min, x_max = x.min() - 1, x.max() + 1
    y_min, y_max = y.min() - 1, y.max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    return xx, yy


def plot_contours(ax, clf, xx, yy, **params):
    """Plot the decision boundaries for a classifier.

    Parameters
    ----------
    ax: matplotlib axes object
    clf: a classifier
    xx: meshgrid ndarray
    yy: meshgrid ndarray
    params: dictionary of params to pass to contourf, optional
    """
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    out = ax.contourf(xx, yy, Z, **params)
    return out

def fit_kernel(kernel, x, y, C,homogenous):
    d = int(kernel[-1]) if 'poly' in kernel else 0
    if 'poly' in kernel:
        kernel, d = kernel.split('_')
        clf = svm.SVC(kernel=kernel,C=C,degree=int(d),coef0=homogenous)
    elif 'rbf' in kernel:
        kernel, gamma = kernel.split('_')
        gamma = float(gamma)
        clf = svm.SVC(kernel=kernel,C=C,degree=d,coef0=homogenous,gamma=gamma)
    else:  #Linear  
        clf = svm.SVC(kernel=kernel,C=C,coef0=homogenous)
    clf.fit(x,y)
    print(f'\t\tFitted {clf}')
    return clf
    
def sectionAB(x,y,c,sec):
    print(f'\tRunning Section {sec}')
    assert sec in {'A','B'}
    homogenous = 1 if sec=='A' else 0
    titles = ('linear','poly_2','poly_3')
    models = tuple( 
        fit_kernel(kernel,x,y,c,homogenous)
        for kernel in titles
    )
    plot_results(models, titles, x, y)

def sectionC(xt,yt,c, xv,yv):
    def plot_scores(titles,scores):
        plt.plot(titles,scores)
        plt.ylabel('Accuracy')
        plt.xlabel('Model')
        plt.title('Accuracy On Noisy Labels')
        plt.legend
        plt.show()
        
    def label_modification(y):
        pos_0 = len([yi for yi in y if yi>0])
        for i,yi in enumerate(y):
            if yi <0:
                y[i]= -y[i] if np.random.random()<= 0.1 else y[i]
        pos_1 = len([yi for yi in y if yi>0])
        return y
    print(f'\tRunning Section C')
    nyt = label_modification(yt)
    nyv = label_modification(yv)
    gamma_list = [10**i for i in range(-5,6)]
    titles = (
        'poly_2', *[f'rbf_{gamma}' for gamma in gamma_list]
    )
    models = tuple( 
        fit_kernel(kernel,xt,nyt,c,0)
        for kernel in titles
    )
    scores = tuple(
        model.score(xv,nyv)
        for model in models
    )

    plot_results(
        models=[m for i,m in enumerate(models) if i in {0,7}],
        titles=[t for i,t in enumerate(titles) if i in {0,7}],
        X=xt, y=nyt
    )

    print("\n\t\tTraining Scores")
    for model,score in zip(models,scores):
        print(f"\t\t\t{str(model):<43} : {100*model.score(xt,nyt):.2f}%")
    # plot_scores(titles,scores)
    plot_results(models, titles, xt, nyt)

    print("\n\t\tVaildation Scores")
    for model,score in zip(models,scores):
        print(f"\t\t\t{str(model):<43} : {100*score:.2f}%")
    plot_results(models, titles, xv, nyv)


    
def generate_samples(n=100):
    radius = np.hstack([np.random.random(n), np.random.random(n) + 1.5])
    angles = 2 * math.pi * np.random.random(2 * n)
    X1 = (radius * np.cos(angles)).reshape((2 * n, 1))
    X2 = (radius * np.sin(angles)).reshape((2 * n, 1))
    X = np.concatenate([X1, X2], axis=1)
    y = np.concatenate([np.ones((n, 1)), -np.ones((n, 1))], axis=0).reshape([-1])
    return X,y

def main():
    C_hard = 1000000.0  # SVM regularization parameter
    C = 10
    n = 100
    print('Intro to ML Ex4 2022 Q2: SVM ')
    # Data is labeled by a circle
    x_train,y_train = generate_samples(n)
    x_valid, y_valid = generate_samples(n)
    sectionAB(x_train,y_train, C,"A")
    sectionAB( x_train,y_train, C,"B")
    sectionC(x_train,y_train,C,x_valid, y_valid)
    print("Done :)")

    

if __name__ == '__main__':
    main()