#################################
# Your name: Omri Elad 
# ID       : 204620702
# Python version 3.9.12
#################################

# Please import and use stuff only from the packages numpy, sklearn, matplotlib.
import sys
from matplotlib import pyplot as plt
import numpy as np
from process_data import parse_data
import matplotlib.patches as patches

np.random.seed(7)


def run_adaboost(X_train, y_train, T):
    """
    Returns: 

        hypotheses : 
            A list of T tuples describing the hypotheses chosen by the algorithm. 
            Each tuple has 3 elements (h_pred, h_index, h_theta), where h_pred is 
            the returned value (+1 or -1) if the count at index h_index is <= h_theta.

        alpha_vals : 
            A list of T float values, which are the alpha values obtained in every 
            iteration of the algorithm.
    """
    print(f'\tRunning AdaBoost with {T=} on {len(X_train)} samples:')
    n = len(X_train)
    dt = [1/n for i in range(n)]
    h,a = [], []
    for t in range(T):
        ht = get_best_hypothesis(dt,X_train,y_train)
        h.append(ht)
        et = error(dt, X_train, y_train,ht)
        wt = w(et)
        a.append(wt)
        dt=next_dt(dt,et,ht,wt,X_train,y_train)
        print(f'\t\t{t=:<4} : {ht=}')
    return h,a

##############################################
def sqrtet(et):
    return 2 * np.sqrt(et * (1 - et))

def error(dt,x_train,y_train,ht):
    return sum(
        [
            dt[i] if wrong_activation(ht,xi,yi) else 0
            for i,(xi,yi) in enumerate(zip(x_train,y_train))
        ]
        )

def w(error_t):
    return 0.5 * np.log((1 - error_t) / error_t)

def activate(ht,x):
    if x[ht[1]] <= ht[2]:
        return ht[0]
    return -ht[0]

def wrong_activation(ht,xi,yi):
    return activate(ht, xi) != yi

def next_dt(dt,et,ht,wt,x_train,y_train):
    return [
            (dt[i] * np.exp(-wt * yi * activate(ht, xi))) / sqrtet(et)
            for i,(xi,yi) in  enumerate(zip(x_train,y_train))
            ]    

def get_best_hypothesis_valued(d,x_train,y_train,value):

    min_f = sys.maxsize
    for j in range(len(x_train[0])):
        value_index_column_j = sorted(
            [
                (i,xi[j], yi, di)
                for i,(xi,yi,di) in enumerate(zip(x_train,y_train,d))
            ],
            key=lambda z:z[1]
        )
        value_index_column_j.append(
            (0, value_index_column_j[-1][1] + 1, 0, 0)
            )
        f = sum(
            j[3]
            for j in value_index_column_j if j[2]==value
        )
        if f < min_f:
            min_f = f
            min_theta = value_index_column_j[0][1] - 1
            min_j = j
            
        for i in range(len(value_index_column_j) - 1):
            f -= value_index_column_j[i][2] * value_index_column_j[i][3]
            if f < min_f and value_index_column_j[i][1] != value_index_column_j[i + 1][1]:
                min_f = f
                min_theta = 0.5 * (value_index_column_j[i][1] + value_index_column_j[i + 1][1])
                min_j = j
    return min_j, min_theta, min_f

def get_best_hypothesis(d, x_train, y_train):
    j1, t1, f1 = get_best_hypothesis_valued(d, x_train, y_train,1)
    j2, t2, f2 = get_best_hypothesis_valued(d, x_train, y_train,-1)
    if f1 < f2:
        return (1, j1, t1)
    return (-1, j2, t2)

def ada(hypotheses,alphas,x):
    assert len(hypotheses)==len(alphas)
    if sum([alpha*activate(h,x) for alpha,h in zip(alphas,hypotheses)])>=0:
        return 1
    return -1

def calc_error(h, alphas, x, y):
    s = sum(
        [
            1
            for xi,yi in zip(x,y) if ada(h,alphas,xi)!=yi
        ]
    )
    return s/len(x)

def error_exp_loss(h, alphas, x, y):
    s = sum(
    [
        np.exp(
            -yi * sum(
                [
                alphas[j] * activate(h[j], xi) for j in range(len(h))
                ]
                )
            )
        for xi,yi in zip(x,y)
    ]
    )
    return s/len(x)

def top_words(vocabulary,hypotheses,alphas):
    print(f'The most helpful words are:')
    [
        print(f'word={vocabulary[hypothesis[1]]:<20} {alpha=:.2f} {hypothesis=} ')
        for hypothesis,alpha in zip(hypotheses,alphas)
    ]
    

def plot_results(Ts, train,test, title):
    train_patch = patches.Patch(color='blue', label='Train')
    test_patch = patches.Patch(color='green', label='Test')
    plt.plot(Ts,train,color='blue',marker='o')
    plt.plot(Ts,test,color='green',marker='x')
    plt.xlabel("iteration")
    plt.title(title)
    plt.legend(handles=[train_patch,test_patch])
    plt.show()
##############################################


def main():
    T = 80
    Ts = [t for t in range(T)]
    print(f'\nInto To Machine Learning Ex5: AdaBoost')
    data = parse_data()
    if not data:
        return
    (X_train, y_train, X_test, y_test, vocab) = data

    hypotheses, alpha_vals = run_adaboost(X_train, y_train, T)

    train_error = [
        calc_error(hypotheses[:t+1],alpha_vals[:t+1],X_train,y_train)
        for t in Ts
    ]
    # print(train_error)
    train_exp_loss =[
        error_exp_loss(hypotheses[:t+1],alpha_vals[:t+1],X_train,y_train)
        for t in Ts
    ]
    test_error = [
        calc_error(hypotheses[:t+1],alpha_vals[:t+1],X_test,y_test)
        for t in Ts
    ]
    test_exp_loss = [
        error_exp_loss(hypotheses[:t+1],alpha_vals[:t+1],X_test,y_test)
        for t in Ts
    ]

    ## Q1 & Q3
    plot_results(Ts,train_error,test_error,"Error as function of iteration")
    ## Q2
    top_words(vocabulary=vocab,hypotheses=hypotheses,alphas=alpha_vals)
    ## Q3
    plot_results(Ts,train_exp_loss,test_exp_loss,"Exp-Loss as function of iteration")

    print('\n\nMischief Managed\n')

if __name__ == '__main__':
    main()


