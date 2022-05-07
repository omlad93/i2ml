from dataclasses import dataclass
from numpy import linspace, random, linalg
# from numpy.random import RandomState
from random import choice
import matplotlib.pyplot as plot
from math import pow, fabs, e
from sklearn.datasets import fetch_openml


def question_1(N: int = 200_000,
               n: int = 20,
               values_count: int = 50,
               possible_values: list[int] = [0, 1]) -> None:

    def get_averages(N: int, n: int, possible_values: list[int]) -> list[float]:
        averages = []
        for i in range(n):
            sum = 0
            for j in range(N):
                x = choice(possible_values)
                sum += x
            averages.append(sum/N)
        return averages
    print(f'starting run for Q1: with N={N},n={n}')
    print('\tCalculating avereges')
    averages = get_averages(N, n, possible_values)
    epsilons = linspace(0, 1, values_count)
    empiric_results, hoeffding = [], []

    print('\tCalculating empiric results and hoeffding bounds\n')
    for epsilon in epsilons:
        y = 0
        for x in averages:
            y = y+1 if fabs(x-0.5) > epsilon else y
        empiric_results.append(y/n)
        hoeffding.append(2*pow(e, -2*epsilon**2*N))

    plot.plot(epsilons, empiric_results)
    plot.plot(epsilons, hoeffding)
    plot.title(f'Plot using N={N},n={n}')
    plot.show()


def question_2():

    def label_prediction(train_set, label_set, query_image, k):
        @dataclass
        class Record:
            dist: float
            label: str

        records: list[Record] = []
        for image, label in zip(train_set, label_set):
            records.append(Record(dist=linalg.norm(image - query_image),
                                  label=label))
        records.sort(key=lambda rec: rec.dist)
        labels = [r.label for r in records][:k]
        counter = {
            label: labels.count(label)
            for label in labels
        }
        return max(counter, key=counter.get)

    def single_test(n: int, k: int, train, train_labels, test, test_labels):
        current_train = train[:n]
        current_labels = train_labels[:n]
        hits, m = 0, len(test)
        for query_image, matching_label in zip(test, test_labels):
            p = label_prediction(current_train, current_labels, query_image, k)
            hits = hits+1 if p == matching_label else hits
        return hits/m

    figure, axes = plot.subplots(nrows=2, ncols=1)
    print(f'starting run for Q2:')
    mnist = fetch_openml('mnist_784', as_frame=False)
    print(f'\tfetched MNIST')
    data = mnist['data']
    labels = mnist['target']
    idx = random.RandomState(0).choice(70000, 11000)
    train = data[idx[:10000], :].astype(int)
    train_labels = labels[idx[:10000]]
    test = data[idx[10000:], :].astype(int)
    test_labels = labels[idx[10000:]]
    print(f'\tloadind data and labels from MNIST is complete')

    fixed_n = 1_000
    fixed_k = 1

    k_range = [k for k in range(1, 101)]
    n_range = [100*i for i in range(1, 51)]
    print(
        f'\trunning a tests with n={fixed_n},k in [{min(k_range)} , ... , {max(k_range)} ]')
    rates2 = [single_test(fixed_n, k, train, train_labels, test, test_labels)
              for k in k_range]
    axes[0].plot(k_range, rates2)
    print(
        f'\trunning a tests with k={fixed_k},n in [{min(n_range)} , ... , {max(n_range)} ]\n')
    rates3 = [
        single_test(n, fixed_k, train, train_labels, test, test_labels)
        for n in n_range
    ]
    axes[1].plot(n_range, rates3)
    figure.tight_layout()
    plot.show()


if __name__ == '__main__':
    question_1()
    question_2()
    print('done')
