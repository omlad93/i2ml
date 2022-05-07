#################################
# Your name: Omri Elad
#            204620702
# PyVersion: Python 3.9.10
#################################
from __future__ import annotations
from re import S
from typing import Any, Iterable
import numpy as np
from numpy.random import uniform, choice
from math import sqrt, exp, e
from math import log as ln
from os.path import sep
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import intervals
from intervals import find_best_interval


class Assignment2(object):
    """Assignment 2 skeleton.

    Please use these function signatures for this assignment and submit this file, together with the intervals.py.
    """

    def sample_from_D(self, m: int) -> np.ndarray:
        """Sample m data samples from D.
        Input: m - an integer, the size of the data sample.

        Returns: np.ndarray of shape (m,2) :
                A two dimensional array of size m that contains the pairs where drawn from the distribution P.
        """
        # TODO: Check

        def prob_y1(x: float) -> float:
            '''
            return the probability y is 1.
                0.8 if x ∈ [0, 0.2] U [0.4, 0.6] U [0.8, 1]
                0.1 if x ∈ (0.2, 0.4) U (0.6, 0.8)
            '''
            in_intervals = [
                a <= x <= b
                for a, b in zip([0.0, 0.4, 0.8], [0.2, 0.6, 1.0])
            ]
            return 0.8 if any(in_intervals) else 0.1  # TODO:Verify

        x = sorted(uniform(0, 1, m))
        y = [
            choice([0, 1], p=[1-prob_y1(xi), prob_y1(xi)])
            for xi in x
        ]
        retval = np.asarray(
            [
                (xi, yi)
                for xi, yi in zip(x, y)
            ]
        )
        assert retval.shape == (m, 2)
        return retval

    def experiment_m_range_erm(self, m_first, m_last, step, k, T):
        """Runs the ERM algorithm.
        Calculates the empirical error and the true error.
        Plots the average empirical and true errors.
        Input: m_first - an integer, the smallest size of the data sample in the range.
               m_last - an integer, the largest size of the data sample in the range.
               step - an integer, the difference between the size of m in each loop.
               k - an integer, the maximum number of intervals.
               T - an integer, the number of times the experiment is performed.

        Returns: np.ndarray of shape (n_steps,2).
            A two dimensional array that contains the average empirical error
            and the average true error for each m in the range accordingly.
        """

        def single_m_erm(self: Assignment2, *, m: int, k: int, average_factor: int = 1) -> tuple[Any, ...]:
            s = self.sample_from_D(m)
            # x =[si[0] for si in s], y= [si[1] for si in s]
            xs, ys = list(zip(*s))
            best, e = find_best_interval(xs, ys, k)
            return (e/m/average_factor, self.true_error(best)/average_factor)

        def gather(results: list[tuple]):
            em, tr = list(zip(*results))
            return (sum(em), sum(tr))

        # TODO: Implement the loop
        m_range = [m for m in range(m_first, m_last + 1, step)]
        empirical, true = list(zip(*
                                   [
                                       gather(single_m_erm(self, m=m, k=k, average_factor=T)
                                              for idx in range(T))
                                       for m in m_range
                                   ]
                                   ))
        assert len(empirical) == len(m_range)
        assert len(true) == len(m_range)
        # TODO & FIXME: Replace empirical & true values ?
        self.plot_errors(f"M Range SRM ({T})", "m values", m_range,
                         true=true, empirical=empirical,)
        return (np.asarray(
                [
                    (emp, tru)
                    for emp, tru in zip(empirical, true)
                ]
                ))

    def experiment_k_range_erm(self, m, k_first, k_last, step):
        """Finds the best hypothesis for k= 1,2,...,10.
        Plots the empirical and true errors as a function of k.
        Input: m - an integer, the size of the data sample.
               k_first - an integer, the maximum number of intervals in the first experiment.
               m_last - an integer, the maximum number of intervals in the last experiment.
               step - an integer, the difference between the size of k in each experiment.

        Returns: The best k value (an integer) according to the ERM algorithm.
        """
       # TODO: Implement the loop

        def single_k_erm(self: Assignment2, *, m: int, k: int, s) -> tuple[Any, ...]:
            xs, ys = list(zip(*s))
            best, e = find_best_interval(xs, ys, k)
            print(f'\t k={k},m={m},e={e}')
            return (e/m, self.true_error(best))

        k_range = [m for m in range(k_first, k_last + 1, step)]
        s = self.sample_from_D(m)
        empirical, true = list(zip(
            *[
                single_k_erm(self, m=m, k=k, s=s)
                for k in k_range
            ]
        ))

        assert len(empirical) == len(k_range)
        assert len(true) == len(k_range)
        self.plot_errors("K Range ERM", "k values", k_range,
                         true=true, empirical=empirical)
        return k_range[empirical.index(min(empirical))]

    def experiment_k_range_srm(self, m, k_first, k_last, step):
        """Run the experiment in (c).
        Plots additionally the penalty for the best ERM hypothesis.
        and the sum of penalty and empirical error.
        Input: m - an integer, the size of the data sample.
               k_first - an integer, the maximum number of intervals in the first experiment.
               m_last - an integer, the maximum number of intervals in the last experiment.
               step - an integer, the difference between the size of k in each experiment.

        Returns: The best k value (an integer) according to the SRM algorithm.
        """

        def single_k_srm(self: Assignment2, *, m: int, k: int, s, delta=0.1) -> tuple[Any, ...]:
            xs, ys = list(zip(*s))
            best, e = find_best_interval(xs, ys, k)
            assert e*m/k >= 0, f'e={e}, m={m}, k={k}, em/k = {e*m/k}'
            penalty = sqrt((2 * k + ln(2 / delta)) / m)  # FIXME
            return (e/m, self.true_error(best), penalty)

        # TODO: Implement the loop
        s = self.sample_from_D(m)
        k_range = [m for m in range(k_first, k_last + 1, step)]
        empirical, true, penalty = list(zip(
            *[
                single_k_srm(
                    self, m=m, k=k, s=s)
                for k in k_range
            ]
        ))
        assert len(empirical) == len(k_range)
        assert len(true) == len(k_range)
        assert len(penalty) == len(k_range)

        # TODO & FIXME: Replace empirical & true values ?
        self.plot_errors(
            "K Range SRM", "k values", k_range,
            true=true, empirical=empirical, penalty=penalty
        )
        sum_penalty_empirical = [em+pe for em, pe in zip(empirical, penalty)]
        min_val = min(sum_penalty_empirical)
        return next(k for k in k_range if sum_penalty_empirical[k] == min_val)

    def cross_validation(self, m, ho: float = 0.2):
        """Finds a k that gives a good test error.
        Input: m - an integer, the size of the data sample.

        Returns: The best k value (an integer) found by the cross validation algorithm.
        """
        def holdout_error(xt, yt, intervals):
            def in_intervals(x, intervals):
                return any([True if i[0] <= x <= i[1] else False for i in intervals])

            def mistake(x, y, intervals):
                return (y and not(in_intervals(x, intervals))) or (not(y) and (in_intervals(x, intervals)))

            mistakes = len([
                0
                for x, y in zip(xt, yt) if mistake(x, y, intervals)
            ])

            return mistakes

        def single_holdout(self: Assignment2, *, xs, ys, k, xt, yt) -> tuple[Any, ...]:
            best, e = find_best_interval(xs, ys, k)
            ho_errors = holdout_error(xt, yt, best)
            print(f'\t k={k}, empirical={e}, ho_errors={ho_errors}')
            return ho_errors/len(xt), best

        k_range = [i for i in range(1, 11)]
        print(
            f'\t out of {m} samples: {int(ho*m)} will go to test and {int((1-ho)*m)} will go to train ')
        x_train, y_train = list(zip(
            *self.sample_from_D(int((1-ho)*m))
        ))
        x_test, y_test = list(zip(
            *self.sample_from_D(int(ho*m))
        ))

        hold_out, intervals = list(zip(
            *[
                single_holdout(self, xs=x_train, ys=y_train,
                               k=k, xt=x_test, yt=y_test)
                for k in k_range
            ]
        ))
        self.plot_holdout("Holdout Error By K", "k values", k_range, hold_out)

        min_ho_error = min(hold_out)
        print('Finished X-Validation')

        informative = next((k, i) for k, i, h in zip(
            k_range, intervals, hold_out) if h == min_ho_error)
        # print(f'\tbest k:{x[0]} \n\twith intervals:\n\t{x[1]}')
        print(f'\tbest k:{informative[0]}')
        y = [
            (f'{xi[0]:.4f}', f'{xi[1]:.4f}')
            for xi in informative[1]
        ]
        print(f'\twith intervals:\n\t{y}')
        print(f'\twith intervals:\n\t{informative[1]}')
        return informative[0]

    #################################
    # Place for additional methods

    def intersection(self, i1: Iterable, i2: Iterable) -> float:
        s1, s2 = (i1, i2) if i1[0] < i2[0] else (i2, i1)  # s1 start before s2
        if s1[1] < s2[0]:  # if s2 starts after s1[1]
            return 0.0
        contained, intersect = s1[1]-s2[0], s2[1]-s2[0]
        return min(contained, intersect)

    def union(self, intervals: Iterable) -> float:
        return sum([
            i[1]-i[0]
            for i in intervals
        ])

    def complements(self, intervals: Iterable) -> Iterable:
        extended = [(0, 0), *intervals, (1, 1)]
        return [
            (extended[idx][1], extended[idx+1][0])
            for idx in range(len(extended)-1)
        ]

    def true_error(self, intervals) -> float:
        one_intervals = tuple(
            (a, b)
            for a, b in zip([0.0, 0.4, 0.8], [0.2, 0.6, 1.0])
        )
        zero_intervals = tuple(
            (a, b)
            for a, b in zip([0.2, 0.6], [0.4, 0.8])
        )
        union = self.union(intervals)
        complements = self.complements(intervals)
        union_complements = self.union(complements)
        intersection_one = sum(
            self.intersection(interval, one_interval)
            for one_interval in one_intervals for interval in intervals
        )
        intersection_zero = sum(
            self.intersection(comp_interval, zero_interval)
            for zero_interval in zero_intervals for comp_interval in complements
        )
        return sum(
            [
                0.2*intersection_one,
                0.9*(union-intersection_one),
                0.8*(union_complements-intersection_zero),
                0.1*intersection_zero
            ]
        )

    def plot_errors(self, plot_title, x_title, x_axis, true, empirical, penalty=None, show: bool = True) -> None:
        blue = patches.Patch(color='blue', label='Empirical Error')
        green = patches.Patch(color='green', label='True Error')
        if penalty is None:
            plt.legend(handles=[blue, green])
            plt.plot(x_axis, empirical, 'bo',
                     x_axis, true, 'gs')
        else:
            ep = [e+p for e, p in zip(empirical, penalty)]
            red = patches.Patch(color='red', label='Penalty')
            grey = patches.Patch(color='Grey', label='Empirical + Penalty')
            plt.legend(handles=[blue, green, red, grey])
            plt.plot(x_axis, empirical, 'bo',
                     x_axis, true, 'gs',
                     x_axis, penalty, 'rx',
                     x_axis, ep, 'grey')
        plt.xlabel(x_title)
        plt.title(plot_title)
        if show:
            plt.show()
        plt.savefig(f'out{sep}{plot_title}.png')

    def plot_holdout(self, plot_title, x_title, x_axis, holdout, show: bool = True) -> None:
        blue = patches.Patch(color='blue', label='holdout Error')
        plt.legend(handles=[blue])
        plt.plot(x_axis, holdout, 'bo')
        plt.xlabel(x_title)
        plt.title(plot_title)
        if show:
            plt.show()
        plt.savefig(f'out{sep}{plot_title}.png')

    #################################


if __name__ == '__main__':
    ass = Assignment2()
    print('Starting')
    # ass.experiment_m_range_erm(10, 100, 5, 3, 100)
    # print('Finished M Range ERM')
    # ass.experiment_k_range_erm(1500, 1, 10, 1)
    # print('Finished K Range ERM')
    # ass.experiment_k_range_srm(1500, 1, 10, 1)
    # print('Finished K Range SRM')
    # x = ass.cross_validation(600)
    x = ass.cross_validation(1500)
