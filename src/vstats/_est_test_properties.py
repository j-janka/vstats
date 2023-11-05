"""Functionality to  estimate properties of statistical tests
by means of simulations"""
import math
import random
from typing import List, Union

import numpy as np
import pandas as pd
import statsmodels.stats.proportion as ssp
from scipy import stats


def _samples_per_batch(
        sample_size: int,
        n_simulation: int) -> np.array:
    """Helper function useful to simulate repetitions of a statistical test
    by means of a vectorized implementation of a test but without using the
    whole RAM. The function returns a one-dimensional Numpy array of
    numbers of samples that can be computed at once by the vectorized
    implementation of the test without blowing up a medium sized RAM.
    Hence each array element describes the size of a batch in terms of the
    number of samples it contains. If you then iterate over these batches,
    you can perform all tests. Note that a batch contains at least one sample,
    no matter how big the sample size is.

    Parameters
    ----------
    sample_size : int
        Sample size for which the test is preformed.
    n_simulation : int
        Number of simulation of the test performed.

    Returns
    -------
    numpy.ndarray, (1,)
        One-dimensional Numpy array of numbers of samples that each can
        be computed at once by the vectorized implementation of a test
        without blowing up a medium sized RAM. The elements of the list
        add up to `n_simulation`.

    Examples
    --------
    >>> _samples_per_batch(sample_size=300, n_simulation=1000000)
    >>> array([333, 333, 333, ..., 333, 333,   1])
    >>> _samples_per_batch(sample_size=300, n_simulation=1000000).sum()
    >>> 1000000
    """
    # At least one sample must fit into memory,
    # otherwise things do not work anyway:
    max_observations_per_batch = max(sample_size, 100000)
    max_samples_per_batch = math.floor(
        max_observations_per_batch / sample_size
    )
    samples_per_batch = np.repeat(
        a=max_samples_per_batch,
        repeats=math.floor(
            n_simulation / max_samples_per_batch
        )
    )
    if n_simulation % max_samples_per_batch != 0:
        samples_per_batch = np.append(
            samples_per_batch,
            n_simulation % max_samples_per_batch
        )
    return samples_per_batch


def _fix_degenerated_columns(
        x: np.ndarray,
        rv: Union[stats.rv_discrete, stats.rv_continuous],
        rng: Union[random.Random, np.random.Generator] = np.random.default_rng()) -> int:  # noqa: E501
    """Helper function useful to fix degenerated samples in a
    two-dimensional Numpy array `x` containing one sample per column.
    Replaces columns of `x` containing only a single value with more
    diverse columns generated with the random variable `rv`.
    The replacement is an in-place operation.

    Parameters
    ----------
    x : numpy.ndarray, (number of observation of group, number of samples)
        Two-dimensional Numpy array of floats or integers.
    rv : random variable from scipy.stats
        Instance of a random variable from `scipy.stats` used to
        generate more diverse columns.
    rng: Instance or derivate of random.Random or numpy.random.Generator
        Pseudo-random number generator that is used as `random_state`
        of the random variable `rv`. We recommend using
        `numpy.random.default_rng` with a seed generated by
        `numpy.random.SeedSequence().entropy`. Note that if this is done,
        the results are only guaranteed to be reproducible with the same
        version of Numpy with which they were generated.

    Returns
    -------
    int
        Number of degenerated columns fixed. Useful for
        monitoring purposes.

    Examples:
    ---------
    >>> import numpy as np
    >>> import scipy
    >>>
    >>> x=np.array([[1,2,3],[4,2,6]])
    >>>
    >>> _fix_degenerated_columns(
    ... x=x,
    ... rv=scipy.stats.rv_discrete(values=([1, 2, 3], [0.2, 0.2, 0.6])),
    ... rng=np.random.default_rng(seed=165372562462609342146380204649518102930)
    ... )
    1
    >>> print(x)
    [[1 2 3]
     [4 3 6]]
    """
    n = x.shape[0]
    is_single_value_column = np.all(x == x[0], axis=0)
    if np.any(is_single_value_column):
        retry = True
        while retry:
            x_temp = rv.rvs(
                size=n * np.sum(is_single_value_column),
                random_state=rng
            )
            x_temp = x_temp.reshape(n, np.sum(is_single_value_column))
            is_single_value_column_in_substitutes = np.all(
                x_temp == x_temp[0],
                axis=0
            )
            if np.all(~is_single_value_column_in_substitutes):
                # Replace degenerated columns with more diverse columns:
                x[:, is_single_value_column] = x_temp
                retry = False
    num_degenerated_columns_fixed = is_single_value_column.sum()
    return num_degenerated_columns_fixed


def get_est_welchs_test_properties(
        n: List[int],
        share_n_group_1: float,
        alpha_test: float,
        rv_1: Union[stats.rv_discrete, stats.rv_continuous],
        rv_2: Union[stats.rv_discrete, stats.rv_continuous],
        conf_level_rejection_prob: float,
        n_simulation: int,
        fix_degenerated_samples: bool = False,
        rng: Union[random.Random, np.random.Generator] = np.random.default_rng()) -> pd.DataFrame:  # noqa: E501
    """Estimates properties of a two-sided Welch's test
    for two independent samples for a list of given total sample sizes
    and for given random variables from `scipy.stats` generating the samples.
    The estimation also includes confidence intervals of the true value
    of the properties. Currently only rejection probabilities of the test
    are estimated. Exact confidence intervals for the true rejection
    probabilities are calculated as in Theorem 8.11 in Georgii (2008, p. 231).
    For details about Welch's test and its assumptions see
    Loveland (2011, p. 48, 101) and section 10.2 in Devore (2018).

    Parameters
    ----------
    n : list of int
        List of total sample sizes to estimate the properties of the test for.
        The total sample size is the sum of the sample size of group 1 and the
        sample size of group 2.
    share_n_group_1 : float
        Desired share of the sample size of group 1 in relation
        to the total sample size `n` of both groups. Note
        that the actual sample size `n_1` of group 1 is rounded up to
        the nearest integer, so that the desired share is not always
        attained exactly. The sample size of group 2 is `n_2 = n - n_1`.
    alpha_test : float
        Nominal significance level of the test applied.
    rv_1 : random variable from scipy.stats
        Random variable used to generate the samples for group 1. Use
        `scipy.stats.norm` to fulfill the assumptions of Welch's test. Use
        other random variables from `scipy.stats` to test the effect of
        violations of the assumptions.
    rv_2 : random variable from scipy.stats
        Random variable used to generate the samples for group 2.
        Same usage as for `rv_1` above.
    conf_level_rejection_prob : float
        Confidence level of the confidence intervals generated for the
        true rejection probabilities.
    n_simulation : int
        Number of tests to perform for estimating properties of the test
        for a given total sample size.
    fix_degenerated_samples : bool, optional
        If `True`, a more diverse sample is drawn for a group if its sample
        contains only a single value. The frequency with which this
        happens is reflected in the columns `share_fixed_samples_group_*`
        in the output, which are only included in the output if
        `fix_degenerated_samples` is `True`. If a fix occurs too frequently,
        the corresponding results are not reliable. This parameter is
        useful when discrete distributions are used to generate the samples
        in order to evaluate the performance of the test under violated
        assumptions, - in this case set `fix_degenerated_samples` to
        `True`. By default `False`.
    rng: Instance or derivate of random.Random or numpy.random.Generator, optional
        Pseudo-random number generator that is used as `random_state`
        of the random variable `rv`. We recommend using
        `numpy.random.default_rng` with a seed generated via
        `numpy.random.SeedSequence().entropy`. Note that if this is done,
        the results are only guaranteed to be reproducible with the same
        version of Numpy with which they were generated.
        By default `numpy.random.default_rng()`.

    Returns
    -------
    pandas.DataFrame
        Result of the estimation of properties of Welch's test with the columns:

        n : integer
            Total sample size for which properties of the test are estimated.
            The total sample size is the sum of the sample size of group 1
            and the sample size of group 2.
        n_1 : integer
            Sample size of group 1.
        n_2 : integer
            Sample size of group 2.
        rejection_prob_est : floating point number
            Estimate of the rejection probability, i.e. of the probability
            that the null hypothesis is rejected by the test.
        ll_conf_int_rejection_prob : floating point number
            Lower limit of the exact confidence interval for the true
            rejection probability.
        ul_conf_int_rejection_prob : floating point number
            Upper limit of the exact confidence interval for the true
            rejection probability.
        share_fixed_samples_group_1 : floating point number
            Share of fixed samples of group 1 with respect to all
            `n_simulation` samples drawn to estimate properties of the test
            for a given total sample size. Fixed samples are samples that
            initially contained only a single value and were replaced by a more
            diverse sample obtained by drawing again until a more diverse sample
            occurred. If this share is too large, the estimates
            of the properties are not reliable. This column is only included, if
            `fix_degenerated_samples` is `True`.
        share_fixed_samples_group_2: floating point number
            Share of fixed samples of group 2 with respect to all
            `n_simulation` samples drawn to estimate properties of the test
            for a given sample size. Details are analogous to those of
            `share_fixed_samples_group_1`. This column is only included, if
            `fix_degenerated_samples` is `True`.

    Examples
    --------
    >>> import numpy as np
    >>> from scipy import stats
    >>> import vstats
    >>>
    >>> min_abs_effect = 3  # minimal absolute effect to detect
    >>>
    >>> vstats.get_est_welchs_test_properties(
    >>>     n=[20, 40, 80, 160, 320, 640],
    >>>     share_n_group_1=0.3,
    >>>     alpha_test=0.05,
    >>>     rv_1=stats.norm(loc=min_abs_effect, scale=8),
    >>>     rv_2=stats.norm(loc=0, scale=10),
    >>>     conf_level_rejection_prob=0.95,
    >>>     n_simulation=10000,
    >>>     rng=np.random.default_rng(seed=165372562462609342146380204649518102930)
    >>> )
    ...      n  n_1  n_2  rejection_prob_est  ll_conf_int_rejection_prob  ul_conf_int_rejection_prob
    ... 0   20    6   14              0.0998                    0.093992                    0.105842
    ... 1   40   12   28              0.1586                    0.151490                    0.165909
    ... 2   80   24   56              0.2784                    0.269631                    0.287298
    ... 3  160   48  112              0.5207                    0.510854                    0.530534
    ... 4  320   96  224              0.8081                    0.800242                    0.815778
    ... 5  640  192  448              0.9767                    0.973551                    0.979567

    For an example on how to display this result, see the docstring of
    the function `create_figure_for_rejection_probabilities` of the package
    `vplotly`.

    References
    ----------
    Devore, Jay L. & Berk, Kenneth N. (2018). Modern Mathematical Statistics
    with Applications. 2nd ed. New York, Dordrecht, Heidelberg, London: Springer.
    ISBN: 978-1-4614-0390-6.

    Loveland, Jennifer L. (2011). Mathematical Justification of Introductory
    Hypothesis Tests and Development of Reference Materials.
    All Graduate Plan B and other Reports. 14.
    https://digitalcommons.usu.edu/gradreports/14

    Georgii, Hans-Otto (2008). Stochastics: Introduction to Probability and
    Statistics. Berlin, New York: De Gruyter.
    https://doi.org/10.1515/9783110206760
    """  # noqa: E501
    results = []

    for sample_size in n:

        num_fixed_samples_group_1 = 0
        num_fixed_samples_group_2 = 0

        n_1 = math.ceil(sample_size * share_n_group_1)
        n_2 = sample_size - n_1

        # Perform simulation in batches of samples to boost speed
        # using the fact that stats.ttest_ind is vectorized:
        bernoulli_trial: List[bool] = []
        for samples_per_batch in _samples_per_batch(sample_size, n_simulation):
            # Draw samples for the current batch for group 1 and
            # placed each sample in a column:
            x_1 = rv_1.rvs(
                size=n_1 * samples_per_batch,
                random_state=rng
            )
            x_1 = x_1.reshape(n_1, samples_per_batch)

            if fix_degenerated_samples:
                # Avoid RuntimeWarning in stats.ttest_ind occurring,
                # if sample of a group contains only a single value.
                num_fixed_samples_group_1 += \
                    _fix_degenerated_columns(
                        x_1,
                        rv_1,
                        rng
                    )

            # Draw samples for the current batch for group 2 and
            # placed each sample in a column:
            x_2 = rv_2.rvs(
                size=n_2 * samples_per_batch,
                random_state=rng
            )
            x_2 = x_2.reshape(n_2, samples_per_batch)

            if fix_degenerated_samples:
                # Avoid RuntimeWarning in stats.ttest_ind occurring
                # if sample of a group contains only a single value.
                num_fixed_samples_group_2 += \
                    _fix_degenerated_columns(
                        x_2,
                        rv_2,
                        rng
                    )

            test_result = stats.ttest_ind(a=x_1, b=x_2, equal_var=False)
            bernoulli_trial = np.concatenate(
                (bernoulli_trial, (test_result.pvalue <= alpha_test))
            )

        number_rejections = sum(bernoulli_trial)
        rejection_prob_est = number_rejections / n_simulation

        conf_int_rejection_prob = ssp.proportion_confint(
            nobs=n_simulation,
            count=number_rejections,
            alpha=1 - conf_level_rejection_prob,
            method='beta'
        )

        result_bernoulli_trial = {
            "n": sample_size,
            "n_1": n_1,
            "n_2": n_2,
            "rejection_prob_est": rejection_prob_est,
            "ll_conf_int_rejection_prob": conf_int_rejection_prob[0],
            "ul_conf_int_rejection_prob": conf_int_rejection_prob[1]
        }

        if fix_degenerated_samples:
            result_bernoulli_trial["share_fixed_samples_group_1"] = round(
                num_fixed_samples_group_1 / n_simulation, 2)
            result_bernoulli_trial["share_fixed_samples_group_2"] = round(
                num_fixed_samples_group_2 / n_simulation, 2)

        results.append(result_bernoulli_trial)

    df = pd.DataFrame(results)
    return df
