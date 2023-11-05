"""Implementation of a two-sided Welch's test"""
from typing import Dict, List, Optional

from numpy import abs, mean, sqrt, var
from scipy import stats


def welchs_test(
        x_1: List[float],
        x_2: List[float],
        alpha: float,
        conf_level_effect: Optional[float] = None) -> Dict:

    """Two-sided Welch's test for two independent samples.
    Null hypothesis: The expected values of the population
    distributions underlying the samples `x_1` and `x_2` are equal.
    Alternative hypothesis: The expected values of the population
    distributions underlying the samples `x_1` and `x_2` are not equal.
    For details about Welch's test and its assumptions see
    Loveland (2011, p. 48, 101) and section 10.2 in Devore (2018).

    Parameters
    ----------
    x_1 : list of float
        First sample. The sample must not contain NAs.
    x_2 : list of float
        Second sample. The sample must not contain NAs.
    alpha : float
        Nominal significance level of the test.
    conf_level_effect: float, optional
        Can be used to set the nominal confidence level
        `conf_int_effect['conf_level']` of the
        confidence interval for the true effect size in the
        output to a desired value. See there for details.
        Default None.

    Returns
    -------
    dict
        Dictionary with the following fields:

        n_1 : int
            Size of the sample `x_1`.
        n_2 : int
            Size of the sample `x_2`.
        mean_1 : float
            Mean of the sample `x_1`.
        mean_2 : float
            Mean of the sample `x_2`.
        sd_1 : float
            Corrected sample standard deviation of sample `x_1`
            with denominator `n_1-1`.
        sd_2 : float
            Corrected sample standard deviation of sample `x_2`
            with denominator `n_2-1`.
        decision : str
            Decision resulting from the test at the given
            significance level `alpha`.
        est_effect : float
            Estimated effect, i.e. difference of the two sample means:
            `est_effect = mean_1-mean_2`
        conf_int_effect : dict of three float
            Contains the lower limit `ll`, the upper limit `ul` and the nominal
            confidence level `conv_level` of the confidence interval for the
            true effect, i.e. for the difference `mu_1-mu_2` of the expected
            values of the distributions underlying the samples `x_1` and `x_2`.
            If the input parameter `conf_level_effect` is None, we have
            `conv_level = 1-alpha`, else `conv_level = conf_level_effect`.
        alpha : float
            Nominal significance level of the test.
        p_value : float
            p-value resulting from the application of the test
            to the samples `x_1` and `x_2`.
        t : float
            Value of the test statistic of Welch's test for the
            samples `x_1` and `x_2`.
        df : float
            Degrees of freedom of the approximated t distribution
            of the test statistics of Welch's test. This value depends
            on the samples `x_1` and `x_2`.

    Examples
    ---------
    >>> import vstats
    >>>
    >>> sample_1 = [
    ...     91.69768212, 111.9563377, 107.83704558, 104.80731265,
    ...     97.29852169, 100.07073963, 100.28847412, 96.96491787,
    ...     96.5376013, 93.91064332, 96.17914335, 102.13996333,
    ...     97.40572285, 96.84834019, 99.51065002, 103.77422803,
    ...     106.27652877, 95.75790182, 96.72876759, 97.1026353
    ... ]
    >>> sample_2 = [
    ...     91.43722234, 102.14238579, 91.75545681, 110.08756459,
    ...     108.99524315, 102.77401765, 91.89613184, 107.32784105,
    ...     100.24714189, 114.40913719, 99.53071592, 107.54587797
    ... ]
    >>> vstats.welchs_test(
    ...     x_1=sample_1,
    ...     x_2=sample_2,
    ...     alpha=0.05
    ... )
    {'n_1': 20,
     'n_2': 12,
     'mean_1': 99.65465786149998,
     'mean_2': 102.34572801583333,
     'sd_1': 5.06019026149053,
     'sd_2': 7.713255846118943,
     'decision': 'The null hypothesis is not rejected at the significance level 0.05.',
     'est_effect': -2.691070154333346,
     'conf_int_effect': {'ll': -7.9661770911439005,
     'ul': 2.5840367824772086,
     'conf_level': 0.95},
     'alpha': 0.05,
     'p_value': 0.2965367701571776,
     't': -1.0774511848851365,
     'df': 16.767292820078413}

    References
    ----------
    Devore, Jay L. & Berk, Kenneth N. (2018). Modern Mathematical Statistics
    with Applications. 2nd ed. New York, Dordrecht, Heidelberg, London: Springer.
    ISBN: 978-1-4614-0390-6.

    Loveland, Jennifer L. (2011). Mathematical Justification of Introductory
    Hypothesis Tests and Development of Reference Materials.
    All Graduate Plan B and other Reports. 14.
    https://digitalcommons.usu.edu/gradreports/14
    """  # noqa: E501
    n_1 = len(x_1)
    n_2 = len(x_2)

    var_1 = var(x_1, ddof=1)  # sample variance with divisor n-1
    var_2 = var(x_2, ddof=1)

    sd_1 = sqrt(var_1)  # only needed for output
    sd_2 = sqrt(var_2)

    # According to Theorem 24 in Loveland (2011, p. 48):
    df = float(
        (var_1/n_1 + var_2/n_2)**2 / ((var_1**2/n_1**2)/(n_1-1)
                                      + (var_2**2/n_2**2)/(n_2-1)
                                      )
    )

    # Calculate confidence interval for the true effect:
    if conf_level_effect is None:
        conf_level_effect = 1-alpha
    # (0.5+conf_level_effect/2)-quantil:
    conf_int_effect_width_factor = (
        stats.t(df=df).ppf(0.5 + conf_level_effect/2)
    )
    mean_1 = float(mean(x_1))
    mean_2 = float(mean(x_2))
    conf_int_effect_lower_limit = (
        (mean_1-mean_2)
        - conf_int_effect_width_factor*sqrt(var_1/n_1 + var_2/n_2)
    )
    conf_int_effect_upper_limit = (
        (mean_1-mean_2)
        + conf_int_effect_width_factor*sqrt(var_1/n_1 + var_2/n_2)
    )
    conf_int_effect = {
        "ll": conf_int_effect_lower_limit,
        "ul": conf_int_effect_upper_limit,
        "conf_level": conf_level_effect
    }

    # value of t statistic for the given data:
    t = (mean_1-mean_2) / sqrt(var_1/n_1 + var_2/n_2)

    # Below sf is the survival function.
    # sf = (1 - cfd), but sf is sometimes more accurate than cfd! Cf.
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.t.html
    p_value = (stats.t(df=df).sf(abs(t)))*2  # By symmetry of t distribution.

    if p_value <= alpha:
        decision = ("The null hypothesis is rejected at the significance"
                    f" level {alpha}."
                    )
    else:
        decision = ("The null hypothesis is not rejected at the significance"
                    f" level {alpha}."
                    )

    result = {
        "n_1": n_1,
        "n_2": n_2,
        "mean_1": mean_1,
        "mean_2": mean_2,
        "sd_1": sd_1,
        "sd_2": sd_2,
        "decision": decision,
        "est_effect": mean_1-mean_2,
        "conf_int_effect": conf_int_effect,
        "alpha": alpha,
        "p_value": p_value,
        "t": t,
        "df": df
    }

    return result
