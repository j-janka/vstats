import numpy as np
import pandas as pd
from scipy import stats

import vstats


def test_get_est_welchs_test_properties_1():
    actual = vstats.get_est_welchs_test_properties(
        n=[40],
        share_n_group_1=0.2,
        alpha_test=0.05,
        rv_1=stats.norm(loc=50, scale=7),
        rv_2=stats.norm(loc=50, scale=3),
        conf_level_rejection_prob=0.95,
        n_simulation=10000,
        # Reproducible across Numpy versions:
        rng=np.random.RandomState(1653)
    )

    expected = pd.DataFrame(
        {
            "n": [40],
            "n_1": [8],
            "n_2": [32],
            "rejection_prob_est": [0.051],
            "ll_conf_int_rejection_prob": [0.046769417099724646],
            "ul_conf_int_rejection_prob": [0.05549442726462064]
        }
    )

    pd.testing.assert_frame_equal(
        actual,
        expected,
        check_dtype=False
    )


def test_get_est_welchs_test_properties_2():
    actual = vstats.get_est_welchs_test_properties(
        n=[40],
        share_n_group_1=0.2,
        alpha_test=0.05,
        rv_1=stats.rv_discrete(values=([1, 2, 3], [0.2, 0.2, 0.6])),
        rv_2=stats.rv_discrete(values=([1, 2, 3], [0.8, 0.1, 0.1])),
        conf_level_rejection_prob=0.95,
        n_simulation=10000,
        fix_degenerated_samples=True,
        # Reproducible across Numpy versions:
        rng=np.random.RandomState(1653)
    )

    expected = pd.DataFrame(
        {
            "n": [40],
            "n_1": [8],
            "n_2": [32],
            "rejection_prob_est": [0.8363],
            "ll_conf_int_rejection_prob": [0.8289],
            "ul_conf_int_rejection_prob": [0.843503],
            "share_fixed_samples_group_1": [0.02],
            "share_fixed_samples_group_2": [0.00]
        }
    )

    pd.testing.assert_frame_equal(
        actual,
        expected,
        check_dtype=False
    )


def test_welchs_test_1():
    sample_1 = [
        91.69768212, 111.9563377, 107.83704558, 104.80731265,
        97.29852169, 100.07073963, 100.28847412, 96.96491787,
        96.5376013, 93.91064332, 96.17914335, 102.13996333,
        97.40572285, 96.84834019, 99.51065002, 103.77422803,
        106.27652877, 95.75790182, 96.72876759, 97.1026353
    ]
    sample_2 = [
        91.43722234, 102.14238579, 91.75545681, 110.08756459,
        108.99524315, 102.77401765, 91.89613184, 107.32784105,
        100.24714189, 114.40913719, 99.53071592, 107.54587797
    ]
    actual = vstats.welchs_test(
        x_1=sample_1,
        x_2=sample_2,
        alpha=0.05
    )

    expected = {
        'n_1': 20,
        'n_2': 12,
        'mean_1': 99.65465786149998,
        'mean_2': 102.34572801583333,
        'sd_1': 5.06019026149053,
        'sd_2': 7.713255846118943,
        'decision': ('The null hypothesis is not rejected '
                     'at the significance level 0.05.'
                     ),
        'est_effect': -2.691070154333346,
        'conf_int_effect': {'ll': -7.9661770911439005,
                            'ul': 2.5840367824772086,
                            'conf_level': 0.95},
        'alpha': 0.05,
        'p_value': 0.2965367701571776,
        't': -1.0774511848851365,
        'df': 16.767292820078413
    }

    assert actual == expected


def test_welchs_test_2():
    sample_1 = [
        91.69768212, 111.9563377, 107.83704558, 104.80731265,
        97.29852169, 100.07073963, 100.28847412, 96.96491787,
        96.5376013, 93.91064332, 96.17914335, 102.13996333,
        97.40572285, 96.84834019, 99.51065002, 103.77422803,
        106.27652877, 95.75790182, 96.72876759, 97.1026353
    ]
    sample_2 = [
        91.43722234, 102.14238579, 91.75545681, 110.08756459,
        108.99524315, 102.77401765, 91.89613184, 107.32784105,
        100.24714189, 114.40913719, 99.53071592, 107.54587797
    ]
    actual = vstats.welchs_test(
        x_1=sample_1,
        x_2=sample_2,
        alpha=0.05,
        conf_level_effect=0.8
    )

    expected = {
        'n_1': 20,
        'n_2': 12,
        'mean_1': 99.65465786149998,
        'mean_2': 102.34572801583333,
        'sd_1': 5.06019026149053,
        'sd_2': 7.713255846118943,
        'decision': ('The null hypothesis is not rejected '
                     'at the significance level 0.05.'
                     ),
        'est_effect': -2.691070154333346,
        'conf_int_effect': {'ll': -6.023223029167781,
                            'ul': 0.6410827205010894,
                            'conf_level': 0.8},
        'alpha': 0.05,
        'p_value': 0.2965367701571776,
        't': -1.0774511848851365,
        'df': 16.767292820078413
    }

    assert actual == expected
