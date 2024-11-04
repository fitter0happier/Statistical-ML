import unittest

import numpy as np
import bayes

data = np.load("data_33rpz_bayes.npz", allow_pickle=True)


class TestBayesDiscrete(unittest.TestCase):
    def setUp(self):
        self.discreteA = {'Prior': 0.6153846153846154,
                          'Prob': np.array([0.0125, 0., 0., 0.0125, 0.025, 0.0125,
                                            0.025, 0.0375, 0.075, 0.1, 0.2125, 0.1375, 0.15,
                                            0.1, 0.0875, 0.0125, 0., 0., 0., 0., 0.])}
        self.discreteC = {'Prior': 0.38461538461538464,
                          'Prob': np.array([0., 0., 0., 0.02, 0.02, 0.22, 0.46,
                                            0.16, 0.1, 0.02, 0., 0., 0., 0., 0., 0., 0., 0.,
                                            0., 0., 0.])}

    def test_bayes_risk_discrete(self):
        q_discrete = np.array([0] * 10 + [1] + [0] * 10)
        W = np.array([[0, 1], [1, 0]])
        R_discrete = bayes.bayes_risk_discrete(self.discreteA, self.discreteC, W, q_discrete)
        np.testing.assert_almost_equal(R_discrete, 0.5153846153846154)

    def test_find_strategy_discrete_example_1(self):
        W = np.array([[0, 1], [1, 0]])
        q_discrete = bayes.find_strategy_discrete(self.discreteA, self.discreteC, W)
        np.testing.assert_array_equal(q_discrete, [0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])

    def test_find_strategy_discrete_example_2(self):
        distribution1 = {}
        distribution2 = {}
        distribution1['Prior'] = 0.3
        distribution2['Prior'] = 0.7
        distribution1['Prob'] = np.array([0.2, 0.3, 0.4, 0.1])
        distribution2['Prob'] = np.array([0.5, 0.4, 0.1, 0.0])
        W = np.array([[0, 1], [1, 0]])
        q = bayes.find_strategy_discrete(distribution1, distribution2, W)
        np.testing.assert_array_equal(q, [1, 1, 0, 0])

    def test_classification_error(self):
        W = np.array([[0, 1], [1, 0]])
        q_discrete = bayes.find_strategy_discrete(self.discreteA, self.discreteC, W)
        images_test = data["images_test"]
        labels_test = data["labels_test"]
        measurements_discrete = bayes.compute_measurement_lr_discrete(images_test)
        labels_estimated_discrete = bayes.classify_discrete(measurements_discrete, q_discrete)
        error_discrete = bayes.classification_error(labels_estimated_discrete, labels_test)

        np.testing.assert_almost_equal(error_discrete, 0.225)


class TestBayes2Norm(unittest.TestCase):
    def setUp(self) -> None:
        self.contA = {'Mean': 124.2625,
                      'Sigma': 1434.45420083,
                      'Prior': 0.61538462}
        self.contC = {'Mean': -2010.98,
                      'Sigma': 558.42857106,
                      'Prior': 0.38461538}
        self.q_cont = {'t2': -1248.7684903033442, 't1': -3535.997150276402, 'decision': np.array([0, 1, 0], dtype=np.int32)}

    def test_find_strategy_2norm_CW(self):
        q_cont = bayes.find_strategy_2normal(self.contA, self.contC)
        np.testing.assert_array_equal(q_cont['decision'], np.array([0, 1, 0], dtype=np.int32))
        np.testing.assert_almost_equal(q_cont['t1'], -3535.997150276402)
        np.testing.assert_almost_equal(q_cont['t2'], -1248.7684903033442)

    def test_bayes_risk_2normal_CW(self):
        R_cont = bayes.bayes_risk_2normal(self.contA, self.contC, self.q_cont)
        np.testing.assert_almost_equal(R_cont, 0.13519281686757106)

    def test_classification_error_CW(self):
        images_test = data["images_test"]
        labels_test = data["labels_test"]

        measurements_cont = bayes.compute_measurement_lr_cont(images_test)
        labels_estimated_cont = bayes.classify_2normal(measurements_cont, self.q_cont)
        error_cont = bayes.classification_error(labels_estimated_cont, labels_test)

        np.testing.assert_almost_equal(error_cont, 0.1)


if __name__ == "__main__":
    unittest.main()
