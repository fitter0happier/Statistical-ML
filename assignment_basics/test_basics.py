import unittest

import numpy as np
from basics import *

class TestMatrixManip(unittest.TestCase):
    ''' Unittests of individual outputs of matrix_manip function from assignment "basics" '''
    def setUp(self) -> None:
        """
        It defines the same input data as from the web example
        """
        self.A = np.array([[16, 2, 3, 13],
                           [5, 11, 10, 8],
                           [9, 7, 6, 12],
                           [4, 14, 15, 1]])

        self.B = np.array([[3, 4, 9, 4, 3, 6, 6, 2, 3, 4],
                           [9, 2, 10, 1, 4, 3, 7, 1, 3, 5]])

        self.your_output = matrix_manip(self.A, self.B)

    def test_types_shapes(self):
        A = self.A
        B = self.B
        self.assertEqual(self.your_output['A_transpose'].shape, tuple(reversed(A.shape)))
        self.assertEqual(self.your_output['A_transpose'].dtype, A.dtype)

        self.assertEqual(self.your_output['A_3rd_col'].shape, (A.shape[0], 1))
        self.assertEqual(self.your_output['A_3rd_col'].dtype, A.dtype)

        self.assertEqual(self.your_output['A_slice'].shape, (2, 3))
        self.assertEqual(self.your_output['A_slice'].dtype, A.dtype)

        self.assertEqual(self.your_output['A_gr_inc'].shape, (A.shape[0], A.shape[1] + 1))
        self.assertEqual(self.your_output['A_gr_inc'].dtype, A.dtype)

        self.assertEqual(self.your_output['C'].shape, (A.shape[0], A.shape[0]))
        self.assertEqual(self.your_output['C'].dtype, A.dtype)

        self.assertEqual(type(self.your_output['A_weighted_col_sum']), float)

        self.assertEqual(self.your_output['D'].shape, (2, B.shape[1]))
        self.assertEqual(self.your_output['D'].dtype, B.dtype)

        self.assertEqual(len(self.your_output['D_select'].shape), 2)
        self.assertEqual(self.your_output['D_select'].shape[0], 2)
        self.assertEqual(self.your_output['D_select'].dtype, B.dtype)

    def test_A_transpose(self):
        A_transpose = np.array([[16,  5,  9,  4],
                                [ 2, 11,  7, 14],
                                [ 3, 10,  6, 15],
                                [13,  8, 12,  1]])
        np.testing.assert_array_equal(self.your_output['A_transpose'],
                                      A_transpose)

    def test_A_3rd_col(self):
        A_3rd_col = np.array([[3], [10], [6], [15]])
        np.testing.assert_array_equal(self.your_output['A_3rd_col'],
                                      A_3rd_col)

    def test_A_slice(self):
        A_slice = np.array([[7, 6, 12], [14, 15, 1]])
        np.testing.assert_array_equal(self.your_output['A_slice'],
                                      A_slice)

    def test_A_gr_inc(self):
        A_gr_inc = np.array([[17, 2, 3, 14, 1],
                             [6, 12, 11, 9, 1],
                             [10, 8, 7, 13, 1],
                             [5, 15, 16, 1, 1]])
        np.testing.assert_array_equal(self.your_output['A_gr_inc'],
                                      A_gr_inc)

    def test_C(self):
        C = np.array([[499, 286, 390, 178],
                      [286, 383, 351, 396],
                      [390, 351, 383, 296],
                      [178, 396, 296, 508]])

        np.testing.assert_array_equal(self.your_output['C'],
                                      C)

    def test_A_weighted_col_sum(self):
        A_weighted_col_sum = 391
        np.testing.assert_array_equal(self.your_output['A_weighted_col_sum'],
                                      A_weighted_col_sum)

    def test_D(self):
        D = np.array([[-1, 0, 5, 0, -1, 2, 2, -2, -1, 0],
                      [3, -4, 4, -5, -2, -3, 1, -5, -3, -1]])
        np.testing.assert_array_equal(self.your_output['D'],
                                      D)

    def test_D_select(self):
        D_select = np.array([[0, 5, 0, -2],
                             [-4, 4, -5, -5]])
        np.testing.assert_array_equal(self.your_output['D_select'],
                                      D_select)


class TestComputeLetterMean(unittest.TestCase):
    def setUp(self) -> None:
        loaded_data = np.load("data_33rpz_basics.npz")
        self.alphabet = loaded_data["alphabet"]
        self.images = loaded_data["images"]
        self.labels = loaded_data["labels"]

    def test_types_shapes(self):
        letter_mean = compute_letter_mean('R', self.alphabet, self.images, self.labels)
        self.assertEqual(letter_mean.dtype, np.uint8)
        self.assertEqual(letter_mean.shape, self.images.shape[:2])

    def test_compute_letter_mean(self):
        gt_letter_mean = np.array([106,  49,  15,   7,   9,  29,  75, 134, 187, 224], dtype=np.uint8)
        letter_mean = compute_letter_mean('R', self.alphabet, self.images, self.labels)
        np.testing.assert_array_equal(letter_mean[5,:], gt_letter_mean)


class TestComputeLRFeatures(unittest.TestCase):
    def setUp(self) -> None:
        loaded_data = np.load("data_33rpz_basics.npz")
        self.alphabet = loaded_data["alphabet"]
        self.images = loaded_data["images"]
        self.labels = loaded_data["labels"]

    def test_compute_lr_features(self):
        gt_features = np.array([  120,  1223,  -144,  -161,   197, -2921,  -998,  -944,  -120,
                                 -304,  -884, -1461, -1233,  1444,  1705,  1332,   881,   212,
                                   92,   319, -3104, -2829,   255,     1, -1763,  2230,  1916,
                                 -335,  -257, -3568, -5204, -1144,  -641,   525,   182,  -768,
                                 -844,  1536,  1139,   522,   495,   353,  -251,  1345,   439,
                                 1114, -2087,  -107,  -563,  1491, -1935, -1640,  1979,  2215,
                                  906,  1726,  1332,   365,   825,  2776,  1282,   708,  1010,
                                  429,  1141,  1145,  1896,     7,  -642,  -657,    36,   368,
                                 1079,    79,  -483,   327,  -135,   888,  2270,  2211,  3860,
                                 1248,  1371,  -857,   100,  -134,  -946,  1954,  1979, -1575,
                                 -837,  1363,   803,   546, -1916, -1808,   370,  -435,  -363,
                                  497])
        features = compute_lr_features('A', self.alphabet, self.images, self.labels)
        np.testing.assert_array_equal(features, gt_features)

if __name__ == "__main__":
    unittest.main()