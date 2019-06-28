from unittest import TestCase
from deep_hebbian_utils.onehot import OneHot

class TestOnehot(TestCase):

    def test_vec_from_pos_middle(self):
        result = OneHot.vec_from_pos(5, 3, 40)
        self.assertEqual(result, '0000111' + '0' * 33)

    def test_vec_from_pos_left(self):
        result = OneHot.vec_from_pos(0, 3, 40)
        self.assertEqual(result, '11'+'0' * 38)

    def test_vec_from_pos_right(self):
        result = OneHot.vec_from_pos(39, 3, 40)
        self.assertEqual(result, '0' * 38 + '11')

    def test_vec_from_pos_middle_1(self):
        result = OneHot.vec_from_pos(5, 3, 37)
        self.assertEqual(result, '0000111' + '0' * 30)

    def test_vec_from_pos_left_1(self):
        result = OneHot.vec_from_pos(0, 3, 37)
        self.assertEqual(result, '11'+'0' * 35)

    def test_vec_from_pos_right_1(self):
        result = OneHot.vec_from_pos(33, 3, 37)
        self.assertEqual(result, '0' * 32 + '111' + '0' * 2)

    def test_vec_from_pos_even_1(self):
        result = OneHot.vec_from_pos(5, 2, 37)
        self.assertEqual(result, '0' * 4 + '11' + '0' * 31)

    def test_vec_from_pos_invalid_pos(self):
        self.assertRaises(AssertionError, OneHot.vec_from_pos, -1, 4, 40)

    def test_vec_from_pos_invalid_pos1(self):
        self.assertRaises(AssertionError, OneHot.vec_from_pos, 40, 4, 40)

    def test_vec_from_pos_invalid(self):
        self.assertRaises(AssertionError, OneHot.vec_from_pos, 0, 10, 10)
