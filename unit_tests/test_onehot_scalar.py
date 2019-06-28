from unittest import TestCase
from deep_hebbian_utils.onehot import OneHot


class TestOnehotScalar(TestCase):

    def test_from_scalar_value_too_large(self):
        self.assertRaises(AssertionError, OneHot.scalar_params, 50, 0, 50, 150)

    def test_from_scalar_value_too_small(self):
        self.assertRaises(AssertionError, OneHot.scalar_params, -1, 0, 50, 150)

    def test_from_scalar_width_too_small(self):
        self.assertRaises(AssertionError, OneHot.scalar_params, 20, 0, 51, 50)

    def test_from_scalar(self):
        pos, value_width, width = OneHot.scalar_params(20, 0, 50, 150)
        offset = int((value_width-1)/2)
        self.assertEqual(pos, int((150/(50-0))*20)+offset)
        self.assertEqual(value_width, 3)

    def test_from_scalar1(self):
        pos, value_width, width = OneHot.scalar_params(0, 0, 50, 150)
        offset = int((value_width-1)/2)
        self.assertEqual(pos, offset)

    def test_from_scalar2(self):
        pos, value_width, width = OneHot.scalar_params(49, 0, 50, 150)
        offset = int((value_width-1)/2)
        self.assertEqual(pos, int((150/(50-0))*49)+offset)

    def test_from_scalar3(self):
        pos, value_width, width = OneHot.scalar_params(50, 1, 51, 150)
        offset = int((value_width-1)/2)
        self.assertEqual(pos, int((150/(51-1))*(50-1))+offset)

    def test_from_scalar4(self):
        pos, value_width, width = OneHot.scalar_params(1, 1, 51, 150)
        offset = int((value_width - 1) / 2)
        self.assertEqual(pos, int((150/(51-1))*(1-1))+offset)

    def test_from_scalar_overlap0(self):
        pos, value_width, width = OneHot.scalar_params(0, 0, 50, 152, 1)

        self.assertEqual(pos, 1)

    def test_from_scalar_overlap1(self):
        pos, value_width, width = OneHot.scalar_params(1, 0, 50, 152, 1)

        self.assertEqual(pos, 4)

    def test_from_scalar_overlap2(self):
        pos, value_width, width = OneHot.scalar_params(2, 0, 50, 152, 1)

        self.assertEqual(pos, 7)

    def test_from_scalar_overlap3(self):
        pos, value_width, width = OneHot.scalar_params(2, 0, 50, 300, 2)

        self.assertEqual(pos, (value_width* 2) + ((value_width-1)/2) - (2 * 2))

    def test_from_scalar_overlap4_odd(self):
        pos, value_width, width = OneHot.scalar_params(2, 0, 50, 325, 2)

        self.assertEqual(pos, (value_width* 2) + ((value_width)/2)-1 - (2 * 2))
