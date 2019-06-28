from unittest import TestCase
from deep_hebbian_utils.onehot import OneHot
from dateutil import parser

class TestOneHotGetDate(TestCase):

    def test_get_date(self):

        test_date = parser.parse("2018-11-01 16:45")
        result = OneHot.get_date(test_date)
        self.assertEqual(result, '0000111' + '0' * 33)
