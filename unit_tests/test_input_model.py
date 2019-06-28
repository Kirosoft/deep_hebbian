from unittest import TestCase
from deep_hebbian_utils.inputmodel import InputModel


class TestInputModel(TestCase):

    def test_get_model_params(self):

        input_data = {
            "date_time": "2018-11-02 19:16",
            "new_user": True,
            "ad_success": 2,
            "ad_fail": 5,
            "two_factor_success": 1,
            "two_factor_fail": 2
        }
        input_vector, input_width = InputModel.get_model_params(".\model.json", input_data)

        self.assertEqual(input_width, 490)
