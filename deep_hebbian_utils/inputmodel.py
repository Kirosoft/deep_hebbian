import json
from deep_hebbian_utils.onehot import OneHot


class InputModel:

    def __init__(self, filename):
        self.model= self.read_model(filename)

    def read_model(self, filename):

        with open(filename) as f:
            data = json.load(f)

        return data

    def get_model_params(self, data_in):
        output_vector = ""
        total_width = 0

        for x in self.model["model"]:
            param_dict = {
                "value": data_in[x["input"]]
            }
            if "output_width" in x:
                param_dict["width"] = x["output_width"]
            if "output_overlap" in x:
                param_dict["overlap"] = x["output_overlap"]
            if "output_min" in x:
                param_dict["min_val"] = x["output_min"]
            if "output_max" in x:
                param_dict["max_val"] = x["output_max"]
            if "clamp" in x:
                param_dict["clamp"] = x["clamp"]

            func = getattr(OneHot, x["coding_function"])
            v  = func(**param_dict)
            total_width += len(v)
            output_vector += v

        return output_vector, total_width
