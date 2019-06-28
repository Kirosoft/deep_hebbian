from datetime import datetime, timedelta, tzinfo
from dateutil.parser import parse

class OneHot:

    def __init__(self):
        print("init")

    @staticmethod
    def clamp(n, smallest, largest):
        return max(smallest, min(n, largest))

    @staticmethod
    def vec_from_pos(pos, value_width, width):

        is_odd = 1 if value_width % 2 != 0 else 0

        if pos < 0 or pos >= width:
            raise AssertionError('pos must be in the range 0 < pos < width')
        if value_width >= width:
            raise AssertionError('value width must be smaller than the width')

        # calculate left/right 0 boundaries
        half_width = int((value_width - is_odd) / 2)
        left_pos = pos - half_width
        right_pos = pos + half_width

        # clip for the edges
        left_pos = 0 if left_pos < 0 else left_pos
        right_pos = width - is_odd if right_pos >= width else right_pos

        # workout the left/right overlaps for clipping against either edge
        overlap_left = pos - half_width
        overlap_right = (width - is_odd) - (pos + half_width)

        value_width_left = half_width + overlap_left if overlap_left < 0 else half_width
        value_width_right = half_width + overlap_right if overlap_right < 0 else half_width

        return ('0' * left_pos) + (('1' * value_width_left) + ('1'*is_odd) + ('1' * value_width_right)) \
            + ('0' * ((width - 1) - right_pos)) + ('0' * (1 - is_odd))

    @staticmethod
    def scalar_params(value, min_val=0, max_val=40, width=80, overlap=0):
        range_val = max_val - min_val
        if range_val >= width:
            raise AssertionError('Width of the vector must be larger than the value range')
        if value >= max_val:
            raise AssertionError('Value must be less than max')
        if value < min_val:
            raise AssertionError('Value must be larger or equal to min')

        overlap_offset = (value - min_val) * overlap
        value_width = int((width + ((range_val - 2) * overlap)) / range_val)
        pos = (value_width * (value - min_val))
        is_odd = 1 if value_width % 2 != 0 else 0
        offset = int((value_width - is_odd) / 2) - 1 + is_odd

        return pos + offset - overlap_offset, value_width, width

    @staticmethod
    def get_time_of_day_from_filetime(value, width=50, overlap=0):

        dt = datetime.utcfromtimestamp(value / 1000)

        return OneHot.get_time_of_day(dt.isoformat(), width, overlap)


    @staticmethod
    def get_time_of_day_enum(value):

        dt = datetime.utcfromtimestamp(value / 1000)

        date = parse(dt.isoformat())

        hour = date.time().hour

        if hour < 8:
            result = 0
        elif hour < 12:
            result = 1
        elif hour < 14:
            result = 2
        elif hour < 18:
            result = 3
        else:
            result = 4

        return result

    @staticmethod
    def get_duration_scalar(value, width=50):

        return OneHot.get_scalar(value, width, 0, 100000)

    @staticmethod
    def get_time_of_day(value, width=50, overlap=0):

        date = parse(value)

        hour = date.time().hour

        if hour < 8:
            result = 0
        elif hour < 12:
            result = 1
        elif hour < 14:
            result = 2
        elif hour < 18:
            result = 3
        else:
            result = 4

        return OneHot.get_scalar(result, width, 0, 5, overlap)

    @staticmethod
    def get_boolean(value, width=10):
        pos, value_width, width = OneHot.scalar_params(value, 0, 2, width)
        return OneHot.vec_from_pos(pos, value_width, width)

    @staticmethod
    def get_scalar(value, width=250, min_val=0, max_val=100, overlap=2, clip=False):

        if clip:
            value = OneHot.clamp(value, min_val, max_val)

        pos, value_width, width = OneHot.scalar_params(value, min_val, max_val, width, overlap)
        return OneHot.vec_from_pos(pos, value_width, width)

    @staticmethod
    def is_weekend(value, width=20):

        date = parse(value)

        weekday = date.weekday()
        weekend = True if (weekday == 7 or weekday == 8) else False

        return OneHot.get_boolean(weekend, width)

