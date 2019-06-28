from deep_hebbian_utils.onehot import OneHot


def is_valid_message(message):
    valid_message_codes = [1009,1010,1011,1012,1033,1034,1035,1036,1037,1047]
    result = False

    for msg in message['EVENT_HIST'].keys():
        msgCode = int(msg)
        result = msgCode in valid_message_codes
        if result:
            break

    return result


def pre_process_message(inputData):
    fail_messages = [1033,1034,1035,1036,1037]
    success_messages = [1009,1010,1011,1012,1047]

    # add duration
    end = inputData['W_END']
    start = inputData['W_START']
    inputData['EVENT_DURATION'] = end - start
    inputData['SUCCESS_EVENT_COUNT']= min(9,sum([ inputData['EVENT_HIST'][mc] for mc in inputData['EVENT_HIST'].keys() if int(mc) in success_messages]))
    inputData['FAIL_EVENT_COUNT']= min(9,sum([ inputData['EVENT_HIST'][mc] for mc in inputData['EVENT_HIST'].keys() if int(mc) in fail_messages]))
    inputData['TIME_OF_DAY_ENUM'] = OneHot.get_time_of_day_enum(start)
    return inputData

