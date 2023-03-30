import os
from gym_threshold_begone.envs.connector._fifo_connector import FiFoConnector
from gym_threshold_begone.envs.connector._socket_connector import SocketConnector
from ._model_file_connector import ModelFileConnector
from .connection_methods import ConnectionMethods

WRITE_PATH = "../jread"
READ_PATH = "../jwrite"


def await_connection(connection_method: ConnectionMethods = ConnectionMethods.ModelFile, **kwargs):
    global WRITE_PATH
    global READ_PATH

    if connection_method == ConnectionMethods.FIFO:
        _try_fifo()
        print("Created fifo pipes! WRITE_PATH:" + os.path.abspath(WRITE_PATH) + " READ_PATH:" + os.path.abspath(
            READ_PATH))
        connector = FiFoConnector(WRITE_PATH, READ_PATH)
    elif connection_method == ConnectionMethods.Socket:
        connector = SocketConnector()
    else:
        connector = ModelFileConnector(**kwargs)
    return connector


def _try_fifo():
    global WRITE_PATH
    global READ_PATH
    if os.path.exists(WRITE_PATH):
        print("Unlinking writepath: " + os.path.abspath(WRITE_PATH))
        os.unlink(WRITE_PATH)
    if os.path.exists(READ_PATH):
        print("Unlinking readpath: " + os.path.abspath(READ_PATH))
        os.unlink(READ_PATH)

    while True:
        try:
            if not os.path.exists(WRITE_PATH):
                os.mkfifo(WRITE_PATH)
            if not os.path.exists(READ_PATH):
                os.mkfifo(READ_PATH)
            return True
        # Running this on windows bare-metal will result in an AttributeError
        except AttributeError:
            print("Failed to use fifo for IPC...")
            return False
        # Running this on windows inside docker will however raise OSErrors, but still create the pipe!?!
        except OSError:
            print("OSError!")
            if os.path.exists(WRITE_PATH):
                print("Unlinking writepath: " + os.path.abspath(WRITE_PATH))
                os.unlink(WRITE_PATH)
            if os.path.exists(READ_PATH):
                print("Unlinking readpath: " + os.path.abspath(READ_PATH))
            print("Failed to use fifo for IPC...")
            return False


def _receive_reward_and_state(filelike):
    # print("receiving reward parameters...")
    result = {}

    adapted = _readline(filelike)
    adapted = adapted.strip()
    adapted = True if adapted == 'true' else False
    result['adapted'] = adapted

    cost = _readline(filelike)
    cost = cost.strip()
    cost = float(cost)
    result['cost'] = cost

    done = _readline(filelike)
    done = done.strip()
    done = True if done == 'true' else False
    result['done'] = done

    if done:
        true = _readline(filelike)
        true = true.strip()
        true = True if true == 'true' else False
        result['true'] = true

    case_id = _readline(filelike)
    case_id = case_id.strip()
    case_id = float(case_id)
    result['case_id'] = case_id

    actual_duration = _readline(filelike)
    actual_duration = actual_duration.strip()
    actual_duration = float(actual_duration)
    result['actual_duration'] = actual_duration

    predicted_duration = _readline(filelike)
    predicted_duration = predicted_duration.strip()
    predicted_duration = float(predicted_duration)
    result['predicted_duration'] = predicted_duration

    planned_duration = _readline(filelike)
    planned_duration = planned_duration.strip()
    planned_duration = float(planned_duration)
    result['planned_duration'] = planned_duration

    reliability = _readline(filelike)
    reliability = reliability.strip()
    reliability = float(reliability)
    result['reliability'] = reliability

    position = _readline(filelike)
    position = position.strip()
    position = float(position)
    result['position'] = position

    process_length = _readline(filelike)
    process_length = process_length.strip()
    process_length = float(process_length)
    result['process_length'] = process_length

    return result


# Running this with the fifo_cownnector on windows inside docker will result in the readline() method on the filelike
# object  to not block and wait till an entire line can be read, but to return immediatly whatever is available in the
# pipe!?!
# Because I've given up on using fifo pipes when running in docker on windows, this bit of code has become useless.
# But I'm too lazy to change it back, which is why it's still here.
# UPDATE: Apparently, with the WSL2 backend, docker for windows now supports fifo pipes correctly. Sooooo, I guess the
# moral of the story is, you just have to wait long enough for your problems to fix themselves
def _readline(filelike):
    result = ""
    while not result.endswith(os.linesep):
        result += filelike.readline()
    return result
