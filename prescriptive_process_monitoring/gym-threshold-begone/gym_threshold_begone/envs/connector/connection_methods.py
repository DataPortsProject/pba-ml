from enum import Enum


class ConnectionMethods(str, Enum):
    FIFO = "fifo"
    Socket = "socket"
    ModelFile = "model_file"
