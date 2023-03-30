import socket
import os
from gym_threshold_begone.envs.connector import environment_connector
from gym_threshold_begone.envs.connector._base_connector import BaseConnector


class SocketConnector(BaseConnector):

    def __init__(self):
        s = socket.socket()

        port = 1337

        s.bind(('', port))

        # put the socket into listening mode
        s.listen(5)
        print("socket is listening")

        self.c, addr = s.accept()
        print('Got connection from', addr)
        s.close()

        self.net = self.c.makefile("rw", 65536, newline=os.linesep)

    def send_action(self, action):
        # print("sending action...")
        self.net.writelines([str(action) + os.linesep])
        self.net.flush()
        # print("action sent: " + str(action))

    def receive_reward_and_state(self):
        return environment_connector._receive_reward_and_state(self.net)

    def close(self):
        self.c.close()
        self.net.close()
