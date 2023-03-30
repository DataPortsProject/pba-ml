import os
from gym_threshold_begone.envs.connector import environment_connector
from gym_threshold_begone.envs.connector._base_connector import BaseConnector


class FiFoConnector(BaseConnector):

    def __init__(self, pathwrite, pathread):
        self.pathwrite = pathwrite
        self.pathread = pathread
        self.writepipe = open(pathwrite, "w")
        self.readpipe = open(pathread, "r")

    def send_action(self, action):
        # print("sending action...")
        self.writepipe.write(str(action) + os.linesep)
        self.writepipe.flush()
        # print("action sent: " + str(action))

    def receive_reward_and_state(self):
        return environment_connector._receive_reward_and_state(self.readpipe)

    def close(self):
        self.writepipe.close()
        self.readpipe.close()
        os.unlink(self.pathwrite)
        os.unlink(self.pathread)
