import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from gym import spaces
from gym_threshold_begone.envs._baseenv import BaseEnv


class NonPropheticCuriosity(BaseEnv):
    metadata = {'render.modes': ['human']}
    summary_writer = None
    reward_success = 1.5
    reward_multiplier = 1.0

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        #################################
        # Parameter fuer das Environment
        #################################
        self.action_space = spaces.Discrete(2)  # set action space to adaption true or false
        # Hier dimensionen des state-arrays anpassen:
        #        low_array = np.array([0, 0, 0])
        #        high_array = np.array([np.finfo(np.float32).max, np.finfo(np.float32).max, np.finfo(np.float32).max])

        # state = rel. position, reliability, pred. deviation, avg. adapted
        self.avg_adapted = 0.5  # computed in a faster ghetto way
        self.avg_non_adapt_true = 0.5
        low_array = np.array([0., 0., np.finfo(np.float32).min, 0., 0.])
        high_array = np.array([1., np.finfo(np.float32).max, np.finfo(np.float32).max, 1., 1.])
        self.observation_space = spaces.Box(low=low_array, high=high_array)

    def step(self, action=None):
        if action is None:
            action = 0

        self.send_action(int(action))
        self.action_value = action

        self.receive_reward_and_state()

        self.do_logging(action)  # log with the old state to accurately calculate the action probabilities

        info = {}
        self.state = self._create_state()

        return self.state, self.reward, self.done, info

    def compute_reward(self, adapted, cost, done, predicted_duration, planned_duration, reliability, position,
                       process_length, actual_duration=0.):
        if not done:
            reward = 0.
        else:
            # if adapted:  # THIS IS NOT AT ALL AVERAGING! PLS FIX! But it might work better than actual averaging?
            #     if self.avg_adapted < 1.:
            #         self.avg_adapted += 0.01
            # else:
            #     if self.avg_adapted > 0.:
            #         self.avg_adapted -= 0.01

            temp = self.adapt_in_ep[-100:]
            length_temp = len(temp)
            if length_temp == 0:
                self.avg_adapted = 1
            else:
                self.avg_adapted = sum(temp) / length_temp

            temp2 = self.true_per_ep[-100:]  # replacing the algorithm from below thad does not compute an average
            for i in range(len(temp)):
                if temp[len(temp) - i - 1] == 1:
                    temp2.pop(len(temp) - i - 1)
            length_temp2 = len(temp2)
            if length_temp2 == 0:
                self.avg_non_adapt_true = 0
            else:
                self.avg_non_adapt_true = sum(temp2) / length_temp2

            trueness_mod = max(min(((-1 * (self.avg_non_adapt_true - 0.5)) + 0.2) * 5. * 6., 6.), 0.)
            # factor calculated from true avg, where it ranges from 0 to 6 where true avg ranges from 0.7 to 0.5

            curiosity_reward_mod = max(self.avg_adapted - 0.5, 0.) * trueness_mod
            alpha = 1. - (((1. - 0.5) / process_length) * position)
            alpha_adjustet_for_curiosity = alpha * (
                    1. - curiosity_reward_mod)  # turning around the derivative of alpha in case of high curiosity,
            # thereby pushing the agent towards finishing cases, if he doesn't do so already
            violation = actual_duration > planned_duration
            reward = 0.
            if adapted:
                reward -= 0.5  # Adaptation costs something
                reward += alpha_adjustet_for_curiosity  # With alpha probability the adaption is successful
                # (the situation, where no adaptation is nessecary is not taken to account)
            else:
                if violation:
                    # if self.avg_non_adapt_true > 1.:
                    #     self.avg_non_adapt_true -= 0.01
                    reward = -1.  # Pain, suffering, everyone dies!
                else:
                    # if self.avg_non_adapt_true < 1.:
                    #     self.avg_non_adapt_true += 0.01
                    reward = NonPropheticCuriosity.reward_success  # Great! Best case scenario!

        return reward * NonPropheticCuriosity.reward_multiplier

    def reset(self):
        self.send_action(-1)
        self.receive_reward_and_state()

        self.state = self._create_state()

        return self.state

    def _create_state(self):
        relative_position = self.position / self.process_length
        prediction_deviation = (self.predicted_duration - self.planned_duration) / self.planned_duration
        self.state = np.array(
            [relative_position, self.reliability, prediction_deviation, self.avg_adapted, self.avg_non_adapt_true])
        return self.state

    def render(self, mode='human'):
        # we don't need this
        return
