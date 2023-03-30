import os
from pathlib import Path
import datetime
import pandas as pd

import gym
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from gym_threshold_begone.envs.connector import environment_connector


# import logging
# import sys
# logging.basicConfig(filename='test.log', level=logging.DEBUG)
# logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))


def get_average_last_entries_from_numeric_list(numeric_list: list, max_entries):
    return sum(numeric_list[-max_entries:]) / max_entries


def get_average_last_entries_from_numeric_list_excluding(numeric_list: list, max_entries, excluded_value, end_index):
    temp_list = np.array(numeric_list[np.maximum(0, end_index - max_entries):end_index + 1].copy())
    temp_list = temp_list[temp_list != excluded_value]

    if len(temp_list) == 0:
        return 0
    else:
        return sum(temp_list[:]) / len(temp_list)


def calc_gliding_average_excluding(numeric_list: list, amount_averaged, excluded_values: list = []):
    result = []
    value_sum = 0.
    amount = 0
    for i in range(len(numeric_list)):
        begin_index = max(0, i - amount_averaged)
        end_index = i
        if not excluded_values.__contains__(numeric_list[end_index]):
            value_sum += numeric_list[end_index]
            amount += 1
        if begin_index - 1 >= 0:
            if not excluded_values.__contains__(numeric_list[begin_index - 1]):
                value_sum -= numeric_list[begin_index - 1]
                amount -= 1
        if amount != 0:
            result.append(value_sum / amount)
        else:
            result.append(0)
    return result


def calc_accuracy_measurements(true_positves, true_negatives, false_positives, false_negatives):
    if false_positives + false_negatives + true_positves + true_negatives == 0:
        accuracy = 0
    else:
        accuracy = (true_positves + true_negatives) / (
                false_positives + false_negatives + true_positves + true_negatives)
    if true_positves + false_negatives == 0:
        recall = 0
    else:
        recall = true_positves / (true_positves + false_negatives)
    if true_negatives + false_positives == 0:
        specificity = 0
    else:
        specificity = true_negatives / (true_negatives + false_positives)

    if true_positves + false_positives == 0:
        precision = 0
    else:
        precision = true_positves / (true_positves + false_positives)
    if true_negatives + false_negatives == 0:
        negative_predictive_value = 0
    else:
        negative_predictive_value = true_negatives / (true_negatives + false_negatives)

    balanced_accuracy = (recall + specificity) / 2.
    if true_positves + false_positives == 0 or true_positves + false_negatives == 0 or true_negatives + false_positives == 0 or true_negatives + false_negatives == 0:
        mcc = 0
    else:
        mcc = (true_positves * true_negatives - false_positives * false_negatives) / np.sqrt(float(
            (true_positves + false_positives) * (true_positves + false_negatives) * (
                    true_negatives + false_positives) * (true_negatives + false_negatives)))

    return accuracy, recall, specificity, precision, negative_predictive_value, balanced_accuracy, mcc


def calc_gliding_accuracy_measurements(true_per_episode: list, adapt_per_episode: list, amount_averaged,
                                       variable_episodes_excluded):
    accuracies = []  # (TP+TN)/(TP+TN+FP+FN)
    recalls = []  # TP/(TP+FN)
    specificities = []  # TN/(TN+FP)
    precisions = []  # TP/(TP+FP)
    negative_predictive_values = []  # TN/(TN+FN)
    balanced_accuracies = []  # (recall + specificity) /2
    mccs = []  # (TP*TN-FP*FN) / sqrt((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN))
    true_positves = 0
    true_positves_total = 0
    true_positves_variable = 0
    true_positves_2000 = 0
    true_positves_5000 = 0
    true_negatives = 0
    true_negatives_total = 0
    true_negatives_variable = 0
    true_negatives_2000 = 0
    true_negatives_5000 = 0
    false_positives = 0
    false_positives_total = 0
    false_positives_variable = 0
    false_positives_2000 = 0
    false_positives_5000 = 0
    false_negatives = 0
    false_negatives_total = 0
    false_negatives_variable = 0
    false_negatives_2000 = 0
    false_negatives_5000 = 0

    for i in range(len(true_per_episode)):
        begin_index = max(0, i - amount_averaged)
        end_index = i

        if begin_index - 1 >= 0:
            if true_per_episode[begin_index - 1]:
                if adapt_per_episode[begin_index - 1]:
                    true_positves -= 1
                else:
                    true_negatives -= 1
            else:
                if adapt_per_episode[begin_index - 1]:
                    false_positives -= 1
                else:
                    false_negatives -= 1

        if true_per_episode[end_index]:  # true
            if adapt_per_episode[end_index]:  # adapted
                true_positves += 1
                true_positves_total += 1
                if end_index < variable_episodes_excluded:
                    true_positves_variable += 1
                if end_index < 2000:
                    true_positves_2000 += 1
                if end_index < 5000:
                    true_positves_5000 += 1
            else:  # not adapted
                true_negatives += 1
                true_negatives_total += 1
                if end_index < variable_episodes_excluded:
                    true_negatives_variable += 1
                if end_index < 2000:
                    true_negatives_2000 += 1
                if end_index < 5000:
                    true_negatives_5000 += 1
        else:  # not true
            if adapt_per_episode[end_index]:
                false_positives += 1
                false_positives_total += 1
                if end_index < variable_episodes_excluded:
                    false_positives_variable += 1
                if end_index < 2000:
                    false_positives_2000 += 1
                if end_index < 5000:
                    false_positives_5000 += 1
            else:
                false_negatives += 1
                false_negatives_total += 1
                if end_index < variable_episodes_excluded:
                    false_negatives_variable += 1
                if end_index < 2000:
                    false_negatives_2000 += 1
                if end_index < 5000:
                    false_negatives_5000 += 1

        accuracy, recall, specificity, precision, negative_predictive_value, balanced_accuracy, mcc = calc_accuracy_measurements(
            true_positves, true_negatives, false_positives, false_negatives)

        accuracies.append(accuracy)
        recalls.append(recall)
        specificities.append(specificity)
        precisions.append(precision)
        negative_predictive_values.append(negative_predictive_value)

        balanced_accuracies.append(balanced_accuracy)
        mccs.append(mcc)

    accuracy_variable, recall_variable, specificity_variable, precision_variable, negative_predictive_value_variable, balanced_accuracy_variable, mcc_variable = calc_accuracy_measurements(
        true_positves_total - true_positves_variable, true_negatives_total - true_negatives_variable,
        false_positives_total - false_positives_variable, false_negatives_total - false_negatives_variable)
    accuracy_2000, recall_2000, specificity_2000, precision_2000, negative_predictive_value_2000, balanced_accuracy_2000, mcc_2000 = calc_accuracy_measurements(
        true_positves_total - true_positves_2000, true_negatives_total - true_negatives_2000,
        false_positives_total - false_positives_2000, false_negatives_total - false_negatives_2000)
    accuracy_5000, recall_5000, specificity_5000, precision_5000, negative_predictive_value_5000, balanced_accuracy_5000, mcc_5000 = calc_accuracy_measurements(
        true_positves_total - true_positves_5000, true_negatives_total - true_negatives_5000,
        false_positives_total - false_positives_5000, false_negatives_total - false_negatives_5000)
    accuracy_total, recall_total, specificity_total, precision_total, negative_predictive_value_total, balanced_accuracy_total, mcc_total = calc_accuracy_measurements(
        true_positves_total, true_negatives_total, false_positives_total, false_negatives_total)

    variables = [accuracy_variable, recall_variable, specificity_variable, precision_variable,
                 negative_predictive_value_variable,
                 balanced_accuracy_variable, mcc_variable]
    two_thousands = [accuracy_2000, recall_2000, specificity_2000, precision_2000, negative_predictive_value_2000,
                     balanced_accuracy_2000, mcc_2000]
    five_thousands = [accuracy_5000, recall_5000, specificity_5000, precision_5000, negative_predictive_value_5000,
                      balanced_accuracy_5000, mcc_5000]
    totals = [accuracy_total, recall_total, specificity_total, precision_total, negative_predictive_value_total,
              balanced_accuracy_total, mcc_total]

    return accuracies, recalls, specificities, precisions, negative_predictive_values, balanced_accuracies, mccs, variables, two_thousands, five_thousands, totals


class BaseEnv(gym.Env):
    show_graphs = False
    log_tensorboard = False
    experiment_number = 1
    experiment_name = ""
    excluded_cases_metrics = 1000

    def __init__(self, prescriptions_save_path: str = "prescriptions", **kwargs):
        self.connector = environment_connector.await_connection(**kwargs)
        self.abort = False
        self.save_path = prescriptions_save_path

        self.total_steps = 0
        self.episode_count = 0
        self.adapted_count = 0

        # learning parameter
        self.reward = 0
        self.rewards_per_episode = []
        self.action_value = 0
        self.case_id = -1
        self.actual_duration = -1
        self.predicted_duration = 0
        self.planned_duration = 0
        self.position = 0
        self.cost = 0
        self.process_length = 0
        self.done = False
        self.adapted = 0
        self.reliability = 0

        self.true = False

        self.summary_writer = None
        self.model = None
        self.state = None

        # English Motherfucker, do you speak it?
        # Metriken fuer gesamten Verlauf des Experiments, d.h. bei jedem Step:
        self.actions = []  # Action pro Step
        self.action_1_probability = []
        self.rewards = []  # Reward pro Step (-Kosten)
        self.costs = []  # Kosten pro Step (d.h. 50/length bei nicht-adapt)
        # self.violation_predicted = []  # 1 if in each step a violation is predicted
        self.adapted_no_violation = []  # 1 if adapted though no violation was predicted

        # Metriken fuer jede Episode, d.h. jeden Case:
        self.case_id_list = []
        # self.avg_action_per_ep = []  # Durchschnitt ueber alle Aktionen einer Episode
        # self.tmp_avg_action_per_ep = []
        self.avg_adapted_100 = []  # Average end of episode over the last 100, between 1(adapted) and 0(adapted)
        # self.avg_reward_per_ep = []  # Durschnittlicher Reward der Episode
        self.tmp_cumul_reward_per_ep = []
        # self.avg_cost_per_ep = []  # Definitely not the average cost per episode!
        self.tmp_cumul_cost_per_ep = []

        # lists for metrics
        self.cumul_cost_per_ep = []  # Kumulative Kosten pro Episode
        self.cumul_reward_per_ep = []  # Kumulativer Reward pro Episode
        self.true_per_ep = []  # 1 if Episode ending decision is right, 0 otherwise
        self.adapt_in_ep = []  # Info ob in Episode adaptiert wurde
        self.case_length_per_episode = []  # Länge der Episode
        self.position_of_adaptation_per_episode = []  # Position der Adaption pro Episode; -1 := keine Adaption in der Episode
        self.earliness = []  # Position der Adaption durch Länge der Episode, -1 := keine Adaption in der Episode

        self.percentage_true_last_100 = []  # percentage of true decisions among the last 100
        # self.true_positive_per_positive = []  # Only includes episodes that end in adaption, 1 if true, 0 if false
        # self.percentage_true_positive_100 = []  # percentage of true decisions among the last 100 positives
        # self.true_negative_per_negative = []  # Only includes episodes that don't end in adaption, 1 if true, 0 if false
        # self.percentage_true_negative_100 = []  # percentage of true decisions among the last 100 negatives
        self.action_1_probabilities_per_step = []  # include a list for each episode that logs the action probabilities per step

    def send_action(self, action):
        self.connector.send_action(action)

    def receive_reward_and_state(self):
        # print("receiving reward parameters...")

        reward_state_dict = self.connector.receive_reward_and_state()
        if reward_state_dict.__contains__('abort') and reward_state_dict['abort']:
            self.abort = True

        self.true = reward_state_dict['true']
        self.adapted = reward_state_dict['adapted']
        self.cost = reward_state_dict['cost']
        self.done = reward_state_dict['done']
        self.case_id = reward_state_dict['case_id']
        self.actual_duration = reward_state_dict['actual_duration']
        self.predicted_duration = reward_state_dict['predicted_duration']
        self.planned_duration = reward_state_dict['planned_duration']
        self.reliability = reward_state_dict['reliability']
        self.position = reward_state_dict['position']
        self.process_length = reward_state_dict['process_length']

        reward = self.compute_reward(self.adapted, self.cost, self.done, self.predicted_duration, self.planned_duration,
                                     self.reliability, self.position,
                                     self.process_length, actual_duration=self.actual_duration)
        self.reward = reward
        return reward, self.done, self.predicted_duration, self.planned_duration, self.reliability, self.position, \
               self.process_length, self.cost, self.adapted

    def log_with_tensorboard(self, tag, simple_value, step):
        if self.summary_writer is not None and BaseEnv.log_tensorboard:
            summary = tf.Summary(
                value=[tf.Summary.Value(tag=tag, simple_value=simple_value)])
            self.summary_writer.add_summary(summary, step)

    def do_logging(self, action):
        # Reward, Kosten und Aktion in Gesamtliste speichern:
        self.rewards.append(self.reward)
        self.costs.append(self.cost)
        self.actions.append(int(action))
        if self.predicted_duration > self.planned_duration:
            # pred_violation = 1
            adap_no_pred = 0
        else:
            # pred_violation = 0
            if int(action) == 1:
                adap_no_pred = 1
            else:
                adap_no_pred = 0

        # self.violation_predicted.append(pred_violation)
        self.adapted_no_violation.append(adap_no_pred)
        # Reward, Kosten und Aktion in temporaeren Listen fuer Episode speichern:
        self.tmp_cumul_cost_per_ep.append(self.cost)
        self.tmp_cumul_reward_per_ep.append(self.reward)
        # self.tmp_avg_action_per_ep.append(int(action))

        observation_input = np.reshape(self.state, (1, self.observation_space.shape[0]))
        self.action_1_probability.append(np.exp(self.model.actor(observation_input))[0, 1])

        # self.log_with_tensorboard(tag='custom/reward', simple_value=self.reward,
        #                           step=self.total_steps)
        # self.log_with_tensorboard(tag='custom/action', simple_value=self.action_value,
        #                           step=self.total_steps)
        # self.log_with_tensorboard(tag='custom/cost', simple_value=-self.cost,
        #                           step=self.total_steps)
        # self.log_with_tensorboard(tag='custom/violation_predicted', simple_value=pred_violation,
        #                           step=self.total_steps)
        self.log_with_tensorboard(tag='custom/adapted_though_no_violation_predicted', simple_value=adap_no_pred,
                                  step=self.total_steps)

        self.total_steps += 1
        if self.total_steps % 1000 == 0:
            if len(self.percentage_true_last_100) > 0:
                print("Step " + str(self.total_steps) + " in episode " + str(self.episode_count) + " with " + str(
                    self.percentage_true_last_100[-1]) + " true decisions")
            else:
                print("Step " + str(self.total_steps) + " in episode " + str(self.episode_count))

        if self.done:  # Ende einer Episode
            self.case_id_list.append(self.case_id)
            # Speichern der kumulativen und durschnittlichen Episodenkosten
            cumul_episode_cost = sum(self.tmp_cumul_cost_per_ep)
            self.cumul_cost_per_ep.append(cumul_episode_cost)
            # avg_episode_cost = cumul_episode_cost / len(self.tmp_cumul_cost_per_ep)
            # self.avg_cost_per_ep.append(avg_episode_cost)
            self.tmp_cumul_cost_per_ep = []

            # Speichern des kumulativen und durschnittlichen Rewards pro Episode
            cumul_episode_reward = sum(self.tmp_cumul_reward_per_ep)

            # avg_episode_reward = cumul_episode_reward / len(self.tmp_cumul_reward_per_ep)
            self.cumul_reward_per_ep.append(cumul_episode_reward)
            # self.avg_reward_per_ep.append(avg_episode_reward)
            self.tmp_cumul_reward_per_ep = []

            # Speichern der durschnittlichen Aktionen einer Episode und ob adaptiert wurde
            # avg_actions = sum(self.tmp_avg_action_per_ep) / len(self.tmp_avg_action_per_ep)
            # self.avg_action_per_ep.append(avg_actions)
            # elf.tmp_avg_action_per_ep = []

            true_negative_status = self.true
            self.true_per_ep.append(true_negative_status)
            if self.adapted:
                self.adapt_in_ep.append(1)

                self.position_of_adaptation_per_episode.append(self.position)
                self.earliness.append(1. - (self.position / self.process_length))
                self.log_with_tensorboard(tag='episode_result/earliness',
                                          simple_value=self.earliness[-1],
                                          step=self.episode_count)
                # self.true_positive_per_positive.append(true_negative_status)
            else:
                self.adapt_in_ep.append(0)
                self.position_of_adaptation_per_episode.append(-1)
                self.earliness.append(-1)

            self.case_length_per_episode.append(self.process_length)

            # self.true_negative_per_negative.append(true_negative_status)
            avg_adapted_100_value = sum(self.adapt_in_ep[-100:]) / 100
            self.avg_adapted_100.append(avg_adapted_100_value)

            percentage_true_last_100_value = get_average_last_entries_from_numeric_list(self.true_per_ep, 100)
            self.percentage_true_last_100.append(percentage_true_last_100_value)

            self.action_1_probabilities_per_step.append(self.action_1_probability)
            for i in range(len(self.action_1_probability)):
                self.log_with_tensorboard(tag='custom/action_1_proabability_step_' + str(i + 1),
                                          simple_value=self.action_1_probability[i], step=self.episode_count)
            self.action_1_probability = []

            # percentage_true_positive_100_value = get_average_last_entries_from_numeric_list(
            #    self.true_positive_per_positive, 100)
            # self.percentage_true_positive_100.append(percentage_true_positive_100_value)
            # percentage_true_negative_100_value = get_average_last_entries_from_numeric_list(
            #    self.true_negative_per_negative, 100)
            # self.percentage_true_negative_100.append(percentage_true_negative_100_value)

            self.log_with_tensorboard(tag='episode_reward/episode_cost', simple_value=-cumul_episode_cost,
                                      step=self.episode_count)
            # self.log_with_tensorboard(tag='episode_reward/episode_reward*', simple_value=cumul_episode_reward,
            #                           step=self.episode_count)
            # self.log_with_tensorboard(tag='episode_result/adapted', simple_value=self.adapted,
            #                           step=self.episode_count)
            self.log_with_tensorboard(tag='episode_result/average_adapted_last_100_episodes',
                                      simple_value=avg_adapted_100_value,
                                      step=self.episode_count)
            # self.log_with_tensorboard(tag='episode_result/average_action', simple_value=avg_actions,
            #                           step=self.episode_count)
            # self.log_with_tensorboard(tag='episode_result/ended_correctly', simple_value=true_negative_status,
            #                           step=self.episode_count)
            # if self.adapted:
            #     self.log_with_tensorboard(tag='episode_result/positives_ended_correctly',
            #                               simple_value=true_negative_status,
            #                               step=self.adapted_count)
            # else:
            #     self.log_with_tensorboard(tag='episode_result/negatives_ended_correctly',
            #                               simple_value=true_negative_status,
            #                               step=self.episode_count - self.adapted_count)
            self.log_with_tensorboard(tag='episode_result/percentage_of_trues_among_last_100_episodes',
                                      simple_value=percentage_true_last_100_value,
                                      step=self.episode_count)
            # self.log_with_tensorboard(tag='episode_result/percentage_of_trues_among_last_100_positives',
            #                           simple_value=percentage_true_positive_100_value,
            #                           step=self.episode_count)
            # self.log_with_tensorboard(tag='episode_result/percentage_of_trues_among_last_100_negatives',
            #                           simple_value=percentage_true_negative_100_value,
            #                           step=self.episode_count)
            #
            # self.log_with_tensorboard(tag='episode_result/percentage_Correct_Adaptation_Decisions',
            #                           simple_value=(self.true_per_ep[-1000:].count(1) / 1000) * 100,
            #                           step=self.episode_count)
            # self.log_with_tensorboard(tag='episode_result/adapt_in_ep',
            #                           simple_value=(self.adapt_in_ep.count(1) / self.adapt_in_ep.__len__()) * 100,
            #                           step=self.episode_count)
            self.episode_count += 1
            if self.adapted:
                self.adapted_count += 1

    def compute_reward(self, adapted, cost, done, predicted_duration, planned_duration, reliability, position,
                       process_length, actual_duration=0.):
        pass

    def _create_state(self):
        relative_position = self.position / self.process_length
        prediction_deviation = (self.predicted_duration - self.planned_duration) / self.planned_duration
        self.state = np.array(
            [relative_position, self.reliability, prediction_deviation])
        return self.state

    def reset(self):
        self.send_action(-1)
        self.receive_reward_and_state()

        self.state = self._create_state()

        return self.state

    def render(self, mode='human'):
        pass

    def step(self, action):
        pass

    def close(self):
        # print("Closing file and socket...")
        self.connector.close()
        print("Closed!")
        self.plot_experiment_data()
        path_string = self.save_path + "/" + os.path.basename(self.__class__.__name__) + "_" + str(
            BaseEnv.experiment_number)
        Path(path_string).mkdir(parents=True, exist_ok=True)
        self.write_experiment_data_to_csv(path_string)
        BaseEnv.experiment_number += 1

    def plot_experiment_data(self):
        print("plotting_data")

    def write_experiment_data_to_csv(self, save_path):

        splitpath = "splits"
        if not os.listdir(save_path).__contains__(splitpath):
            os.mkdir(save_path + "/" + splitpath)

        earliness_avg = calc_gliding_average_excluding(self.earliness, 100, [-1])
        true_avg_100 = calc_gliding_average_excluding(self.true_per_ep, 100)
        true_avg_1000 = calc_gliding_average_excluding(self.true_per_ep, 1000)
        adapt_avg = calc_gliding_average_excluding(self.adapt_in_ep, 100)
        cost_avg = calc_gliding_average_excluding(self.cumul_cost_per_ep, 100)
        reward_avg = calc_gliding_average_excluding(self.cumul_reward_per_ep, 100)

        accuracies, recalls, specificities, precisions, negative_predictive_values, balanced_accuracies, mccs, variables, two_thousands, five_thousands, totals = calc_gliding_accuracy_measurements(
            self.true_per_ep, self.adapt_in_ep, 100, BaseEnv.excluded_cases_metrics)

        variables.append(
            get_average_last_entries_from_numeric_list_excluding(self.earliness, max(1, len(
                self.earliness) - BaseEnv.excluded_cases_metrics), -1,
                                                                 len(self.earliness) - 1))
        variables.append(
            get_average_last_entries_from_numeric_list_excluding(self.adapt_in_ep, max(1, len(
                self.adapt_in_ep) - BaseEnv.excluded_cases_metrics),
                                                                 None, len(self.adapt_in_ep) - 1))
        variables.append(
            get_average_last_entries_from_numeric_list_excluding(self.cumul_reward_per_ep,
                                                                 max(1, len(
                                                                     self.cumul_reward_per_ep) - BaseEnv.excluded_cases_metrics),
                                                                 None,
                                                                 len(self.cumul_reward_per_ep) - 1))
        two_thousands.append(
            get_average_last_entries_from_numeric_list_excluding(self.earliness, max(1, len(self.earliness) - 2000), -1,
                                                                 len(self.earliness) - 1))
        two_thousands.append(
            get_average_last_entries_from_numeric_list_excluding(self.adapt_in_ep, max(1, len(self.adapt_in_ep) - 2000),
                                                                 None, len(self.adapt_in_ep) - 1))
        two_thousands.append(
            get_average_last_entries_from_numeric_list_excluding(self.cumul_reward_per_ep,
                                                                 max(1, len(self.cumul_reward_per_ep) - 2000), None,
                                                                 len(self.cumul_reward_per_ep) - 1))
        five_thousands.append(
            get_average_last_entries_from_numeric_list_excluding(self.earliness, max(1, len(self.earliness) - 5000), -1,
                                                                 len(self.earliness) - 1))
        five_thousands.append(
            get_average_last_entries_from_numeric_list_excluding(self.adapt_in_ep, max(1, len(self.adapt_in_ep) - 5000),
                                                                 None, len(self.adapt_in_ep) - 1))
        five_thousands.append(
            get_average_last_entries_from_numeric_list_excluding(self.cumul_reward_per_ep,
                                                                 max(1, len(self.cumul_reward_per_ep) - 5000), None,
                                                                 len(self.cumul_reward_per_ep) - 1))
        totals.append(
            get_average_last_entries_from_numeric_list_excluding(self.earliness, len(self.earliness), -1,
                                                                 len(self.earliness) - 1))
        totals.append(
            get_average_last_entries_from_numeric_list_excluding(self.adapt_in_ep, len(self.adapt_in_ep),
                                                                 None, len(self.adapt_in_ep) - 1))
        totals.append(
            get_average_last_entries_from_numeric_list_excluding(self.cumul_reward_per_ep,
                                                                 len(self.cumul_reward_per_ep), None,
                                                                 len(self.cumul_reward_per_ep) - 1))
        max_steps = max(map(len, self.action_1_probabilities_per_step))
        action_1 = np.array(
            [a1prob + [-1] * (max_steps - len(a1prob)) for a1prob in self.action_1_probabilities_per_step])
        action_1 = np.transpose(action_1)

        zip_argument = [self.case_id_list,
                        earliness_avg,
                        true_avg_100,
                        true_avg_1000,
                        adapt_avg,
                        self.adapt_in_ep,
                        cost_avg,
                        self.cumul_cost_per_ep,
                        reward_avg,
                        self.cumul_reward_per_ep,
                        self.true_per_ep,
                        self.position_of_adaptation_per_episode,
                        self.case_length_per_episode
                        ] + [a1 for a1 in action_1]

        columns = ['case_id',
                   'earliness_avg',
                   'true_avg_100',
                   'true_avg_1000',
                   'adaption_rate_avg',
                   'adapt_per_ep',
                   'costs_avg',
                   'cost_per_ep',
                   'rewards_avg',
                   'reward_per_ep',
                   'true_per_ep',
                   'position_adaptation_per_ep',
                   'case_length_per_ep'
                   ] + ['step_' + str(i + 1) + '_probability' for i in range(len(action_1))]

        dataframe_of_all_metrics = pd.DataFrame(list(zip(*zip_argument)),
                                                columns=columns)

        dataframe_of_all_metrics.to_csv(save_path + "/diagnostic_metrics.csv", header=True, index=False)

        zip_argument = [earliness_avg, accuracies,
                        balanced_accuracies, mccs, adapt_avg, reward_avg]
        columns = ["earliness", "accuracy",
                   "balanced_accuracy", "mcc", "adaptation_rate", "reward"]

        dataframe_of_accuracy_metrics = pd.DataFrame(list(zip(*zip_argument)), columns=columns)

        dataframe_of_accuracy_metrics.to_csv(save_path + "/gliding_accuracy_metrics.csv", header=True, index=False)

        accuracies_plot = dataframe_of_accuracy_metrics.plot(kind="line", linewidth=0.1, figsize=(10, 10),
                                                             ylim=(-0.25, 1), sharex=False, grid=True,
                                                             use_index=True, legend=True, subplots=False)

        accuracies_plot.get_figure().savefig(save_path + "/accuracy_metrics.pdf")

        accuracies_plot_subs = dataframe_of_accuracy_metrics.plot(kind="line", linewidth=0.1, figsize=(10, 50),
                                                                  ylim=(-0.25, 1), sharex=False, grid=True,
                                                                  use_index=True,
                                                                  legend=True, subplots=True)

        accuracies_plot_subs[0].get_figure().savefig(save_path + "/accuracy_metrics_subplots_.pdf")

        split_parts = 10.
        for i in range(int(split_parts)):
            accuracies_plot_split = dataframe_of_accuracy_metrics.plot(kind="line", linewidth=0.1, figsize=(10, 50),
                                                                       ylim=(-0.25, 1),
                                                                       xlim=(i * len(
                                                                           dataframe_of_accuracy_metrics) / split_parts,
                                                                             (i + 1) * len(
                                                                                 dataframe_of_accuracy_metrics) / split_parts),
                                                                       sharex=False, grid=True, use_index=True,
                                                                       legend=True,
                                                                       subplots=True)
            accuracies_plot_split[0].get_figure().savefig(
                save_path + "/" + splitpath + "/accuracy_metrics_subplots_split_" + str(i) + ".pdf")
            accuracies_plot_split = dataframe_of_accuracy_metrics.plot(kind="line", linewidth=0.1, figsize=(10, 10),
                                                                       ylim=(-0.25, 1),
                                                                       xlim=(i * len(
                                                                           dataframe_of_accuracy_metrics) / split_parts,
                                                                             (i + 1) * len(
                                                                                 dataframe_of_accuracy_metrics) / split_parts),
                                                                       sharex=False, grid=True, use_index=True,
                                                                       legend=True,
                                                                       subplots=False)
            accuracies_plot_split.get_figure().savefig(
                save_path + "/" + splitpath + "/accuracy_metrics_split_" + str(i) + ".pdf")
            plt.close('all')

        totals_collumns = ["accuracy", "recall", "spcificity", "precision",
                           "negative_predictive_value", "balanced_accuracy", "mcc",
                           "earliness", "adaptation_rate", "average_reward_per_episode"]

        dataframe_of_total_metrics = pd.DataFrame([variables, two_thousands, five_thousands, totals],
                                                  columns=totals_collumns)
        dataframe_of_total_metrics.to_csv(save_path + "/total_accuracy_metrics.csv", header=True, index=False)

        plt.close('all')

        #  self.write_totals(totals, totals_collumns, save_path, "totals.csv")
        #  self.write_totals(five_thousands, totals_collumns, save_path, "five_thousands.csv")
        #  self.write_totals(two_thousands, totals_collumns, save_path, "two_thousands.csv")
        #  self.write_totals(variables, totals_collumns, save_path, "variables.csv")

    def write_totals(self, totals: list, column_names: list, save_path, file_name):
        if not os.listdir(save_path).__contains__(file_name):
            f = open(save_path + "/" + file_name, "a")
            f.write("experiment_name,datetime,")
            for n in column_names:
                f.write(n + ",")
        else:
            f = open(save_path + "/" + file_name, "a")
        f.write("\n")
        f.write(BaseEnv.experiment_name + "," + str(datetime.datetime.now()) + ",")
        for t in totals:
            f.write(str(t) + ",")
        f.close()
