import unicodecsv
from gym_threshold_begone.envs.connector import environment_connector
from gym_threshold_begone.envs.connector._base_connector import BaseConnector


class ModelFileConnector(BaseConnector):

    def __init__(self, predictive_output_directory="predictions", **kwargs):
        self.ensemble_file = open(predictive_output_directory + "/ensemble.csv", "rb")
        self.reader = unicodecsv.reader(self.ensemble_file, delimiter=',')
        self.line = next(self.reader)  # skip header
        self.adapted = False
        self.done = False
        self.first = True
        self.case_id = -1
        self.previous_id = -2
        self.process_length = 0
        self.position = 0
        self.actual_duration = 0
        self.planned_duration = 0
        self.predicted_duration = 0
        self.prediction_deviation = 0
        self.reliability = 0
        self.abort = False
        self._advance_line()

    def send_action(self, action):
        if action >= 0:
            self.previous_id = self.case_id
            if action >= 1:
                self.adapted = True
                self.done = True
            if not self.done:
                self._advance_line()

    def receive_reward_and_state(self):
        if self.position + 1 == self.process_length or self.adapted:
            if self.done:
                result = self._build_result_dict()
                while self.previous_id == self.case_id and not self.abort:
                    self._advance_line()
                self.done = False
                self.adapted = False
                return result
            else:
                self.done = True
        return self._build_result_dict()

    def _build_result_dict(self):
        result = dict()
        result['adapted'] = self.adapted
        result['done'] = self.done
        result['case_id'] = self.case_id
        result['actual_duration'] = self.actual_duration
        result['predicted_duration'] = self.predicted_duration
        result['planned_duration'] = self.planned_duration
        result['reliability'] = self.reliability
        result['position'] = self.position
        result['process_length'] = self.process_length
        result['cost'] = 0  # TODO: Cost Model
        violation = (self.actual_duration - self.planned_duration) > 0
        true = (violation == result['adapted'])
        result['abort'] = self.abort
        result['true'] = true
        return result

    def _advance_line(self):
        self._read_line()
        self._parse_line()

    def _read_line(self):
        self.line = next(self.reader, None)

    def _parse_line(self):
        if self.line is None:
            self.abort = True
        else:
            self.case_id = int(self.line[0])
            self.process_length = float(self.line[1])
            self.position = int(self.line[2])
            self.actual_duration = float(self.line[3])
            self.planned_duration = float(self.line[4])
            self.predicted_duration = float(self.line[5])
            self.prediction_deviation = float(self.line[6])
            self.reliability = float(self.line[7])

    def close(self):
        self.ensemble_file.close()
