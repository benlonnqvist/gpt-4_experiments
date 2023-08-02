import os
import json
import csv
import string

import cv2
import pandas as pd

from benchmarks.staircase import PEST


class Benchmark:
    def __init__(self, data_root_directory: str, setup_file_name: str, visual_degrees: float, name: str):
        self.data_root_directory = data_root_directory
        self.experimental_setup = self.read_experimental_setup(setup_file_name=setup_file_name)
        self.visual_degrees = visual_degrees
        self.check_experimental_setup_contents_ok()
        self.name = name

        self.current_block_index = 0
        self.current_block_name = None
        self.current_block_directory = 'none'
        self.current_trial_in_block = 0
        self.model_correct_responses = {}

    def start_new_block(self):
        raise NotImplementedError

    def end_block(self):
        raise NotImplementedError

    def run_stimulus_selection(self):
        raise NotImplementedError

    def approximate_experiment_token_length(self):
        raise NotImplementedError

    def check_experimental_setup_contents_ok(self):
        assert 'blocks' in self.experimental_setup
        assert isinstance(self.experimental_setup['blocks'], dict)
        for block in self.experimental_setup['blocks'].keys():
            assert isinstance(self.experimental_setup['blocks'][block], dict)
            assert 'name' in self.experimental_setup['blocks'][block]
            assert 'feedback' in self.experimental_setup['blocks'][block]
            assert 'trials' in self.experimental_setup['blocks'][block]

        assert 'shared_instruction' in self.experimental_setup
        assert isinstance(self.experimental_setup['shared_instruction'], str)

        assert 'experiment-specific_instruction' in self.experimental_setup
        assert isinstance(self.experimental_setup['experiment-specific_instruction'], str)

        assert 'feedback_string' in self.experimental_setup
        assert isinstance(self.experimental_setup['feedback_string'], dict)
        assert 'correct' in self.experimental_setup['feedback_string'].keys()
        assert 'incorrect' in self.experimental_setup['feedback_string'].keys()

        assert 'stimulus_message' in self.experimental_setup

    @staticmethod
    def read_experimental_setup(setup_file_name: str) -> dict:
        with open(setup_file_name, 'r') as f:
            experimental_setup = json.load(f)
        return experimental_setup

    @staticmethod
    def is_response_correct(response, stimulus):
        raise NotImplementedError

    @staticmethod
    def process_response_string(response: str) -> str:
        response = response.lower()
        response = "".join([character for character in response if character not in string.punctuation])
        return response.strip()


class Malania2007(Benchmark):
    setup_file_name = os.path.join('.', 'benchmarks', 'malania2007.json')
    visual_degrees = 2.986667
    name = 'BENCHMARKmalania2007'

    def __init__(self, data_root_directory: str):
        super().__init__(data_root_directory, self.setup_file_name, self.visual_degrees, self.name)
        self._current_block_index = 0
        self.current_block_index = str(self._current_block_index)
        self.current_block_name = 'none'
        self.current_block_directory = os.path.join(self.data_root_directory, self.current_block_name)
        self.current_trial_in_block = 0
        self.model_correct_responses = {}
        self.stimulus_metadata, self.stimulus_directory = None, None
        self.staircase = None

    def run_stimulus_selection(self):
        if not self.current_trial_in_block == 0:
            self.staircase.update(self.model_correct_responses[self.current_block_index][-1])
        selected_stimulus = self.select_stimulus()
        self.current_trial_in_block += 1
        return selected_stimulus

    def start_new_block(self):
        self.current_trial_in_block = 0
        self.current_block_name = self.experimental_setup['blocks'][self.current_block_index]['name']
        self.current_block_directory = os.path.join(self.data_root_directory, self.current_block_name)
        self.stimulus_metadata, self.stimulus_directory = self.load_metadata(self.data_root_directory)
        self.staircase = PEST(possible_values=self.stimulus_metadata['vernier_offset'].unique(), starting_value=150)

    def end_block(self):
        self._current_block_index += 1
        self.current_block_index = str(self._current_block_index)

    def select_stimulus(self) -> str:
        # Filter the metadata to include only those stimuli with the current vernier offset
        current_offset_stimuli = self.stimulus_metadata[
            self.stimulus_metadata['vernier_offset'] == self.staircase.get_current_val()]

        # Select a random stimulus from this filtered set
        selected_stimulus = current_offset_stimuli.sample(1)
        return selected_stimulus

    def load_metadata(self, root_directory):
        metadata_directory = os.path.join(root_directory, self.current_block_name, 'metadata.csv')
        image_directory = os.path.join(root_directory, self.current_block_name, 'images')
        stimuli = pd.read_csv(metadata_directory)
        stimuli = stimuli.astype({
            'image_size_x': 'int',
            'image_size_y': 'int',
            'image_size_c': 'int',
            'image_size_degrees': 'float',
            'vernier_height': 'float',
            'vernier_offset': 'float',
            'image_label': 'str',
            'flanker_height': 'float',
            'flanker_spacing': 'float',
            'line_width': 'float',
            'flanker_distance': 'float',
            'num_flankers': 'int',
            'vernier_position_x': 'int',
            'vernier_position_y': 'int',
            'stimulus_id': 'str',
            'filename': 'str'
        })

        return stimuli, image_directory

    def is_response_correct(self, response, stimulus):
        response = self.process_response_string(response)
        if response == stimulus['image_label'].item():
            return 1
        else:
            return 0

    def approximate_experiment_token_length(self):
        raise NotImplementedError


class TestMalania2007(Malania2007):
    def __init__(self, data_root_directory: str, setup_file_name):
        super().__init__(data_root_directory)
        self.setup_file_name = setup_file_name
        self.experimental_setup = self.read_experimental_setup(setup_file_name=self.setup_file_name)
        self.check_experimental_setup_contents_ok()
        self.name = 'testmalania2007'
