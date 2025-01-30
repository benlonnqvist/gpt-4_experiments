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
        self.staircase = None

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

    def update_staircase(self):
        raise NotImplementedError

    def end_trial(self):
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
    def find_correct_response_in_string(input_str: str) -> str:
        raise NotImplementedError

    def process_response_string(self, response: str) -> str:
        response = response.lower()
        response = "".join([character for character in response if character not in string.punctuation])
        response = self.find_correct_response_in_string(response)
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

    def run_stimulus_selection(self):
        selected_stimulus = self.select_stimulus()
        return selected_stimulus

    def start_new_block(self):
        self.current_trial_in_block = 0
        self.current_block_name = self.experimental_setup['blocks'][self.current_block_index]['name']
        self.model_correct_responses[self.current_block_index] = []
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
            is_correct = 1
        else:
            is_correct = 0
        self.model_correct_responses[self.current_block_index].append(is_correct)
        return is_correct

    def update_staircase(self):
        self.staircase.update(self.model_correct_responses[self.current_block_index][-1])

    def end_trial(self):
        self.current_trial_in_block += 1

    def approximate_experiment_token_length(self):
        raise NotImplementedError


class TestMalania2007(Malania2007):
    def __init__(self, data_root_directory: str, setup_file_name):
        super().__init__(data_root_directory)
        self.setup_file_name = setup_file_name
        self.experimental_setup = self.read_experimental_setup(setup_file_name=self.setup_file_name)
        self.check_experimental_setup_contents_ok()
        self.name = 'testmalania2007'

    def is_response_correct(self, response, stimulus):
        response = self.process_response_string(response)
        if response == stimulus['image_label'].item():
            is_correct = 1
        else:
            is_correct = 0
        self.model_correct_responses[self.current_block_index].append(is_correct)
        return is_correct


class ANONAUTHOR22024(Benchmark):
    setup_file_name = os.path.join('.', 'benchmarks', 'ANONAUTHOR22024', 'ANONAUTHOR22024.json')
    visual_degrees = 8.
    name = 'BENCHMARKANONAUTHOR22024'

    def __init__(self,
                 data_root_directory: str,
                 subject_group: str = 'segments'):
        super().__init__(data_root_directory, self.setup_file_name, self.visual_degrees, self.name)
        self._current_block_index = 0
        self.current_block_index = str(self._current_block_index)
        self.current_block_name = 'none'
        self.current_block_directory = os.path.join(self.data_root_directory)
        self.current_trial_in_block = 0
        self.model_correct_responses = {}
        self.subject_group = [subject_group] * 9 + ['contours', 'RGB']
        self.stimulus_metadata, self.stimulus_directory = None, None

    def run_stimulus_selection(self):
        selected_stimulus = self.select_stimulus()
        return selected_stimulus

    def start_new_block(self):
        self.current_trial_in_block = 0
        self.current_block_name = self.experimental_setup['blocks'][self.current_block_index]['name']
        self.model_correct_responses[self.current_block_index] = []
        self.current_block_directory = os.path.join(self.data_root_directory)
        self.stimulus_metadata, self.stimulus_directory = self.load_metadata(
            self.data_root_directory,
            self.current_block_name
        )

    def end_block(self):
        self._current_block_index += 1
        self.current_block_index = str(self._current_block_index)

    def select_stimulus(self) -> str:
        # select a random stimulus_id from self.stimulus_metadata['stimulus_id']
        selected_stimulus = self.stimulus_metadata.sample(1)
        # remove selected_stimulus from self.stimulus_metadata
        self.stimulus_metadata = self.stimulus_metadata[
            self.stimulus_metadata['stimulus_id'] != selected_stimulus['stimulus_id'].item()]
        return selected_stimulus

    def load_metadata(self, root_directory, percentage_elements: str):
        if self.current_block_name == 'practice':
            metadata_directory = os.path.join(root_directory, 'MetaData_Stimuli_training.csv')
        else:
            metadata_directory = os.path.join(root_directory, 'MetaData_Stimuli_experiment.csv')
        image_directory = os.path.join(root_directory, 'Stimuli')
        stimuli = pd.read_csv(metadata_directory)
        stimuli = stimuli.astype({
            'image_height': 'int',
            'image_width': 'int',
            'channel': 'int',
            'percentage_elements': 'str',
            'representation_mode': 'str',
            'object_id': 'int',
            'category': 'str',
            'visual_degrees': 'float',
            'stimulus_id': 'str',
            'file_name': 'str'
        })
        # filter stimuli by percentage_elements and representation_mode == self.subject_group
        stimuli = stimuli[(stimuli['percentage_elements'] == percentage_elements) &
                          (stimuli['representation_mode'] == self.subject_group[self._current_block_index])]
        return stimuli, image_directory

    def is_response_correct(self, response, stimulus):
        response = self.process_response_string(response)
        if response == stimulus['category'].item().lower():
            is_correct = 1
        else:
            is_correct = 0
        self.model_correct_responses[self.current_block_index].append(is_correct)
        return is_correct

    def end_trial(self):
        self.current_trial_in_block += 1

    def approximate_experiment_token_length(self):
        raise NotImplementedError

    @staticmethod
    def find_correct_response_in_string(input_str: str) -> str:
        correct_responses = ['truck', 'cup', 'bowl', 'binoculars', 'glasses', 'beanie', 'pan', 'sewing machine', 'shovel', 'banana', 'boot', 'lamp']
        for response in correct_responses:
            if response in input_str:
                return response
        return 'INVALID RESPONSE'


class ANONAUTHOR12024(Benchmark):
    setup_file_name = os.path.join('.', 'benchmarks', 'ANONAUTHOR12024', 'ANONAUTHOR12024.json')
    visual_degrees = 8.
    name = 'BENCHMARKpathfinder'

    def __init__(self, data_root_directory: str):
        super().__init__(data_root_directory, self.setup_file_name, self.visual_degrees, self.name)
        self._current_block_index = 0
        self.current_block_index = str(self._current_block_index)
        self.current_block_name = 'none'
        self.current_block_directory = os.path.join(self.data_root_directory)
        self.current_trial_in_block = 0
        self.subject_group = ['all', ]
        self.model_correct_responses = {}
        self.stimulus_metadata, self.stimulus_directory = None, None

    def run_stimulus_selection(self):
        selected_stimulus = self.select_stimulus()
        return selected_stimulus

    def start_new_block(self):
        self.current_trial_in_block = 0
        self.current_block_name = self.experimental_setup['blocks'][self.current_block_index]['name']
        self.model_correct_responses[self.current_block_index] = []
        self.current_block_directory = os.path.join(self.data_root_directory)
        self.stimulus_metadata, self.stimulus_directory = self.load_metadata(
            self.data_root_directory,
            self.current_block_name
        )

    def end_block(self):
        self._current_block_index += 1
        self.current_block_index = str(self._current_block_index)

    def select_stimulus(self) -> str:
        # select a random stimulus_id from self.stimulus_metadata['stimulus_id']
        selected_stimulus = self.stimulus_metadata.sample(1)
        # remove selected_stimulus from self.stimulus_metadata
        self.stimulus_metadata = self.stimulus_metadata[
            self.stimulus_metadata['stimulus_id'] != selected_stimulus['stimulus_id'].item()]
        return selected_stimulus

    def load_metadata(self, root_directory, percentage_elements: str):
        metadata_directory = os.path.join(root_directory, 'stimuli', 'metadata.csv')
        image_directory = os.path.join(root_directory, 'stimuli', 'images')
        stimuli = pd.read_csv(metadata_directory)
        stimuli = stimuli.assign(
            image_height=1080,
            image_width=1920,
            visual_degrees=8.,
        )
        stimuli = stimuli.astype({
            'path': 'str',
            'idx': 'int',
            'dashed': 'str',
            'correct_response_key': 'str',
            'condition': 'str',
            'n_curves': 'str',
            'curve_length': 'int',
            'n_cross': 'int',
            'image_height': 'int',
            'image_width': 'int',
        })
        stimuli = stimuli.rename(columns={'idx': 'stimulus_id',
                                          'path': 'file_name'})
        return stimuli, image_directory

    def is_response_correct(self, response, stimulus):
        response = self.process_response_string(response)
        if response == stimulus['correct_response_key'].item().lower():
            is_correct = 1
        else:
            is_correct = 0
        self.model_correct_responses[self.current_block_index].append(is_correct)
        return is_correct

    def end_trial(self):
        self.current_trial_in_block += 1

    @staticmethod
    def find_correct_response_in_string(input_str: str) -> str:
        correct_responses = ['f', 'j']
        for response in correct_responses:
            if response in input_str:
                return response
        return 'INVALID RESPONSE'


class ShowImage(Benchmark):
    setup_file_name = os.path.join('.', 'benchmarks', 'ANONAUTHOR2Unpublished')
    visual_degrees = 8.
    name = 'BENCHMARKshowimage'

    def __init__(self, data_root_directory: str):
        super().__init__(data_root_directory, self.setup_file_name, self.visual_degrees, self.name)
        raise NotImplementedError
