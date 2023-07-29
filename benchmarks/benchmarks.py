import os
import json
import csv
import string

import cv2
import pandas as pd

from staircase import PEST


class Benchmark:
    def __init__(self, setup_file_name: str, candidate_visual_degrees: float):
        self.experimental_setup = self.read_experimental_setup(setup_file_name=setup_file_name)
        self.check_experimental_setup_contents_ok()
        self.candidate_visual_degrees = candidate_visual_degrees

        self.current_block_index = 0
        self.current_block = None
        self.current_trial_in_block = 0
        self.model_correct_responses = {}

    def start_new_block(self):
        raise NotImplementedError

    def end_block(self):
        raise NotImplementedError

    def run_trial(self):
        raise NotImplementedError

    def approximate_experiment_token_length(self):
        raise NotImplementedError

    def resize_input_image(self, image, source_visual_degrees, candidate_visual_degrees):
        resize_factor = candidate_visual_degrees / source_visual_degrees
        resized_image = cv2.resize(image, None, fx=resize_factor, fy=resize_factor, interpolation=cv2.INTER_LINEAR)

        # If candidate_visual_degrees > source_visual_degrees, we need to pad the image with zeros
        if candidate_visual_degrees > source_visual_degrees:
            return self.pad_image(image, resized_image)

        # If candidate_visual_degrees < source_visual_degrees, we need to crop the central portion of the image
        elif candidate_visual_degrees < source_visual_degrees:
            return self.crop_image(image, resized_image)

        return resized_image

    def check_experimental_setup_contents_ok(self):
        assert 'blocks' in self.experimental_setup
        assert isinstance(self.experimental_setup['blocks'], dict)
        for block in self.experimental_setup['blocks'].keys():
            assert isinstance(block, list)
            assert 'name' in block
            assert 'feedback' in block
            assert 'trials' in block

        assert 'shared_instruction' in self.experimental_setup
        assert isinstance(self.experimental_setup['shared_instruction'], str)

        assert 'experiment-specific_instruction' in self.experimental_setup
        assert isinstance(self.experimental_setup['experiment-specific_instruction'], str)

        assert 'feedback_string' in self.experimental_setup
        assert isinstance(self.experimental_setup['feedback_string'], dict)
        assert 'correct' in self.experimental_setup['feedback_string'].keys()
        assert 'incorrect' in self.experimental_setup['feedback_string'].keys()

    @staticmethod
    def read_experimental_setup(setup_file_name: str) -> dict:
        with open(setup_file_name, 'r') as f:
            experimental_setup = json.load(f)
        return experimental_setup

    @staticmethod
    def pad_image(original_image, resized_image):
        height, width, _ = original_image.shape
        resized_height, resized_width, _ = resized_image.shape

        top = (height - resized_height) // 2
        bottom = height - top - resized_height
        left = (width - resized_width) // 2
        right = width - left - resized_width

        # Pad the resized image with zeros to match the original image size
        padded_image = cv2.copyMakeBorder(resized_image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[0, 0, 0])
        return padded_image

    @staticmethod
    def crop_image(original_image, resized_image):
        height, width, _ = original_image.shape
        resized_height, resized_width, _ = resized_image.shape

        top = (resized_height - height) // 2
        bottom = resized_height - top + height
        left = (resized_width - width) // 2
        right = resized_width - left + width

        # Crop the central portion of the resized image to match the original image size
        cropped_image = resized_image[top:bottom, left:right]
        return cropped_image

    @staticmethod
    def is_response_correct(response, stimulus):
        raise NotImplementedError

    @staticmethod
    def process_response_string(response: str) -> str:
        response = response.lower()
        response = "".join([character for character in response if character not in string.punctuation])
        return response.strip()


class Malania2007(Benchmark):
    setup_file_name = 'malania2007.json'

    def __init__(self, candidate_visual_degrees: float, data_root_directory: str):
        super().__init__(self.setup_file_name, candidate_visual_degrees)
        self.data_root_directory = data_root_directory
        self.current_block_index = 0
        self.current_block = None
        self.current_trial_in_block = 0
        self.model_correct_responses = {}
        self.stimulus_metadata, self.stimulus_directory = None, None
        self.staircase = None

    def run_trial(self):
        if not self.current_trial_in_block == 0:
            self.staircase.update(self.model_correct_responses[self.current_block][-1])
        selected_stimulus = self.select_stimulus()
        self.current_trial_in_block += 1
        return selected_stimulus

    def start_new_block(self):
        self.current_trial_in_block = 0
        self.current_block = self.experimental_setup['blocks'].keys()[str(self.current_block_index)]
        self.stimulus_metadata, self.stimulus_directory = self.load_metadata(self.data_root_directory)
        self.staircase = PEST(possible_values=self.stimulus_metadata['vernier_offset'].unique(), starting_value=150)

    def end_block(self):
        self.current_block_index += 1

    def select_stimulus(self):
        # Filter the metadata to include only those stimuli with the current vernier offset
        current_offset_stimuli = self.stimulus_metadata[
            self.stimulus_metadata['vernier_offset'] == self.staircase.get_current_val()]

        # Select a random stimulus from this filtered set
        selected_stimulus = current_offset_stimuli.sample(1)
        return selected_stimulus

    def load_metadata(self, root_directory):
        metadata_directory = os.path.join(root_directory, self.current_block, 'metadata.csv')
        image_directory = os.path.join(root_directory, self.current_block, 'images')
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
        if response == stimulus['image_label']:
            return 1
        else:
            return 0

    def approximate_experiment_token_length(self):
        raise NotImplementedError
