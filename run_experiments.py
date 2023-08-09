import os
import json

from benchmarks.benchmarks import Benchmark, Malania2007, TestMalania2007
from local_functions import load_image, collect_hidden_params  # to hide potentially proprietary information

import openai


class ExperimentRunner:
    def __init__(self,
                 model: str,
                 temperature: float,
                 benchmark: Benchmark,
                 candidate_visual_degrees: float,
                 generic_system_message: str,
                 debug_mode: bool = False,
                 message_history_length: int = 10):
        self.model = model
        self.temperature = temperature
        self.benchmark = benchmark
        self.candidate_visual_degrees = candidate_visual_degrees
        system_message = generic_system_message + '\n' + self.benchmark.experimental_setup['shared_instruction']
        self.base_messages = [
            {"role": "system", "content": system_message},
            {"role": "user",   "content": self.benchmark.experimental_setup['experiment-specific_instruction']}
        ]
        self.full_message_history = self.base_messages
        self.current_messages = self.base_messages
        self.debug_mode = debug_mode
        self.message_history_length = message_history_length
        self.message_json_log_name = self.create_json_log(self.base_messages)

    def run_experiment(self):
        chat_params = collect_hidden_params(model=self.model, messages=self.messages, temperature=self.temperature)
        response_to_instruction = openai.ChatCompletion.create(**chat_params)
        self.parse_raw_response(response_to_instruction)
        for _ in self.benchmark.experimental_setup['blocks']:
            self.run_block()

    def run_block(self):
        self.benchmark.start_new_block()
        block_has_feedback = self.benchmark.experimental_setup['blocks'][self.benchmark.current_block_index]['feedback']
        trials_in_block = self.benchmark.experimental_setup['blocks'][self.benchmark.current_block_index]['trials']
        for trial in range(trials_in_block):
            stimulus = self.benchmark.run_stimulus_selection()
            stimulus_path = os.path.join(self.benchmark.current_block_directory, 'images', stimulus['filename'].item())
            image = load_image(stimulus_path, self.benchmark.visual_degrees, self.candidate_visual_degrees,
                               debug_mode=self.debug_mode)
            self.process_message({"role": "user",
                                  "content": [self.benchmark.experimental_setup['stimulus_message'], {"image": image}]},
                                 block_has_feedback)
            chat_params = collect_hidden_params(model=self.model,
                                                messages=self.prepare_model_message(),
                                                temperature=self.temperature)
            response = openai.ChatCompletion.create(**chat_params)
            model_response_message = self.parse_raw_response(response)
            self.process_message(model_response_message, block_has_feedback)
            response_is_correct = self.benchmark.is_response_correct(model_response_message, stimulus)
            self.benchmark.model_correct_responses[self.benchmark.current_block_index].append(response_is_correct)
            if block_has_feedback:
                feedback_type = "correct" if response_is_correct else "incorrect"
                self.process_message({"role": "user",
                                      "content": self.benchmark.experimental_setup['feedback_string'][feedback_type]},
                                     block_has_feedback)
        self.benchmark.end_block()

    def process_message(self, message: dict, block_has_feedback: int):
        """
        A method to process messages and keep track of which messages are ongoing in the current conversation and which
        messages should only be kept in the message history.

        :param message: the message dict.
        :param block_has_feedback: A bool indicating whether the block has feedback. We want to keep track of the
                                   number of conversation sets, not the absolute number of messages, and so if there
                                   is an additional piece of feedback, we need to add one to the length of the
                                   conversation that is saved.
        :return:
        """
        self.full_message_history.append(message)
        self.current_messages.append(message)
        if len(self.current_messages) > (2 + int(block_has_feedback)) * self.message_history_length:
            self.current_messages.pop(0)

    def prepare_model_message(self):
        self.base_messages.extend(self.current_messages)

    def parse_raw_response(self, response):
        model_response_message = response['choices'][0]['message']['content']
        self.full_message_history.append({"role": "assistant",
                                          "content": model_response_message})
        self.save_full_response_to_json(response)
        return model_response_message

    def handle_response(self, response, stimulus):
        block_id = self.benchmark.current_block_index
        is_correct = self.benchmark.is_response_correct(response, stimulus)
        if block_id not in self.benchmark.model_correct_responses:
            self.benchmark.model_correct_responses[block_id] = []
        self.benchmark.model_correct_responses[block_id].append(is_correct)

    def save_full_response_to_json(self, response):
        with open(self.message_json_log_name, 'r') as f:
            data = json.load(f)
        data['messages'].append(response)
        with open(self.message_json_log_name, 'w') as f:
            json.dump(data, f)

    def create_json_log(self, initial_data, root: str = './logs'):
        # TODO: add model params etc to initial_data too.
        # TODO: put in logs/ folder
        seed = 0
        message_json_log_name = f'LOG_MODEL{self.model}_TEMP{self.temperature}_BENCHMARK{self.benchmark.name}_' \
                                f'VISDEG{self.candidate_visual_degrees}_SEED{seed}.json'
        # Check if a file with the current name exists, if so increment the seed and create a new name
        while os.path.exists(os.path.join(root, message_json_log_name)):
            seed += 1
            message_json_log_name = f'LOG_MODEL{self.model}_TEMP{self.temperature}_BENCHMARK{self.benchmark.name}_' \
                                    f'VISDEG{self.candidate_visual_degrees}_SEED{seed}.json'
        save_path = os.path.join(root, message_json_log_name)

        with open(save_path, 'w') as f:
            json.dump({'messages': initial_data}, f)

        return save_path


if __name__ == '__main__':
    with open('local_info.json', 'r') as f:
        local_info = json.load(f)
    API_ACCESS_KEY = local_info["API_ACCESS_KEY"]
    MODEL_NAME = local_info["MODEL_NAME"]
    ORGANIZATION = local_info["ORGANIZATION"]
    GENERIC_SYSTEM_MESSAGE = local_info["GENERIC_SYSTEM_MESSAGE"]

    openai.organization = ORGANIZATION

    testmalania = TestMalania2007(data_root_directory=os.path.join('.', 'benchmarks', 'malania2007'),
                                  setup_file_name=os.path.join('.', 'benchmarks', 'test.json'))
    experiment = ExperimentRunner(model=MODEL_NAME, temperature=0., benchmark=testmalania,
                                  candidate_visual_degrees=testmalania.visual_degrees,
                                  generic_system_message=GENERIC_SYSTEM_MESSAGE, debug_mode=True)
    experiment.run_experiment()
