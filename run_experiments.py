import os
import json

from api_key import API_ACCESS_KEY
from benchmarks.benchmarks import Benchmark, Malania2007

import openai


class ExperimentRunner:
    def __init__(self,
                 model: str,
                 temperature: float,
                 benchmark: Benchmark,
                 candidate_visual_degrees: float,
                 data_root_directory: str):
        self.model = model
        self.temperature = temperature
        self.benchmark = benchmark
        self.messages = [
            {"role": "system", "content": self.benchmark.experimental_setup['shared_instruction']},
            {"role": "user",   "content": self.benchmark.experimental_setup['experiment-specific_instruction']}
        ]
        self.message_json_log_name = self.create_json_log(self.messages)

    def run_experiment(self):
        response_to_instruction = openai.ChatCompletion.create(
            model=self.model,
            messages=self.messages,
            temperature=self.temperature
        )
        self.parse_raw_response(response_to_instruction)
        for _ in self.benchmark.experimental_setup['blocks']:
            self.run_block()

    def run_block(self):
        self.benchmark.start_new_block()
        block_has_feedback = self.benchmark.experimental_setup['blocks'][self.benchmark.current_block]['feedback']
        trials_in_block = self.benchmark.experimental_setup['blocks'][self.benchmark.current_block]['trials']
        for trial in range(trials_in_block):
            stimulus = self.benchmark.run_trial()
            self.messages.append({"role": "user", "content": stimulus['filename']})
            response = openai.ChatCompletion.create(
                model=self.model,
                messages=self.messages,
                temperature=self.temperature
            )
            self.parse_raw_response(response)
            response_is_correct = self.benchmark.is_response_correct(response, stimulus)
            self.benchmark.model_correct_responses[self.benchmark.current_block].append(response_is_correct)
            if block_has_feedback:
                feedback_type = "correct" if response_is_correct else "incorrect"
                self.messages.append({"role": "user",
                                      "content": self.benchmark.experimental_setup['feedback_string'][feedback_type]})
        self.benchmark.end_block()

    def parse_raw_response(self, response):
        self.messages.append({"role": "assistant",
                              "content": response['choices'][0]['message']['content']})
        self.save_full_response_to_json(response)

    def handle_response(self, response, stimulus):
        block_id = self.benchmark.current_block_index
        is_correct = self.benchmark.is_response_correct(response, stimulus)
        if block_id not in self.benchmark.model_correct_responses:
            self.benchmark.model_correct_responses[block_id] = []
        self.benchmark.model_correct_responses[block_id].append(is_correct)

    def save_full_response_to_json(self, response):
        with open(self.message_json_log_name, 'r') as f:
            data = json.load(f)
        data.append(response)
        with open(self.message_json_log_name, 'w') as f:
            json.dump(data, f)

    def create_json_log(self, initial_data):
        seed = 0
        message_json_log_name = f'LOG_{self.model}_{self.temperature}_{self.benchmark}_' \
                                f'{self.benchmark.candidate_visual_degrees}visdeg_{seed}.json'

        # Check if a file with the current name exists, if so increment the seed and create a new name
        while os.path.exists(self.message_json_log_name):
            seed += 1
            message_json_log_name = f'LOG_{self.model}_{self.temperature}_{self.benchmark}_' \
                                    f'{self.benchmark.candidate_visual_degrees}visdeg_{seed}.json'

        with open(message_json_log_name, 'w') as f:
            json.dump({initial_data}, f)

        return message_json_log_name
