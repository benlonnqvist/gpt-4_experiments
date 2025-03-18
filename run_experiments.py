import os
import json
import time
import copy

from tqdm import tqdm

from benchmarks.benchmark import Benchmark, Scialom2024, Lonnqvist2024
from local_functions import load_image, collect_hidden_params  # to hide potentially proprietary information
from data_handler import DataHandler, ScialomDataHandler, LonnqvistDataHandler

from openai import OpenAI


class ExperimentRunner:
    def __init__(self,
                 model: str,
                 temperature: float,
                 benchmark: Benchmark,
                 data_handler: DataHandler,
                 candidate_visual_degrees: float,
                 generic_system_message: str,
                 debug_mode: bool = False,
                 message_history_length: int = 6  # a few trials of history
                 ):
        self.data_handler = data_handler
        self.model = model
        self.temperature = temperature
        self.benchmark = benchmark
        self.candidate_visual_degrees = candidate_visual_degrees
        self.debug_mode = debug_mode
        system_message = generic_system_message + '\n' + self.benchmark.experimental_setup['shared_instruction']
        example_stimulus_path = os.path.join('benchmarks', 'Lonnqvist2024', 'stimuli', 'image_instructions.png')
        example_image = load_image(example_stimulus_path, self.benchmark.visual_degrees, self.candidate_visual_degrees,
                           debug_mode=self.debug_mode)
        self.base_messages = [
            {"role": "system", "content": system_message},
            {"role": "user",   "content": self.benchmark.experimental_setup['experiment-specific_instruction']},
            {"role": "user",   "content": "Here is an example image instruction.", "image": example_image}
        ]
        self.full_message_history = copy.deepcopy(self.base_messages)
        self.current_messages = []
        self.message_history_length = message_history_length
        self.message_json_log_name = self.create_json_log(self.base_messages)
        self.client = OpenAI(api_key=API_ACCESS_KEY, organization=ORGANIZATION)

    def run_experiment(self):
        chat_params = collect_hidden_params(model=self.model,
                                            messages=self.prepare_model_message(),
                                            temperature=self.temperature)
        model_response_to_instruction = self.client.chat.completions.create(**chat_params).choices[0].message.content
        message_to_instruction = {"role": "assistant",
                                  "content": model_response_to_instruction}
        self.process_message(message_to_instruction, block_has_feedback=False, add_to_base_messages=True)
        for _ in self.benchmark.experimental_setup['blocks']:
            self.run_block()
            self.save_all_messages()
            self.clear_all_messages()

    def run_block(self):
        self.benchmark.start_new_block()
        print(f"Starting block {self.benchmark.current_block_index}")
        block_has_feedback = self.benchmark.experimental_setup['blocks'][self.benchmark.current_block_index]['feedback']
        trials_in_block = self.benchmark.experimental_setup['blocks'][self.benchmark.current_block_index]['trials']
        for trial in tqdm(range(trials_in_block)):
            time.sleep(1)
            # get stimulus
            stimulus = self.benchmark.run_stimulus_selection()
            stimulus_path = os.path.join(self.benchmark.current_block_directory, stimulus['file_name'].item())
            image = load_image(stimulus_path, self.benchmark.visual_degrees, self.candidate_visual_degrees,
                               debug_mode=self.debug_mode)

            # prepare message to the model
            self.process_message({"role": "user",
                                  "content": [self.benchmark.experimental_setup['stimulus_message'], {"image": image}]},
                                 block_has_feedback)
            chat_params = collect_hidden_params(model=self.model,
                                                messages=self.prepare_model_message(),
                                                temperature=self.temperature)

            # get model response and process it
            response = self.get_model_response(chat_params, trial)
            model_response_message = response.choices[0].message.content
            self.process_message({"role": "assistant",
                                  "content": model_response_message}, block_has_feedback)
            response_is_correct = self.benchmark.is_response_correct(model_response_message, stimulus)

            self.benchmark.end_trial()
            self.data_handler.add_trial(
                visual_degrees=self.benchmark.visual_degrees,
                is_correct=response_is_correct,
                subject_answer=model_response_message,
                correct_answer=stimulus['correct_response_key'],
                curve_length=stimulus['curve_length'],
                n_cross=stimulus['n_cross'],
                stimulus_id=stimulus['stimulus_id']
            )
            self.benchmark.model_correct_responses[self.benchmark.current_block_index].append(response_is_correct)

            # if the block has feedback, add it to messages
            if block_has_feedback:
                feedback_type = "correct" if response_is_correct else "incorrect"
                self.process_message({"role": "user",
                                      "content": self.benchmark.experimental_setup['feedback_string'][feedback_type]},
                                     block_has_feedback)
        # save data and end block
        self.data_handler.save_data(file_name=os.path.join(self.benchmark.name,
                                                           f'{self.benchmark.name}'))
        self.benchmark.end_block()

    def process_message(self, message: dict, block_has_feedback: int, add_to_base_messages: bool = False):
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
        if add_to_base_messages:
            self.base_messages.append(message)
            self.full_message_history.append(message)
        else:
            self.full_message_history.append(message)
            self.current_messages.append(message)
            if len(self.current_messages) > (2 + int(block_has_feedback)) * self.message_history_length:
                self.current_messages.pop(0)

    def get_model_response(self, chat_params, trial: int):
        response = None
        while response is None:
            try:
                response = self.client.chat.completions.create(**chat_params)
            except Exception as e:
                print(f"Error in response on trial {trial}:")
                print(e)
                print("Retrying...")
                time.sleep(5)
        return response

    def prepare_model_message(self):
        return self.base_messages + self.current_messages

    def handle_response(self, response, stimulus):
        block_id = self.benchmark.current_block_index
        is_correct = self.benchmark.is_response_correct(response, stimulus)
        if block_id not in self.benchmark.model_correct_responses:
            self.benchmark.model_correct_responses[block_id] = []
        self.benchmark.model_correct_responses[block_id].append(is_correct)

    def save_all_messages(self):
        with open(self.message_json_log_name, 'r') as f:
            data = json.load(f)
        data['messages'].append(self.full_message_history)
        with open(self.message_json_log_name, 'w') as f:
            json.dump(data, f)

    def clear_all_messages(self):
        self.full_message_history = copy.deepcopy(self.base_messages)
        self.current_messages = []

    def create_json_log(self, initial_data, root_folder_name: str = 'logs'):
        root = os.path.join('.', root_folder_name)
        if not os.path.exists(root):
            os.makedirs(root)

        seed = 0
        message_json_log_name = f'LOG_MODEL{self.model}_TEMP{self.temperature}_BENCHMARK{self.benchmark.name}_' \
                                f'VISDEG{self.candidate_visual_degrees}_' \
                                f'SEED{seed}.json'
        # Check if a file with the current name exists, if so increment the seed and create a new name
        while os.path.exists(os.path.join(root, message_json_log_name)):
            seed += 1
            message_json_log_name = f'LOG_MODEL{self.model}_TEMP{self.temperature}_BENCHMARK{self.benchmark.name}_' \
                                    f'VISDEG{self.candidate_visual_degrees}_' \
                                    f'SEED{seed}.json'
        save_path = os.path.join(root, message_json_log_name)

        with open(save_path, 'w') as f:
            json.dump({'messages': initial_data}, f)

        return save_path


if __name__ == '__main__':
    with open('local_info.json', 'r') as f:
        local_info = json.load(f)
    API_ACCESS_KEY = local_info["PROJECT_KEY"]
    MODEL_NAME = local_info["MODEL_NAME"]
    ORGANIZATION = local_info["ORGANIZATION"]
    GENERIC_SYSTEM_MESSAGE = local_info["GENERIC_SYSTEM_MESSAGE"]

    testlonnqvist = Lonnqvist2024(data_root_directory=os.path.join('.', 'benchmarks', 'Lonnqvist2024'))
    data_handler = LonnqvistDataHandler(save_root=os.path.join('.', 'experimental_data'))
    experiment = ExperimentRunner(model=MODEL_NAME,
                                  temperature=0.,
                                  benchmark=testlonnqvist,
                                  data_handler=data_handler,
                                  candidate_visual_degrees=testlonnqvist.visual_degrees,
                                  generic_system_message=GENERIC_SYSTEM_MESSAGE,
                                  debug_mode=False)
    experiment.run_experiment()
