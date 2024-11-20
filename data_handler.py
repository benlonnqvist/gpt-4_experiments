import os

import pandas as pd
import matplotlib.pyplot as plt


class DataHandler:
    def __init__(self, save_root):
        self.df = pd.DataFrame()
        self.save_root = save_root

    def add_trial(self, **kwargs):
        raise NotImplementedError

    def save_data(self, file_name):
        if not os.path.exists(self.save_root):
            os.makedirs(self.save_root)
        seed = 0
        proposed_file_name = f"{file_name}_SEED{seed}.csv"
        while os.path.exists(os.path.join(self.save_root, proposed_file_name)):
            seed += 1
            proposed_file_name = f'{file_name}_SEED{seed}.csv'
        save_path = os.path.join(self.save_root, proposed_file_name)
        self.df.to_csv(save_path)

    def load_data(self, file_name):
        self.df = pd.read_csv(os.path.join(self.save_root, file_name), header=0, index_col=[0, 1])


class MalaniaDataHandler(DataHandler):
    def __init__(self, save_root):
        super().__init__(save_root)

    def add_trial(self,
                  block,
                  trial,
                  stimulus_id,
                  image_label,
                  model_response,
                  model_response_is_correct,
                  current_metric_value):

        trial_data = {
            "stimulus_id": stimulus_id.item(),
            "image_label": image_label.item(),
            "model_response": model_response,
            "model_response_is_correct": model_response_is_correct,
            "current_metric_value": current_metric_value
        }

        new_row = pd.DataFrame(trial_data, index=pd.MultiIndex.from_tuples([(block, trial)], names=['block', 'trial']))
        self.df = pd.concat([self.df, new_row])

    def plot_current_metric_value(self, block=None):
        """
        Plot the current metric value against the trial number.

        Args:
        - block (int, optional): If specified, plots only for that block. Otherwise, plots for all blocks.
        """
        plt.figure(figsize=(10, 6))

        if block is not None:
            # Plot for a specific block
            block_data = self.df.loc[block]
            plt.plot(block_data.index, block_data["current_metric_value"], label=f"Block {block}")
        else:
            # Plot for all blocks
            for block_id, data_block in self.df.groupby(level=0):
                plt.plot(data_block.index.get_level_values(1), data_block["current_metric_value"],
                         label=f"Block {block_id}")

        plt.xlabel("Trial Number")
        plt.ylabel("Current Metric Value")
        plt.title("Metric Value across Trials")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()


class ScialomDataHandler(DataHandler):
    def __init__(self, save_root):
        super().__init__(save_root)

    def add_trial(self,
                  subject_group,
                  visual_degrees,
                  is_correct,
                  subject_answer,
                  correct_answer,
                  percentage_elements,
                  stimulus_id):
        trial_data = {
            "subject_group": subject_group,
            "visual_degrees": visual_degrees,
            "image_duration": 200,
            "is_correct": is_correct,
            "subject_answer": subject_answer,
            "correct_answer": correct_answer,
            "percentage_elements": percentage_elements,
            "stimulus_id": stimulus_id
        }
        new_row = pd.DataFrame(trial_data)
        self.df = pd.concat([self.df, new_row])


class LonnqvistDataHandler(DataHandler):
    def __init__(self, save_root):
        super().__init__(save_root)

    # the data handler is not a class to handle participant data but instead the
    #  data of the LLM (trial data tracking etc.)
    def add_trial(self,
                  visual_degrees,
                  is_correct,
                  subject_answer,
                  correct_answer,
                  curve_length,
                  n_cross,
                  stimulus_id,
                  ):
        trial_data = {
            "visual_degrees": visual_degrees,
            "image_duration": 200,
            "is_correct": is_correct,
            "subject_answer": subject_answer,
            "correct_answer": correct_answer,
            "curve_length": curve_length,
            "n_cross": n_cross,
            "stimulus_id": stimulus_id
        }
        new_row = pd.DataFrame(trial_data)
        self.df = pd.concat([self.df, new_row])


import random


def generate_mock_datahandler(save_root=os.path.join('.', 'experimental_data'),
                              num_blocks=5, num_trials_per_block=10):
    # Create the DataHandler instance
    data_handler = DataHandler(save_root=save_root)

    # Generate mock data for blocks and trials
    for block in range(num_blocks):
        for trial in range(num_trials_per_block):
            stimulus_id = random.randint(1, 1000)
            image_label = random.choice(["cat", "dog", "bird", "fish"])
            model_response = random.choice(["cat", "dog", "bird", "fish"])
            model_response_is_correct = model_response == image_label
            current_metric_value = random.random()

            data_handler.add_trial(block=block, trial=trial,
                                   stimulus_id=stimulus_id,
                                   image_label=image_label,
                                   model_response=model_response,
                                   model_response_is_correct=model_response_is_correct,
                                   current_metric_value=current_metric_value)

    data_handler.plot_current_metric_value(block=0)
    data_handler.save_data(file_name="0.csv")
    return data_handler


if __name__ == '__main__':
    generate_mock_datahandler()