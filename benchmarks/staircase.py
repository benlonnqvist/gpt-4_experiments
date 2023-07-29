from typing import Union

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import expit


class PEST:
    def __init__(self, possible_values, starting_value):
        self.possible_values = sorted(possible_values)
        self.current_index = self.find_nearest_index(self.possible_values, starting_value)
        self.last_response = None  # None: no response, 1: correct, 0: incorrect
        self.responses = []
        self.indices = []

    def update(self, response):
        if self.last_response is not None:
            # If the response has changed (i.e., from correct to incorrect or vice versa), adjust
            if self.last_response != response:
                self.current_index = min(len(self.possible_values) - 1, self.current_index + 1)

        # if the response is correct, move to the left in the list (easier)
        if response == 1:
            self.current_index = max(0, self.current_index - 1)
        else:  # if the response is incorrect, move to the right in the list (more difficult)
            self.current_index = min(len(self.possible_values) - 1, self.current_index + 1)

        self.last_response = response
        self.responses.append(response)
        self.indices.append(self.current_index)

    def get_current_val(self):
        return self.possible_values[self.current_index]

    @staticmethod
    def find_nearest_index(array, value):
        array = np.asarray(array)
        idx = (np.abs(array - value)).argmin()
        return idx

    def plot_responses(self):
        y_values = [self.possible_values[i] for i in self.indices]
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, len(self.responses) + 1), y_values, marker='o')
        plt.xlabel('Response Number')
        plt.ylabel('Value of possible_values[self.current_index]')
        plt.title('PEST Responses over Time')
        plt.show()


if __name__ == '__main__':
    # simulate a mock agent
    def agent_response(value, peak=30, spread=10, chance_level=0.5):
        success_rate = expit((value - peak) / spread)
        return np.random.random() < success_rate

    pest = PEST(np.arange(0, 200, 5), 150)
    for _ in range(300):
        current_val = pest.get_current_val()
        response = agent_response(current_val)
        pest.update(response)

    pest.plot_responses()