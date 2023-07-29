from typing import Union

import numpy as np


class PEST:
    def __init__(self, possible_values, starting_value):
        self.possible_values = possible_values
        self.current_index = self.find_nearest_index(self.possible_values, starting_value)
        self.last_response = None  # None: no response, 1: correct, 0: incorrect

    def update(self, response):
        if self.last_response is not None:
            # If the response has changed (i.e., from correct to incorrect or vice versa)
            if self.last_response != response:
                # Decrease the value (i.e., make an adjustment)
                self.current_index = max(0, self.current_index - 1)

        # If the response is correct, move to the right in the list (more difficult)
        if response == 1:
            self.current_index = min(len(self.possible_values) - 1, self.current_index + 1)
        else:  # If the response is incorrect, move to the left in the list (easier)
            self.current_index = max(0, self.current_index - 1)

        self.last_response = response

    def get_current_val(self):
        return self.possible_values[self.current_index]

    @staticmethod
    def find_nearest_index(array, value):
        array = np.asarray(array)
        idx = (np.abs(array - value)).argmin()
        return idx

