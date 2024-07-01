import base64
import json

import cv2
import numpy as np
import matplotlib.pyplot as plt


with open('local_info.json', 'r') as f:
    local_info = json.load(f)
API_ACCESS_KEY = local_info["PROJECT_KEY"]


def collect_hidden_params(model, messages, temperature):
    """A function that sets the hidden parameters into a dictionary of parameters to be called."""
    return {
        "model": model,
        "messages": messages,
        "temperature": temperature
    }


def load_image(image_path: str, benchmark_visual_degrees: float, candidate_visual_degrees: float,
               debug_mode: bool):
    # image_path must be .png
    assert image_path.endswith('.png')
    image = resize_input_image(image_path, benchmark_visual_degrees, candidate_visual_degrees, debug_mode=debug_mode)
    _, img_encoded = cv2.imencode('.png', image)
    image_base64 = base64.b64encode(img_encoded).decode('utf-8')
    if debug_mode:
        plot_image(image_base64)
    return image_base64


def plot_image(image):
    img_binary = base64.b64decode(image)
    img_np_arr = np.frombuffer(img_binary, np.uint8)
    img = cv2.imdecode(img_np_arr, cv2.IMREAD_COLOR)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    plt.imshow(img_rgb)
    plt.axis('off')
    plt.show()


def resize_input_image(image_path: str, benchmark_visual_degrees: float, candidate_visual_degrees: float,
                       debug_mode: bool):
    image = cv2.imread(image_path)

    if debug_mode:
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        plt.imshow(image_rgb)
        plt.axis('off')
        plt.show()

    resize_factor = candidate_visual_degrees / benchmark_visual_degrees

    # If candidate_visual_degrees > benchmark_visual_degrees, we resize and then pad the image
    if candidate_visual_degrees > benchmark_visual_degrees:
        resized_image = cv2.resize(image, None, fx=resize_factor, fy=resize_factor, interpolation=cv2.INTER_LINEAR)
        return pad_image(image, resized_image)

    # If candidate_visual_degrees < benchmark_visual_degrees, we crop first and then resize
    elif candidate_visual_degrees < benchmark_visual_degrees:
        cropped_image = crop_for_zoom(image, resize_factor)
        resized_image = cv2.resize(cropped_image, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_LINEAR)
        return resized_image

    return cv2.resize(image, None, fx=resize_factor, fy=resize_factor, interpolation=cv2.INTER_LINEAR)


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


def crop_for_zoom(image, resize_factor):
    height, width, _ = image.shape
    crop_height = int(height * resize_factor)
    crop_width = int(width * resize_factor)

    top = height // 2 - crop_height // 2
    bottom = top + crop_height
    left = width // 2 - crop_width // 2
    right = left + crop_width

    return image[top:bottom, left:right]
