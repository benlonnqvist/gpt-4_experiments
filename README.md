# GPT-4 Experiments

## Overview

This repository contains experiments with OpenAI's GPT-4 model. It includes scripts and tools designed to handle data, run experiments, and benchmark the performance of GPT-4 across various tasks.

## Repository Structure

- **`benchmarks/`**: Contains benchmark datasets and results used to evaluate GPT-4's performance.
- **`main.py`**: The primary script to execute experiments.
- **`data_handler.py`**: Manages data loading, preprocessing, and augmentation.
- **`local_functions.py`**: Houses utility functions to support experiments.
- **`run_experiments.py`**: Orchestrates the execution of various experiments.
- **`LICENSE`**: Specifies the repository's licensing terms.
- **`README.md`**: Provides an overview and instructions for the repository.

## Installation

To set up the environment for these experiments:

1. **Clone the repository**:

```bash
git clone https://github.com/benlonnqvist/gpt-4_experiments.git
cd gpt-4_experiments
```

2. **Create a virtual environment**:

```bash
python -m venv env
source env/bin/activate  # On Windows: env\Scripts\activate
```

3. **Install dependencies**:

```bash
pip install -r requirements.txt
```

*Note: Ensure that `requirements.txt` is present in the repository with all necessary dependencies listed.*

## Usage

### Data Preparation

Place your datasets in the `benchmarks/` directory. Ensure they are in the expected format required by `data_handler.py`.
Benchmarks typically should contain some metadata in `.json` files. Follow the format of extant benchmarks that are provided. 

### Running Experiments

To initiate experiments, execute the `run_experiments.py` script:

```bash
python run_experiments.py
```

This script uses functions from `data_handler.py` and `local_functions.py` to process data and run experiments.

### Viewing Results

Results and logs will be stored in the `benchmarks/` directory. Review these files to analyze GPT-4's performance.

## Contributing

Contributions are welcome! If you have ideas for improvements or new experiments, please follow these steps:

1. Fork the repository.
2. Create a new branch (`git checkout -b feature/my-feature`).
3. Make your changes and commit them (`git commit -am 'Add new feature'`).
4. Push your changes to the branch (`git push origin feature/my-feature`).
5. Submit a pull request with a clear description of your changes.

Ensure that your code adheres to standard Python coding conventions and is well-documented.

## License

This project is licensed under the MIT License. See the [LICENSE](https://github.com/benlonnqvist/gpt-4_experiments/blob/main/LICENSE) file for more details.

