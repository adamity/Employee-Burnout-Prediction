
# Employee Burnout Prediction

This project aims to predict employee burnout using various machine learning models. The dataset used in this project contains information about employees and their burnout rates.

## Getting Started

### Prerequisites

- Python 3.x
- Virtual environment (optional but recommended)

### Installation

1. Clone the repository:

    ```sh
    git clone <repository-url>
    cd <repository-directory>
    ```

2. Create and activate a virtual environment:

    ```sh
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3. Install the required packages:

    ```sh
    pip install -r requirements.txt
    ```

### Dataset

Download the dataset from [this link](<https://www.kaggle.com/datasets/blurredmachine/are-your-employees-burning-out>) and place the CSV files in the `data/` directory with the name as `EmployeeBurnout_Raw.csv`.

### Running the Project

1. **Exploration**:

    ```sh
    python src/01_exploration.py
    ```

2. **Cleaning**:

    ```sh
    python src/02_cleaning.py
    ```

3. **Processing**:

    ```sh
    python src/03_processing.py
    ```

4. **Training**:

    ```sh
    python src/04_training.py
    ```

5. **PyCaret Implementation**:
    This is an alternative to the above steps. It uses PyCaret to automate the entire process.

    ```sh
    python src/pycaret_pipeline.py
    ```

## License

This project is licensed under the MIT License.
