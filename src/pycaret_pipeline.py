from utils import load_data
from pycaret.regression import *


def main():
    # Load raw data
    data = load_data('data/EmployeeBurnout_Preprocessed.csv')

    # Initial setup for the entire workflow
    setup(
        data=data,
        target='Burn Rate',
        session_id=123,
        verbose=True,
        profile=True,
        remove_multicollinearity=True,
        remove_outliers=True,
        normalize=True,
        transformation=True
    )

    # Train models and save the best one
    best_model = compare_models()
    evaluate_model(best_model)
    save_model(best_model, 'models/Best_Model')


if __name__ == "__main__":
    main()
