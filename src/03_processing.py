from utils import *
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder


def inspect(data):
    inspect_basic_structure(data)
    examine_summary_statistics(data)
    identify_missing_values(data)
    check_for_duplicates(data)
    check_for_constant_columns(data)
    check_data_types(data)


def convert_to_category(data, categorical_columns):
    data = data.copy()

    for column in categorical_columns:
        data[column] = data[column].astype('category')

    return data


def scale_columns(data, columns_to_scale, scaler_type='minmax'):
    data = data.copy()

    if scaler_type == 'minmax':
        scaler = MinMaxScaler()
    elif scaler_type == 'standard':
        scaler = StandardScaler()
    else:
        raise ValueError("Invalid scaler type. Use 'minmax' or 'standard'.")

    data[columns_to_scale] = scaler.fit_transform(data[columns_to_scale])
    return data


def label_encode_columns(data, binary_columns):
    data = data.copy()

    for column in binary_columns:
        encoder = LabelEncoder()
        data[column] = encoder.fit_transform(data[column])

    return data


def main():
    data = load_data('data/EmployeeBurnout_Cleaned.csv')

    print("\033[34mBefore Data Processing\033[0m\n")
    inspect(data)

    categorical_columns = [
        'Gender', 'Company Type', 'WFH Setup Available'
    ]
    data = convert_to_category(data, categorical_columns)

    columns_to_scale = [
        'Designation', 'Resource Allocation', 'Mental Fatigue Score'
    ]
    data = scale_columns(data, columns_to_scale, scaler_type='minmax')

    binary_columns = ['Gender', 'Company Type', 'WFH Setup Available']
    data = label_encode_columns(data, binary_columns)

    print("\033[34mAfter Data Processing\033[0m\n")
    inspect(data)

    save_data(data, 'data/EmployeeBurnout_Preprocessed.csv')


if __name__ == "__main__":
    main()
