from utils import *


def inspect(data):
    inspect_basic_structure(data)
    examine_summary_statistics(data)
    identify_missing_values(data)
    check_for_duplicates(data)
    check_for_constant_columns(data)
    check_data_types(data)


def fix_data_types(data):
    data = data.copy()

    categorical_columns = [
        'Employee ID', 'Date of Joining',
        'Gender', 'Company Type', 'WFH Setup Available'
    ]

    for column in categorical_columns:
        data[column] = data[column].astype('category')

    return data


def handle_outliers(data, method="mask"):
    data = data.copy()
    numeric_columns = data.select_dtypes(include=['float64', 'int64'])
    q1 = numeric_columns.quantile(0.25)
    q3 = numeric_columns.quantile(0.75)
    iqr = q3 - q1

    outliers = (numeric_columns < (q1 - 1.5 * iqr)
                ) | (numeric_columns > (q3 + 1.5 * iqr))

    if method == "mask":
        data[numeric_columns.columns] = data[numeric_columns.columns].mask(
            outliers)
    elif method == "drop":
        data = data[~outliers.any(axis=1)]
    else:
        raise ValueError("Invalid method. Use 'mask' or 'drop'.")

    return data


def handle_missing_values(data, col_threshold):
    data = data.copy()

    for column in data.columns:
        missing_percentage = data[column].isnull().mean() * 100

        if missing_percentage > 0:
            if missing_percentage > col_threshold:
                data.drop(column, axis=1, inplace=True)
            else:
                if data[column].dtype == 'object' or data[column].dtype.name == 'category':
                    data[column] = data[column].fillna(
                        data[column].mode()[0])
                else:
                    skewness = data[column].skew()
                    if -0.5 <= skewness <= 0.5:
                        data[column] = data[column].fillna(
                            data[column].mean())
                    else:
                        data[column] = data[column].fillna(
                            data[column].median())

    return data


def remove_irrelevant_columns(data):
    data = data.copy()
    data.drop(columns=['Employee ID', 'Date of Joining'], inplace=True)
    return data


def drop_constant_columns(data):
    data = data.copy()
    data = data.loc[:, data.apply(pd.Series.nunique) != 1]
    return data


def remove_duplicates(data):
    data = data.copy()
    data.drop_duplicates(inplace=True)
    return data


def main():
    data = load_data('data/EmployeeBurnout_Raw.csv')

    print("\033[34mBefore Data Cleaning\033[0m\n")
    inspect(data)

    data = fix_data_types(data)
    data = handle_outliers(data, "mask")
    data = handle_missing_values(data, 20)
    data = remove_irrelevant_columns(data)
    data = drop_constant_columns(data)
    data = remove_duplicates(data)

    print("\033[34mAfter Data Cleaning\033[0m\n")
    inspect(data)

    save_data(data, 'data/EmployeeBurnout_Cleaned.csv')


if __name__ == "__main__":
    main()
