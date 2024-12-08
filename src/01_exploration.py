from utils import *


def examine_categorical_columns(data):
    print("\033[33mData Value Counts (Employee ID)\033[0m")
    print(data['Employee ID'].value_counts(), "\n")

    print("\033[33mData Value Counts (Date of Joining)\033[0m")
    print(data['Date of Joining'].value_counts(), "\n")

    print("\033[33mData Value Counts (Gender)\033[0m")
    print(data['Gender'].value_counts(), "\n")

    print("\033[33mData Value Counts (Company Type)\033[0m")
    print(data['Company Type'].value_counts(), "\n")

    print("\033[33mData Value Counts (WFH Setup Available)\033[0m")
    print(data['WFH Setup Available'].value_counts(), "\n")


def main():
    data = load_data('data/EmployeeBurnout_Raw.csv')

    inspect_basic_structure(data)
    examine_summary_statistics(data)
    examine_categorical_columns(data)
    identify_missing_values(data)
    check_for_duplicates(data)
    check_for_constant_columns(data)
    check_data_types(data)
    detect_outliers(data)


if __name__ == "__main__":
    main()
