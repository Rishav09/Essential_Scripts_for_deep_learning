"""Author: Rishav Sapahia."""

from sklearn.model_selection import train_test_split
import pandas as pd
import datetime
import os
import argparse
import numpy as np


def split_equal_into_val_test(args):
    """
    Split a Pandas dataframe into two subsets (train, val).

    Following fractional ratios provided by the user, where val and
    test set have the same number of each classes while train set have
    the remaining number of left classes
    Parameters
    ----------
    csv_file : Input data csv file to be passed
    stratify_colname : str
        The name of the column that will be used for stratification. Usually
        this column would be for the label.
    frac_train : float
    frac_val   : float
        The ratios with which the dataframe will be split into train, val, and
        test data. The values should be expressed as float fractions and should
        sum to 1.0.
    random_state : int, None, or RandomStateInstance
        Value to be passed to train_test_split().

    Returns
    -------
    df_train, df_val :
        Dataframes containing the three splits.

    """
    df_input = pd.read_csv(args.csv_file, engine='python').iloc[:, :]
    split_train_file_name, split_train_file_extension = os.path.splitext(args.saved_train_file) # noqa
    split_valid_file_name, split_valid_file_extension = os.path.splitext(args.saved_valid_file) # noqa
    split_test_file_name, split_valid_file_extension = os.path.splitext(args.saved_test_file) # noqa

    if args.frac_train + args.frac_val + args.frac_test != 1.0:
        raise ValueError('fractions %f, %f, %f do not add up to 1.0' %
                         (args.frac_train, args.frac_val, args.frac_val))

    if args.stratify_colname not in df_input.columns:
        raise ValueError('%s is not a column in the dataframe' %
                         (args.stratify_colname))

    # Considering that split is 90% train and rest of it is valid and test.
    # Ugly hack to raise the val size equal to test size

    sfact = int(np.ceil(args.frac_val*len(df_input))/args.no_of_classes)

    # Shuffling the data frame
    df_input = df_input.sample(frac=1, random_state=42)


    # https://stackoverflow.com/questions/52279834/splitting-training-data-with-equal-number-rows-for-each-classes
    df_temp_1 = df_input[df_input[args.stratify_colname] == 0][:sfact]
    df_temp_2 = df_input[df_input[args.stratify_colname] == 1][:sfact]
    df_temp_3 = df_input[df_input[args.stratify_colname] == 2][:sfact]
    df_temp_4 = df_input[df_input[args.stratify_colname] == 3][:sfact]

    dev_test_df = pd.concat([df_temp_1, df_temp_2, df_temp_3, df_temp_4])

    dev_test_y = dev_test_df[args.stratify_colname]

    df_val, df_test, dev_Y, test_Y = train_test_split(
                                        dev_test_df, dev_test_y,
                                        stratify=dev_test_y,
                                        test_size=0.5,
                                         )

    if df_input[args.target_names].dtype == df_input[args.target_names].dtype:
        print("All elements in col1 are of the same type")
    else:
        print("Elements in col1 are not of the same type")

    # https://stackoverflow.com/questions/39880627/in-pandas-how-to-delete-rows-from-a-data-frame-based-on-another-data-frame
    df_train = df_input[~df_input[args.target_names].isin(dev_test_df[args.target_names])] # noqa
    current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    assert len(df_input) == len(df_train) + len(df_val)+len(df_test)
    dynamic_train_file_name = f"{split_train_file_name}_{current_time}{split_train_file_extension}" # noqa
    dynamic_valid_file_name = f"{split_valid_file_name}_{current_time}{split_valid_file_extension}" # noqa
    dynamic_valid_file_name = f"{split_test_file_name}_{current_time}{split_valid_file_extension}" # noqa

    # Currently Change this to pass through create_dataloader file
    df_train.to_csv(dynamic_train_file_name) # noqa
    df_val.to_csv(dynamic_valid_file_name)
    df_test.to_csv(dynamic_valid_file_name) # noqa
    return df_train, df_val, df_test


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("csv_file", help="input csv file")
    parser.add_argument("stratify_colname", help="Target Column Name")
    parser.add_argument("target_names", help="Target Personal Names")
    parser.add_argument("frac_train", type=float, help="Training Fraction")
    parser.add_argument("frac_val", type=float, help="Validation Fraction")
    parser.add_argument("frac_test", type=float, help="Test Fraction")
    parser.add_argument("saved_train_file", help="Saved the splitted train file")
    parser.add_argument("saved_valid_file", help="Saved the splitted valid file")
    parser.add_argument("saved_test_file", help="Saved the splitted test file")
    parser.add_argument("no_of_classes", type=int, help="No of classes")
    parser.add_argument("-v", "--verbose", help="increase output verbosity", action="store_true")
    args = parser.parse_args()

    split_equal_into_val_test(args)
