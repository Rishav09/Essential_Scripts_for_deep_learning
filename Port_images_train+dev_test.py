"""Author: Rishav Sapahia."""

import argparse
import os
import shutil
import pandas as pd


def port_images(args):
    """
    Reads in a CSV file, target names, folder path and moves any files listed in the source.
    folder to a specified training(Containing valid set also) and test folder.

    Args:
    csv_file (str): Path to the CSV file containing the list of files to move.
    source_folder (str): Path to the folder where all the files are.
    train_folder (str): Path to the folder where the train files are.
    test_folder (str): Path to the folder where the test files are.


    Returns:
    None

    Raises:
    ValueError: If the CSV file or folder path is invalid.
    IOError: If there is an error reading the CSV file or moving the files.
    """
    # Check that the CSV file and folder paths are valid
    df = pd.read_csv(args.csv_file, engine='python').iloc[:, :]

    if not os.path.isdir(args.source_folder):
        raise ValueError(f'Invalid source folder path: {args.source_folder}')
    if not os.path.isdir(args.train_folder):
        raise ValueError(f'Invalid train folder path: {args.train_folder}')
    if not os.path.isdir(args.test_folder):
        raise ValueError(f'Invalid test folder path: {args.test_folder}')
    if args.target_names not in df.columns:
        raise ValueError('%s is not a column in the dataframe' %
                         (args.target_names))

    # Main operation now
    source_files = df[args.target_names].to_list()

    for each_file in source_files:
        src_file = os.path.join(args.source_folder, each_file)
        dst_file = os.path.join(args.train_folder, each_file)

        if os.path.exists(src_file):
            shutil.copy(src_file, dst_file)
        else:
            continue

if __name__ == '__main__':
    # Parse command line arguments

    parser = argparse.ArgumentParser(description='Process CSV file and move files to a folder.')
    parser.add_argument('csv_file', type=str, help='Path to the CSV file.')
    parser.add_argument("target_names", help="Target Personal Names")
    parser.add_argument('source_folder', type=str, help='Path to the folder where the files are.')
    parser.add_argument('train_folder', type=str, help='Path to the folder where the train & valid files are.')
    parser.add_argument('test_folder', type=str, help='Path to the folder where the test files are.')

    args = parser.parse_args()
    port_images(args)