import os
import numpy as np
import pandas as pd
import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Preprocess Data')
    parser.add_argument('-s', '--source', default='data/questions_sv_filtered_all.csv', help='Filename of data')
    parser.add_argument('-rs', '--random-state', default=None, help='Seed to use when sampling')
    parser.add_argument('-test', '--test', default=4000, help='Size of the test set')
    parser.add_argument('-train', '--train', default=[50000, 100000], help='Size of the test set')
    parser.add_argument('-d', '--destination', default='datasets/', help='Translate dataset (default: True)')

    args = parser.parse_args()

    if not os.path.exists(args.source):
        raise FileNotFoundError("Data file wasn't found")

    # Load data and filter duplicates
    data = pd.read_csv(args.source, index_col = 0)

    # Permute data
    data = data.sample(frac=1, random_state=args.random_state)

    # Divide sets and save to files
    test = data[:args.test]
    test.to_csv(args.destination + 'test.csv')

    idx_s = args.test + args.train[0]
    train_s = data[args.test:idx_s]
    train_s.to_csv(args.destination + 'train50k.csv')

    idx_m = args.test + args.train[1]
    train_m = data[args.test:idx_m]
    train_m.to_csv(args.destination + 'train100k.csv')

    train_l = data[args.test:]
    train_l.to_csv(args.destination + 'train140k.csv')
