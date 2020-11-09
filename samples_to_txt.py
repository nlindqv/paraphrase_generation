import argparse
import os

import numpy as np
import torch as t
from torch.optim import Adam

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Paraphraser')
    parser.add_argument('--model-name', default='', metavar='MN',
                        help='name of model to save (default: "")')
    parser.add_argument('--iteration', default='', metavar='MN',
                        help='name of model to save (default: "")')
    args = parser.parse_args()

    input_file = 'quora_test'

    if args.iteration != '':
        sampled_file_dst = 'logs/intermediate/sampled_out_{}k_{}{}.txt.npy'.format(
                                    args.iteration, input_file, args.model_name)
        target_file_dst = 'logs/intermediate/target_out_{}k_{}{}.txt.npy'.format(
                                    args.iteration, input_file, args.model_name)
        source_file_dst = 'logs/intermediate/source_out_{}k_{}{}.txt.npy'.format(
                                    args.iteration, input_file, args.model_name)
        result = list(np.load(sampled_file_dst, allow_pickle=True))
        target = list(np.load(target_file_dst, allow_pickle=True))
        source = list(np.load(source_file_dst, allow_pickle=True))
    else:
        sampled_file_dst = 'logs/sampled_out_{}.txt'.format(args.model_name)
        target_file_dst = 'logs/target_out_{}.txt'.format(args.model_name)
        source_file_dst = 'logs/source_out_{}.txt'.format(args.model_name)

        result = list(np.loadtxt(sampled_file_dst, dtype='U', delimiter='\n'))
        target = list(np.loadtxt(target_file_dst, dtype='U', delimiter='\n'))
        source = list(np.loadtxt(source_file_dst, dtype='U', delimiter='\n'))


    for i in range(4000):
        print(f'Sentence pair {i}:')
        if args.iteration == '':
            print('source : ', source[i])
            print('target : ', target[i])
            print('sampled (w/o ref) : ', result[i])
        else:
            print('source : ', ' '.join(source[i]))
            print('target : ', ' '.join(target[i]))
            print('sampled (w/ ref) : ', result[i])
        print('\n')
