import argparse
import os

import numpy as np
import torch as t
from torch.optim import Adam

import sample
from utils.batch_loader import BatchLoader
from model.parameters import Parameters
from model.paraphraser import Paraphraser

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Paraphraser')
    parser.add_argument('--num-iterations', type=int, default=10, metavar='NI',
                        help='num iterations (default: 10)')
    parser.add_argument('--batch-size', type=int, default=4, metavar='BS',
                        help='batch size (default: 4)')
    parser.add_argument('--use-cuda', type=bool, default=False, metavar='CUDA',
                        help='use cuda (default: True)')
    parser.add_argument('--learning-rate', type=float, default=0.00005, metavar='LR',
                        help='learning rate (default: 0.00005)')
    parser.add_argument('--dropout', type=float, default=0.3, metavar='DR',
                        help='dropout (default: 0.3)')
    parser.add_argument('--use-trained', type=bool, default=False, metavar='UT',
                        help='load pretrained model (default: False)')
    parser.add_argument('--model-name', default='', metavar='MN',
                        help='name of model to save (default: "")')
    parser.add_argument('--weight-decay', default=0.0, type=float, metavar='WD',
                        help='L2 regularization penalty (default: 0.0)')
    parser.add_argument('--use-quora', default=False, type=bool, metavar='quora',
                    help='if include quora dataset (default: False)')
    parser.add_argument('--interm-sampling', default=False, type=bool, metavar='IS',
                    help='if sample while training (default: False)')
    parser.add_argument('--use_two_path_loss', default=False, type=bool, metavar='2PL',
                    help='use two path loss while training (default: False)')
    parser.register('type', 'bool', lambda v: v.lower() in ["true", "t", "1"])
    args = parser.parse_args()

    batch_loader = BatchLoader()
    parameters = Parameters(batch_loader.max_seq_len,
                            batch_loader.vocab_size,
                            True)

    paraphraser = Paraphraser(parameters)
    ce_result_valid = []
    kld_result_valid = []
    ce_result_train = []
    kld_result_train = []
    ce_cur_train = []
    kld_cur_train = []

    optimizer = Adam(paraphraser.learnable_parameters(), args.learning_rate,
        weight_decay=args.weight_decay)

    train_step = paraphraser.trainer(optimizer, batch_loader)
    validate = paraphraser.validater(batch_loader)

    train_step(0, args.batch_size, args.use_cuda, args.dropout)

    #
    # for iteration in range(args.num_iterations):
    #
    #     cross_entropy, kld, coef = train_step(iteration, args.batch_size, args.use_cuda, args.dropout)
    #
    #     ce_cur_train += [cross_entropy.data.cpu().numpy()]
    #     kld_cur_train += [kld.data.cpu().numpy()]
    #
    #     # validation
    #     if iteration % 500 == 0:
    #         ce_result_train += [np.mean(ce_cur_train)]
    #         kld_result_train += [np.mean(kld_cur_train)]
    #         ce_cur_train, kld_cur_train = [], []
    #
    #         print('\n')
    #         print('------------TRAIN-------------')
    #         print('----------ITERATION-----------')
    #         print(iteration)
    #         print('--------CROSS-ENTROPY---------')
    #         print(ce_result_train[-1])
    #         print('-------------KLD--------------')
    #         print(kld_result_train[-1])
    #         print('-----------KLD-coef-----------')
    #         print(coef)
    #         print('------------------------------')
    #
    #
    #         # averaging across several batches
    #         cross_entropy, kld = [], []
    #         for i in range(20):
    #             ce, kl, _ = validate(args.batch_size, args.use_cuda)
    #             cross_entropy += [ce.data.cpu().numpy()]
    #             kld += [kl.data.cpu().numpy()]
    #
    #         kld = np.mean(kld)
    #         cross_entropy = np.mean(cross_entropy)
    #         ce_result_valid += [cross_entropy]
    #         kld_result_valid += [kld]
    #
    #         print('\n')
    #         print('------------VALID-------------')
    #         print('--------CROSS-ENTROPY---------')
    #         print(cross_entropy)
    #         print('-------------KLD--------------')
    #         print(kld)
    #         print('------------------------------')
    #
    #         _, _, (sampled, s1, s2) = validate(2, args.use_cuda, need_samples=True)
    #
    #         for i in range(len(sampled)):
    #             result = paraphraser.sample_with_pair(batch_loader, 20, args.use_cuda, s1[i], s2[i])
    #             print('source: ' + s1[i])
    #             print('target: ' + s2[i])
    #             print('valid: ' + sampled[i])
    #             print('sampled: ' + result)
    #             print('...........................')
