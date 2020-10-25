# -*- coding: utf-8 -*-
import argparse
import os
import sys
import time

import numpy as np
import torch as t
from torch.optim import Adam
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import sample
from utils.batch_loader import BatchLoader
from utils.rollout import Rollout
from model.parametersGAN import Parameters
from model.generator import Generator
from model.discriminator import Discriminator
from model.paraphraser import Paraphraser

lambdas = [0.5, 0.5, 0.01]
rollout_num = 4

def trainer(generator, g_optim, discriminator, d_optim, rollout, batch_loader):
    def train(i, batch_size, use_cuda, dropout):
        input = batch_loader.next_batch(batch_size, 'train')
        input = [var.cuda() if use_cuda else var for var in input]

        [encoder_input_source,
         encoder_input_target,
         decoder_input_source,
         decoder_input_target, target] = input

        (logits, logits2), _, kld = generator(dropout,
                (encoder_input_source, encoder_input_target),
                (decoder_input_source, decoder_input_target),
                z=None, use_cuda=use_cuda)

        target = target.view(-1)

        logits = logits.view(-1, generator.params.vocab_size)
        logits2 = logits.view(-1, generator.params.vocab_size)
        ce_1 = F.cross_entropy(logits, target)
        ce_2 = F.cross_entropy(logits2, target)


        # Generate fake data
        prediction = F.softmax(logits, dim=-1)
        samples = prediction.multinomial(1).view(batch_size, -1)
        gen_samples = batch_loader.embed_batch_from_index(samples)

        if use_cuda:
            gen_samples = gen_samples.cuda()

        t0 = time.time_ns()

        rewards = rollout.reward(gen_samples, [encoder_input_source, encoder_input_target], decoder_input_source, use_cuda, batch_loader)
        rewards = Variable(t.tensor(rewards))
        if use_cuda:
            rewards = rewards.cuda()
        neg_lik = F.cross_entropy(logits, target, reduction='none')

        dg_loss = t.mean(neg_lik * rewards.flatten())

        print(f'Time through rollout (4): {(time.time_ns() - t0) / (10 ** 6)} ms')
        print(f'with seq_len: {gen_samples.size()[1]}')
        t0 = time.time_ns()


        g_loss = lambda1 * ce_1 + lambda1 * kld + lambda2 * ce_2
        # generator.params.cross_entropy_penalty_weight * (cross_entropy + cross_entropy2) \
        # + generator.params.get_kld_coef(i) * kld

        g_optim.zero_grad()
        g_loss.backward()
        t.nn.utils.clip_grad_norm_(generator.learnable_parameters(), 10)
        g_optim.step()

        # Train discriminator with real and fake data
        data = t.cat([encoder_input_target, gen_samples], dim=0)

        labels = t.zeros(2*batch_size)
        labels[:batch_size] = 1

        if use_cuda:
            labels = labels.cuda()
            data = data.cuda()

        d_logits = discriminator(data)
        d_loss = F.binary_cross_entropy_with_logits(d_logits, labels)

        d_optim.zero_grad()
        d_loss.backward()
        t.nn.utils.clip_grad_norm_(discriminator.learnable_parameters(), 5)
        d_optim.step()

        return (ce_1, ce_2), kld, generator.params.get_kld_coef(i)

    return train

def validater(generator, discriminator, batch_loader):
    def get_samples(logits, target):
        '''
        logits: [batch, seq_len, vocab_size]
        targets: [batch, seq_len]
        '''
        prediction = F.softmax(logits, dim=-1).data.cpu().numpy()

        target = target.data.cpu().numpy()

        sampled, expected = [], []
        for i in range(prediction.shape[0]):
            sampled  += [' '.join([batch_loader.sample_word_from_distribution(d)
                for d in prediction[i]])]
            expected += [' '.join([batch_loader.get_word_by_idx(idx) for idx in target[i]])]

        return sampled, expected

    def validate(batch_size, use_cuda, need_samples=False):
        if need_samples:
            input, sentences = batch_loader.next_batch(batch_size, 'test', return_sentences=True)
            sentences = [[' '.join(s) for s in q] for q in sentences]
        else:
            input = batch_loader.next_batch(batch_size, 'test')

        input = [var.cuda() if use_cuda else var for var in input]

        [encoder_input_source,
         encoder_input_target,
         decoder_input_source,
         decoder_input_target, target] = input

        (logits, logits2), _, kld = generator(0., (encoder_input_source, encoder_input_target),
                                (decoder_input_source, decoder_input_target),
                                z=None, use_cuda=use_cuda)

        if need_samples:
            [s1, s2] = sentences
            sampled, _ = get_samples(logits, target)
        else:
            s1, s2 = (None, None)
            sampled, _ = (None, None)


        target = target.view(-1)

        cross_entropy, cross_entropy2 = [], []

        logits = logits.view(-1, generator.params.vocab_size)
        cross_entropy = F.cross_entropy(logits, target)

        logits2 = logits2.view(-1, generator.params.vocab_size)
        cross_entropy2 = F.cross_entropy(logits2, target)
        # cross_entropy2 = 0
        return (cross_entropy, cross_entropy2), kld, (sampled, s1, s2)

    return validate

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Paraphraser')
    parser.add_argument('--num-iterations', type=int, default=300000, metavar='NI',
                        help='num iterations (default: 60000)')
    parser.add_argument('--batch-size', type=int, default=32, metavar='BS',
                        help='batch size (default: 32)')
    parser.add_argument('-cuda', '--use-cuda', type=bool, default=False, metavar='CUDA',
                        help='use cuda (default: True)')
    parser.add_argument('--learning-rate', type=float, default=0.00005, metavar='LR',
                        help='learning rate (default: 0.00005)')
    parser.add_argument('--dropout', type=float, default=0.3, metavar='DR',
                        help='dropout (default: 0.3)')
    parser.add_argument('-ut', '--use-trained', type=bool, default=False, metavar='UT',
                        help='load pretrained model (default: False)')
    parser.add_argument('-m', '--model-name', default='', metavar='MN',
                        help='name of model to save (default: "")')
    parser.add_argument('--weight-decay', default=0.0, type=float, metavar='WD',
                        help='L2 regularization penalty (default: 0.0)')
    parser.add_argument('--use-quora', default=False, type=bool, metavar='quora',
                    help='if include quora dataset (default: False)')
    parser.add_argument('--interm-sampling', default=True, type=bool, metavar='IS',
                    help='if sample while training (default: False)')
    parser.add_argument('-tpl', '--use_two_path_loss', default=False, type=bool, metavar='2PL',
                    help='use two path loss while training (default: False)')
    parser.register('type', 'bool', lambda v: v.lower() in ["true", "t", "1"])
    args = parser.parse_args()

    batch_loader = BatchLoader()
    parameters = Parameters(batch_loader.max_seq_len,
                            batch_loader.vocab_size)

    generator = Generator(parameters)
    discriminator = Discriminator(parameters)

    ce_result_valid = []
    kld_result_valid = []
    ce_result_train = []
    kld_result_train = []
    ce_cur_train = []
    kld_cur_train = []
    if args.use_two_path_loss:
        ce2_cur_train = []
        ce2_result_train = []
        ce2_result_valid = []

    if args.use_trained:
        paraphraser.load_state_dict(t.load('saved_models/trained_paraphraser_' + args.model_name))
        ce_result_valid = list(np.load('logs/ce_result_valid_{}.npy'.format(args.model_name)))
        kld_result_valid = list(np.load('logs/kld_result_valid_{}.npy'.format(args.model_name)))
        ce_result_train = list(np.load('logs/ce_result_train_{}.npy'.format(args.model_name)))
        kld_result_train = list(np.load('logs/kld_result_train_{}.npy'.format(args.model_name)))
        if args.use_two_path_loss:
            ce2_result_valid = list(np.load('logs/ce2_result_valid_{}.npy'.format(args.model_name)))
            ce2_result_train = list(np.load('logs/ce2_result_train_{}.npy'.format(args.model_name)))


    if args.use_cuda:
        generator = generator.cuda()
        discriminator = discriminator.cuda()

    g_optim = Adam(generator.learnable_parameters(), args.learning_rate)
    d_optim = Adam(discriminator.learnable_parameters(), args.learning_rate)

    rollout = Rollout(generator, discriminator, 0.8, rollout_num)

    train_step = trainer(generator, g_optim, discriminator, d_optim, rollout, batch_loader)
    validate = validater(generator, discriminator, batch_loader)

    start = time.time_ns()
    for iteration in range(args.num_iterations):
        if iteration <= 10000:
            lambda3 = iteration / (1. * 10000) * lambdas[2]
            lambda2 = iteration / (1. * 10000) * lambdas[1]
            lambda1 = 1 - lambda2


        (cross_entropy, cross_entropy2), kld, coef = train_step(iteration, args.batch_size, args.use_cuda, args.dropout)
        if iteration % 10 == 0:
            print(f'Time per iteration: {((time.time_ns() - start) / (10 ** 6)) / 10} ms')
            start = time.time_ns()

        ce_cur_train += [cross_entropy.data.cpu().numpy()]
        kld_cur_train += [kld.data.cpu().numpy()]
        if args.use_two_path_loss:
            ce2_cur_train += [cross_entropy2.data.cpu().numpy()]

        # validation
        if iteration % 500 == 0:
            ce_result_train += [np.mean(ce_cur_train)]
            kld_result_train += [np.mean(kld_cur_train)]
            ce_cur_train, kld_cur_train = [], []

            if args.use_two_path_loss:
                ce2_result_train += [np.mean(ce2_cur_train)]
                ce2_cur_train = []

            print('\n')
            print('------------TRAIN-------------')
            print('----------ITERATION-----------')
            print(iteration)
            print('--------CROSS-ENTROPY---------')
            print(ce_result_train[-1])
            if args.use_two_path_loss:
                print('----CROSS-ENTROPY-2ND PATH----')
                print(ce2_result_train[-1])
            print('-------------KLD--------------')
            print(kld_result_train[-1])
            print('-----------KLD-coef-----------')
            print(coef)
            print('------------------------------')


            # averaging across several batches
            cross_entropy, cross_entropy2, kld = [], [], []
            for i in range(20):
                (ce, ce2), kl, _ = validate(args.batch_size, args.use_cuda)
                cross_entropy += [ce.data.cpu().numpy()]
                kld += [kl.data.cpu().numpy()]

                if args.use_two_path_loss:
                    cross_entropy2 += [ce2.data.cpu().numpy()]


            kld = np.mean(kld)
            cross_entropy = np.mean(cross_entropy)
            ce_result_valid += [cross_entropy]
            kld_result_valid += [kld]
            if args.use_two_path_loss:
                cross_entropy2 = np.mean(cross_entropy2)
                ce2_result_valid += [cross_entropy2]

            print('\n')
            print('------------VALID-------------')
            print('--------CROSS-ENTROPY---------')
            print(cross_entropy)
            if args.use_two_path_loss:
                print('----CROSS-ENTROPY-2ND-PATH----')
                print(cross_entropy2)
            print('-------------KLD--------------')
            print(kld)
            print('------------------------------')

            _, _, (sampled, s1, s2) = validate(2, args.use_cuda, need_samples=True)

            for i in range(len(sampled)):
                result = generator.sample_with_pair(batch_loader, 20, args.use_cuda, s1[i], s2[i])

                print('source: ' + s1[i])
                print('target: ' + s2[i])
                print('valid: ' + sampled[i])
                print('sampled: ' + result)
                if args.use_two_path_loss:
                    result2 = generator.sample_with_pair(batch_loader, 20, args.use_cuda, s1[i], s2[i], input_only=True)
                    print('sampled 2: ' + result2)
                print('...........................')

        # save model
        if (iteration % 10000 == 0 and iteration != 0) or iteration == (args.num_iterations - 1):
            t.save(generator.state_dict(), 'saved_models/trained_paraphraser_' + args.model_name)
            np.save('logs/ce_result_valid_{}.npy'.format(args.model_name), np.array(ce_result_valid))
            np.save('logs/ce_result_valid_{}.npy'.format(args.model_name), np.array(ce_result_valid))
            np.save('logs/kld_result_valid_{}'.format(args.model_name), np.array(kld_result_valid))
            np.save('logs/ce_result_train_{}.npy'.format(args.model_name), np.array(ce_result_train))
            np.save('logs/kld_result_train_{}'.format(args.model_name), np.array(kld_result_train))
            if args.use_two_path_loss:
                np.save('logs/ce2_result_valid_{}.npy'.format(args.model_name), np.array(ce2_result_valid))
                np.save('logs/ce2_result_train_{}.npy'.format(args.model_name), np.array(ce2_result_train))


        #interm sampling
        if (iteration % 20000 == 0 and iteration != 0) or iteration == (args.num_iterations - 1):
            if args.interm_sampling:
                sample_file = 'quora_test'
                args.use_mean = False
                args.seq_len = 30

                (result, result2), target, source = sample.sample_with_input_file(batch_loader,
                                generator, args, sample_file, args.use_two_path_loss)

                sampled_file_dst = 'logs/intermediate/sampled_out_{}k_{}{}.txt'.format(
                                            iteration//1000, sample_file, args.model_name)
                target_file_dst = 'logs/intermediate/target_out_{}k_{}{}.txt'.format(
                                            iteration//1000, sample_file, args.model_name)
                source_file_dst = 'logs/intermediate/source_out_{}k_{}{}.txt'.format(
                                            iteration//1000, sample_file, args.model_name)
                np.save(sampled_file_dst, np.array(result))
                np.save(target_file_dst, np.array(target))
                np.save(source_file_dst, np.array(source))

                if args.use_two_path_loss:
                    sampled2_file_dst = 'logs/intermediate/sampled2_out_{}k_{}{}.txt'.format(
                                                iteration//1000, sample_file, args.model_name)
                    np.save(sampled2_file_dst, np.array(result))

                print('------------------------------')
                print('results saved to: ')
                print(sampled_file_dst)
                if args.use_two_path_loss:
                    print(sampled2_file_dst)
                print(target_file_dst)
                print(source_file_dst)
