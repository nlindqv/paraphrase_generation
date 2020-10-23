# -*- coding: utf-8 -*-
import argparse
import os

import numpy as np
import torch as t
from torch.optim import Adam
from torch.autograd import Variable
import torch.nn.functional as F

import sample
from utils.batch_loader import BatchLoader
from utils.rollout import Rollout
from model.parametersGAN import Parameters
from model.generator import Generator
from model.discriminator import Discriminator
from model.paraphraser import Paraphraser
# iterations = 200000
# use_trained = False
# converged = False
# lr = 0.0001
# warmup_step = 10000
# batch_size = 32
# use_cuda = False
lambdas = [0.5, 0.5, 0.01]
# dropout = 0.3

def trainer(discriminator, generator, rollout, d_optimizer, g_optimizer, batch_loader):
    def train(i, batch_size, use_cuda, dropout, lambda1, lambda2, lambda3):
        # Sample a batch of {s_o, s_p} from dataset
        input = batch_loader.next_batch(batch_size, 'train')
        input = [var.cuda() if use_cuda else var for var in input]

        [encoder_input_source,
         encoder_input_target,
         decoder_input_source,
         decoder_input_target, target] = input

        # Train Generator
        logits, _, kld = generator(dropout,
                (encoder_input_source, encoder_input_target),
                (decoder_input_source, decoder_input_target),
                z=None, use_cuda=use_cuda)

        logits = logits.view(2, -1, generator.params.vocab_size)
        target = target.view(-1)
        ce_1 = F.cross_entropy(logits[0], target)
        ce_2 = F.cross_entropy(logits[1], target)

        # Sample a sequence to feed into discriminator
        # prediction = [batch_size, seq_len, vocab_size]
        prediction = F.softmax(logits[0].view(batch_size, -1, generator.params.vocab_size), dim=-1).data.cpu().numpy()

        samples = []
        for i in range(prediction.shape[0]):
            samples.append([batch_loader.sample_word_from_distribution(d)
                for d in prediction[i]])
        # samples = [batch_size, seq_len]
        # gen_samples = [batch_size, max_seq_len, 300]

        gen_samples = batch_loader.embed_batch(samples)
        gen_samples = t.Tensor(gen_samples)
        if use_cuda:
            gen_samples = gen_samples.cuda()

        rewards = rollout.reward(gen_samples, 8, input, use_cuda, batch_loader)
        rewards = Variable(t.tensor(rewards))
        if use_cuda:
            rewards = rewards.cuda()
        neg_lik = F.cross_entropy(logits[0], target, size_average=False, reduce=False)

        dg_loss = t.mean(neg_lik * rewards.flatten())

        g_loss = lambda1 * ce_1 \
                + lambda2 * ce_2 \
                + lambda3 * dg_loss \
                + lambda1 * kld

        g_optimizer.zero_grad()
        g_loss.backward()
        t.nn.utils.clip_grad_norm_(generator.learnable_parameters(), 10)
        g_optimizer.step()

        # Train discriminator with real and fake data
        data = t.cat([encoder_input_target, gen_samples], dim=0)

        labels = t.zeros(2*batch_size)
        labels[:batch_size] = 1
        if use_cuda:
            labels = labels.cuda()
            data = data.cuda()

        d_logits = discriminator(data)
        d_loss = F.binary_cross_entropy_with_logits(d_logits, labels)

        d_optimizer.zero_grad()
        d_loss.backward()
        t.nn.utils.clip_grad_norm_(discriminator.learnable_parameters(), 5)
        d_optimizer.step()

        return (ce_1, ce_2, dg_loss, d_loss), kld

    return train


def validate(batch_loader, batch_size, use_cuda, need_samples=False):
    # Sample a batch of {s_o, s_p} from dataset
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

    # Train Generator
    logits, _, kld = generator(0.,
            (encoder_input_source, encoder_input_target),
            (decoder_input_source, decoder_input_target),
            z=None, use_cuda=use_cuda)

    logits = logits.view(2, -1, parameters.vocab_size)
    target = target.view(-1)
    ce_1 = F.cross_entropy(logits[0], target)
    ce_2 = F.cross_entropy(logits[1], target)

    # Sample a sequence to feed into discriminator
    # prediction = [batch_size, seq_len, vocab_size]
    prediction = F.softmax(logits[0].view(batch_size, -1, parameters.vocab_size), dim=-1).data.cpu().numpy()

    samples = []
    if need_samples:
        [s1, s2] = sentences
        sampled = []
    else:
        [s1, s2] = (None, None)
        sampled = None
    for i in range(prediction.shape[0]):
        samples.append([batch_loader.sample_word_from_distribution(d)
            for d in prediction[i]])
        if need_samples:
            sampled  += [' '.join([batch_loader.sample_word_from_distribution(d)
                             for d in prediction[i]])]

    gen_samples = batch_loader.embed_batch(samples)
    gen_samples = t.Tensor(gen_samples)

    rewards = rollout.reward(gen_samples, 2, input, use_cuda, batch_loader)
    rewards = Variable(t.tensor(rewards))
    neg_lik = F.cross_entropy(logits[0], target, size_average=False, reduce=False)

    dg_loss = t.mean(neg_lik * rewards.flatten())

    # Train discriminator with real and fake data
    data = t.cat([encoder_input_target, gen_samples], dim=0)

    labels = t.zeros(2*batch_size)
    labels[:batch_size] = 1

    d_logits = discriminator(data)
    d_loss = F.binary_cross_entropy_with_logits(d_logits, labels)

    return (ce_1, ce_2, kld, dg_loss, d_loss), (sampled, s1, s2)







if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Paraphraser')
    parser.add_argument('--num-iterations', type=int, default=300000, metavar='NI',
                        help='num iterations (default: 60000)')
    parser.add_argument('--batch-size', type=int, default=32, metavar='BS',
                        help='batch size (default: 32)')
    parser.add_argument('-cuda', '--use-cuda', type=bool, default=True, metavar='CUDA',
                        help='use cuda (default: True)')
    parser.add_argument('--learning-rate', type=float, default=0.0001, metavar='LR',
                        help='learning rate (default: 0.0001)')
    parser.add_argument('--dropout', type=float, default=0.3, metavar='DR',
                        help='dropout (default: 0.3)')
    parser.add_argument('-ut', '--use-trained', type=bool, default=False, metavar='UT',
                        help='load pretrained model (default: False)')
    parser.add_argument('-m', '--model-name', default='', metavar='MN',
                        help='name of model to save (default: "")')
    parser.add_argument('--warmup-step', default=10000, type=float, metavar='WS',
                        help='L2 regularization penalty (default: 0.0)')
    parser.add_argument('--use-quora', default=False, type=bool, metavar='quora',
                    help='if include quora dataset (default: False)')
    parser.add_argument('--interm-sampling', default=True, type=bool, metavar='IS',
                    help='if sample while training (default: False)')

    parser.register('type', 'bool', lambda v: v.lower() in ["true", "t", "1"])
    args = parser.parse_args()

    batch_loader = BatchLoader()
    parameters = Parameters(batch_loader.max_seq_len,
                            batch_loader.vocab_size)

    # Loss main path
    ce_result_valid, ce_result_train, ce_cur_train = [], [], []
    # Loss second path
    ce2_result_valid, ce2_result_train, ce2_cur_train = [], [], []
    # KLD loss
    kld_result_valid, kld_result_train, kld_cur_train = [], [], []
    # Generator-discriminator loss
    dg_result_valid, dg_result_train, dg_cur_train = [], [], []
    # Discriminator loss
    d_result_valid, d_result_train, d_cur_train = [], [], []



    print(f'Initiate generator...')
    generator = Generator(parameters)
    print(f'Initiate discriminator...')
    discriminator = Discriminator(parameters)

    if args.use_cuda:
        generator = generator.cuda()
        discriminator = discriminator.cuda()

    if args.use_trained:
        generator.load_state_dict(t.load('saved_models/trained_generator_' + args.model_name))
        discriminator.load_state_dict(t.load('saved_models/trained_discriminator_' + args.model_name))

    print(f'Initiate optimizers...')
    g_optim = Adam(generator.learnable_parameters(), args.learning_rate)
    d_optim = Adam(discriminator.learnable_parameters(), args.learning_rate)

    print(f'Initiate rollout...')
    rollout = Rollout(generator, discriminator, 0.8)
    if use_cuda:
        rllout = rollout.cuda()

    # g_criterion = nn.CrossEntropyLoss()
    # d_criterion = nn.BCEWithLogitsLoss()
    train_step = trainer(discriminator, generator, rollout, d_optim, g_optim, batch_loader)

    print(f'Start adversarial training...')
    for iteration in range(args.num_iterations):
        # Warmup training
        if iteration <= args.warmup_step:
            lambda3 = iteration / (1. * args.warmup_step) * lambdas[2]
            lambda2 = iteration / (1. * args.warmup_step) * lambdas[1]
            lambda1 = 1 - lambda2

        (ce_1, ce_2, dg_loss, d_loss), kld = train_step(iteration, args.batch_size, args.use_cuda, args.dropout, lambda1, lambda2, lambda3)


        # Store losses
        ce_cur_train += [ce_1.data.cpu().numpy()]
        ce2_cur_train += [ce_2.data.cpu().numpy()]
        kld_cur_train += [kld.data.cpu().numpy()]
        dg_cur_train += [dg_loss.data.cpu().numpy()]
        d_cur_train += [d_loss.data.cpu().numpy()]

        # validation
        if iteration % 500 == 0:
            ce_result_train += [np.mean(ce_cur_train)]
            ce2_result_train += [np.mean(ce2_cur_train)]
            kld_result_train += [np.mean(kld_cur_train)]
            dg_result_train += [np.mean(dg_cur_train)]
            d_result_train += [np.mean(d_cur_train)]


            ce_cur_train, ce2_cur_train, kld_cur_train, dg_cur_train, d_cur_train = [], [], [], [], []

            print('\n')
            print('------------TRAIN-------------')
            print('----------ITERATION-----------')
            print(iteration)
            print('--------CROSS-ENTROPY---------')
            print(f'{ce_result_train[-1]}\t (lambda1: {lambda1})')
            print('----CROSS-ENTROPY-2ND PATH----')
            print(f'{ce2_result_train[-1]}\t (lambda2: {lambda2})')
            print('--------------DG--------------')
            print(f'{dg_result_train[-1]}\t (lambda3: {lambda3})')
            print('-------------KLD--------------')
            print(f'{kld_result_train[-1]}\t (lambda1: {lambda1})')
            print('-------Discriminator----------')
            print(f'{d_result_train[-1]}')
            print('------------------------------')


            # averaging across several batches
            ce_1, ce_2, kld, dg_loss, d_loss = [], [], [], [], []
            for i in range(20):

                (c1, c2, kl, dg, d), _ = validate(batch_loader, args.batch_size, args.use_cuda)
                ce_1 += [c1.data.cpu().numpy()]
                ce_2 += [c2.data.cpu().numpy()]
                kld += [kl.data.cpu().numpy()]
                dg_loss += [dg.data.cpu().numpy()]
                d_loss += [d.data.cpu().numpy()]


            ce_1 = np.mean(ce_1)
            ce_2 = np.mean(ce_2)
            kld = np.mean(kld)
            dg_loss = np.mean(dg_loss)
            d_loss = np.mean(d_loss)

            ce_result_valid += [ce_1]
            ce2_result_valid += [ce_2]
            kld_result_valid += [kld]
            dg_result_valid += [dg_loss]
            d_result_valid += [d_loss]

            print('\n')
            print('------------VALID-------------')
            print('--------CROSS-ENTROPY---------')
            print(ce_1)
            print('----CROSS-ENTROPY-2ND-PATH----')
            print(ce_2)
            print('--------------DG--------------')
            print(dg_loss)
            print('-------------KLD--------------')
            print(kld)
            print('-------Discriminator----------')
            print(d_loss)
            print('------------------------------')

            _, (sampled, s1, s2) = validate(batch_loader, 2, args.use_cuda, need_samples=True)

            for i in range(len(sampled)):
                result = generator.sample_with_pair(batch_loader, 20, args.use_cuda, s1[i], s2[i])
                result2 = generator.sample_with_pair(batch_loader, 20, args.use_cuda, s1[i], s2[i], input_only=True)

                print('source: ' + s1[i])
                print('target: ' + s2[i])
                print('valid: ' + sampled[i])
                print('sampled: ' + result)
                print('sampled (no ref): ' + result2)
                print('...........................')

        # save model
        if (iteration % 10000 == 0 and iteration != 0) or iteration == (args.num_iterations - 1):
            t.save(generator.state_dict(), 'saved_models/trained_generator_' + args.model_name)
            t.save(discriminator.state_dict(), 'saved_models/trained_discrminator_' + args.model_name)
            np.save('logs/ce_result_valid_{}.npy'.format(args.model_name), np.array(ce_result_valid))
            np.save('logs/ce_result_train_{}.npy'.format(args.model_name), np.array(ce_result_train))
            np.save('logs/kld_result_valid_{}'.format(args.model_name), np.array(kld_result_valid))
            np.save('logs/kld_result_train_{}'.format(args.model_name), np.array(kld_result_train))
            np.save('logs/ce2_result_valid_{}.npy'.format(args.model_name), np.array(ce2_result_valid))
            np.save('logs/ce2_result_train_{}.npy'.format(args.model_name), np.array(ce2_result_train))
            np.save('logs/dg_result_valid_{}.npy'.format(args.model_name), np.array(dg_result_valid))
            np.save('logs/dg_result_train_{}.npy'.format(args.model_name), np.array(dg_result_train))
            np.save('logs/d_result_valid_{}.npy'.format(args.model_name), np.array(d_result_valid))
            np.save('logs/d_result_train_{}.npy'.format(args.model_name), np.array(d_result_train))

        #interm sampling
        if (iteration % 20000 == 0 and iteration != 0) or iteration == (args.num_iterations - 1):
            if args.interm_sampling:
                sample_file = 'quora_test'
                args.use_mean = False
                args.seq_len = 30

                (result, result2), target, source = sample.sample_with_input_file(batch_loader,
                                generator, args, sample_file, True)

                sampled_file_dst = 'logs/intermediate/sampled_out_{}k_{}{}.txt'.format(
                                            iteration//1000, sample_file, args.model_name)
                target_file_dst = 'logs/intermediate/target_out_{}k_{}{}.txt'.format(
                                            iteration//1000, sample_file, args.model_name)
                source_file_dst = 'logs/intermediate/source_out_{}k_{}{}.txt'.format(
                                            iteration//1000, sample_file, args.model_name)
                sampled2_file_dst = 'logs/intermediate/sampled2_out_{}k_{}{}.txt'.format(
                                            iteration//1000, sample_file, args.model_name)
                                            # sampled2 = no reference
                np.save(sampled_file_dst, np.array(result))
                np.save(sampled2_file_dst, np.array(result))
                np.save(target_file_dst, np.array(target))
                np.save(source_file_dst, np.array(source))


                print('------------------------------')
                print('results saved to: ')
                print(sampled_file_dst)
                print(sampled2_file_dst)
                print(target_file_dst)
                print(source_file_dst)

# End
