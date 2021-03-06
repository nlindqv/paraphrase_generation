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
from torch.cuda import amp

import sample
from utils.batch_loader import BatchLoader
from utils.rollout import Rollout
from model.parametersGAN import Parameters
from model.generator import Generator
from model.discriminator import Discriminator
import gc

lambdas = [0.5, 0.5, 0.01]
rollout_num = 8

def trainer(generator, g_optim, discriminator, d_optim, rollout, batch_loader, scaler):
    def train(i, batch_size, use_cuda, dropout):
        input = batch_loader.next_batch(batch_size, 'train')
        input = [var.cuda() if use_cuda else var for var in input]

        [encoder_input_source,
         encoder_input_target,
         decoder_input_source,
         decoder_input_target, target] = input

        target = target.view(-1)

        g_optim.zero_grad()

        with amp.autocast():
            (logits, logits2), _, kld = generator(dropout,
                    (encoder_input_source, encoder_input_target),
                    (decoder_input_source, decoder_input_target),
                    z=None, use_cuda=use_cuda)

            logits = logits.view(-1, generator.params.vocab_size)
            logits2 = logits2.view(-1, generator.params.vocab_size)
            ce_1 = F.cross_entropy(logits, target)
            ce_2 = F.cross_entropy(logits2, target)

            # Generate fake data
            prediction = F.softmax(logits2, dim=-1)
            samples = prediction.multinomial(1).view(batch_size, -1)
            gen_samples = batch_loader.embed_batch_from_index(samples)
            if use_cuda:
                gen_samples = gen_samples.cuda()

            rewards = rollout.reward(gen_samples, [encoder_input_source, encoder_input_target], decoder_input_source, use_cuda, batch_loader)
            rewards = Variable(t.tensor(rewards))
            if use_cuda:
                rewards = rewards.cuda()
            neg_lik = F.cross_entropy(logits2, target, reduction='none')

            dg_loss = t.mean(neg_lik * rewards.flatten())
            g_loss = lambda1 * ce_1 + lambda1 * kld + lambda2 * ce_2 + lambda3 * dg_loss


        scaler.scale(g_loss).backward()
        scaler.unscale_(g_optim)
        t.nn.utils.clip_grad_norm_(generator.learnable_parameters(), 10)
        scaler.step(g_optim)
        scaler.update()

        # Train discriminator with real and fake data
        data = t.cat([encoder_input_target, gen_samples], dim=0)

        labels = t.zeros(2*batch_size)
        labels[:batch_size] = 1

        if use_cuda:
            labels = labels.cuda()
            data = data.cuda()


        d_optim.zero_grad()
        with amp.autocast():
            d_logits = discriminator(data)
            d_loss = F.binary_cross_entropy_with_logits(d_logits, labels)

        scaler.scale(d_loss).backward()
        scaler.unscale_(d_optim)
        t.nn.utils.clip_grad_norm_(discriminator.learnable_parameters(), 5)
        scaler.step(d_optim)
        scaler.update()

        return (ce_1, ce_2, dg_loss, d_loss), kld

    return train

@t.no_grad()
def validater(generator, discriminator, rollout, batch_loader):
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

    @t.no_grad()
    def validate(batch_size, use_cuda, need_samples=False):
        generator.eval()
        discriminator.eval()

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

        target = target.view(-1)
        logits = logits.view(-1, generator.params.vocab_size)
        logits2 = logits2.view(-1, generator.params.vocab_size)
        ce_1 = F.cross_entropy(logits, target)
        ce_2 = F.cross_entropy(logits2, target)
        if need_samples:
            [s1, s2] = sentences
            sampled = []
        else:
            s1, s2 = (None, None)
            sampled = None

        prediction = F.softmax(logits, dim=-1)
        samples = prediction.multinomial(1).view(batch_size, -1)
        if need_samples:
            for i in range(samples.size(0)):
                sampled += [' '.join(batch_loader.get_word_by_idx(idx) for idx in samples[i])]
        gen_samples = batch_loader.embed_batch_from_index(samples)

        # gen_samples, logits_gen = generator.sample_seq(batch_loader, input, args.use_cuda)


        if use_cuda:
            gen_samples = gen_samples.cuda()

        rewards = rollout.reward(gen_samples, [encoder_input_source, encoder_input_target], decoder_input_source, use_cuda, batch_loader)
        rewards = Variable(t.tensor(rewards))

        if use_cuda:
            rewards = rewards.cuda()
        neg_lik = F.cross_entropy(logits, target, reduction='none')
        dg_loss = t.mean(neg_lik * rewards.flatten())


        # Train discriminator with real and fake data
        data = t.cat([encoder_input_target, gen_samples], dim=0)

        labels = t.zeros(2*batch_size)
        labels[:batch_size] = 1

        if use_cuda:
            labels = labels.cuda()
            data = data.cuda()

        d_logits = discriminator(data)
        d_loss = F.binary_cross_entropy_with_logits(d_logits, labels)

        generator.train()
        discriminator.train()


        return (ce_1, ce_2, kld, dg_loss, d_loss), (sampled, s1, s2)

    return validate

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Paraphraser')
    parser.add_argument('--num-iterations', type=int, default=250000, help='num iterations (default: 60000)')
    parser.add_argument('--batch-size', type=int, default=32, help='batch size (default: 32)')
    parser.add_argument('--use-cuda', type=bool, default=True, help='use cuda (default: True)')
    parser.add_argument('--learning-rate', type=float, default=0.0001, help='learning rate (default: 0.0001)')
    parser.add_argument('--dropout', type=float, default=0.3, help='dropout (default: 0.3)')
    parser.add_argument('--use-trained', type=bool, default=False, help='load pretrained model (default: False)')
    parser.add_argument('--model-name', default='', help='name of model to save (default: "")')
    parser.add_argument('--warmup-step', default=10000, type=float, help='L2 regularization penalty (default: 0.0)')
    parser.add_argument('--interm-sampling', default=True, type=bool, help='if sample while training (default: False)')
    args = parser.parse_args()

    if args.use_cuda and not t.cuda.is_available():
        print('Found no GPU, args.use_cuda = False ')
        args.use_cuda = False

    batch_loader = BatchLoader()
    parameters = Parameters(batch_loader.max_seq_len,
                            batch_loader.vocab_size)

    generator = Generator(parameters)
    discriminator = Discriminator(parameters)

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

    generator = Generator(parameters)
    discriminator = Discriminator(parameters)

    print(f'Number of parameters in generator: {sum(p.numel() for p in generator.learnable_parameters())}')
    print(f'Number of parameters in discriminator: {sum(p.numel() for p in discriminator.learnable_parameters())}')

    # Create locations to store logs
    if not os.path.isdir('logs/'+ args.model_name):
        os.mkdir('logs/'+ args.model_name)
    if args.interm_sampling and not os.path.isdir('logs/'+ args.model_name + '/intermediate'):
        os.mkdir('logs/'+ args.model_name + '/intermediate')


    if args.use_trained:
        generator.load_state_dict(t.load('saved_models/trained_generator_' + args.model_name))
        discriminator.load_state_dict(t.load('saved_models/trained_discriminator_' + args.model_name))
        ce_result_valid = list(np.load('logs/{}/ce_result_valid.npy'.format(args.model_name)))
        ce_result_train = list(np.load('logs/{}/ce_result_train.npy'.format(args.model_name)))
        ce2_result_train = list(np.load('logs/{}/ce2_result_train.npy'.format(args.model_name)))
        ce2_result_valid = list(np.load('logs/{}/ce2_result_valid.npy'.format(args.model_name)))
        kld_result_valid = list(np.load('logs/{}/kld_result_valid.npy'.format(args.model_name)))
        kld_result_train = list(np.load('logs/{}/kld_result_train.npy'.format(args.model_name)))
        dg_result_train = list(np.load('logs/{}/dg_result_train.npy'.format(args.model_name)))
        dg_result_valid = list(np.load('logs/{}/dg_result_valid.npy'.format(args.model_name)))
        d_result_train = list(np.load('logs/{}/d_result_train.npy'.format(args.model_name)))
        d_result_valid = list(np.load('logs/{}/d_result_valid.npy'.format(args.model_name)))

    if args.use_cuda:
        generator = generator.cuda()
        discriminator = discriminator.cuda()

    g_optim = Adam(generator.learnable_parameters(), args.learning_rate)
    d_optim = Adam(discriminator.learnable_parameters(), args.learning_rate)
    # [generator, discriminator], [g_optim, d_optim] = amp.initialize([generator, discriminator], [g_optim, d_optim], opt_level="O1", num_losses=2)

    rollout = Rollout(generator, discriminator, 0.8, rollout_num)

    # discriminator, d_optim = amp.initialize(discriminator, d_optim, opt_level="O1")
    scaler = amp.GradScaler()

    train_step = trainer(generator, g_optim, discriminator, d_optim, rollout, batch_loader, scaler)
    validate = validater(generator, discriminator, rollout, batch_loader)


#    converge_criterion, converge_count = 1000, 0
#    best_total_loss = np.inf

    start = time.time_ns()

    for iteration in range(args.num_iterations):
        if iteration <= args.warmup_step:
            lambda3 = iteration / (1. * args.warmup_step) * lambdas[2]
            lambda2 = iteration / (1. * args.warmup_step) * lambdas[1]
            lambda1 = 1 - lambda2


        (ce_1, ce_2, dg_loss, d_loss), kld = train_step(iteration, args.batch_size, args.use_cuda, args.dropout)
        # t.cuda.empty_cache()

        # Store losses
        ce_cur_train += [ce_1.data.cpu().numpy()]
        ce2_cur_train += [ce_2.data.cpu().numpy()]
        kld_cur_train += [kld.data.cpu().numpy()]
        dg_cur_train += [dg_loss.data.cpu().numpy()]
        d_cur_train += [d_loss.data.cpu().numpy()]

        # validation
        if iteration % 500 == 0     :
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
                (c1, c2, kl, dg, d), _ = validate(args.batch_size, args.use_cuda)
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

            total_loss = ce_1 * lambda1 + ce_2 *lambda2 + kld * lambda1 + dg_loss * lambda3
 #           if iteration > 10000:
 #               if np.isinf(best_total_loss):
 #                   best_total_loss = total_loss
 #               else:
 #                   if total_loss >= best_total_loss:
 #                       converge_count += 1
 #                   else:
 #                       best_total_loss = total_loss
 #                       converge_count = 0

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

            _, (sampled, s1, s2) = validate(2, args.use_cuda, need_samples=True)

            for i in range(len(sampled)):
                result = generator.sample_with_pair(batch_loader, 20, args.use_cuda, s1[i], s2[i])

                print('source: ' + s1[i])
                print('target: ' + s2[i])
                print('sampled: ' + result)
                print('...........................')

        # save model
        if (iteration % 10000 == 0 and iteration != 0) or iteration == (args.num_iterations - 1):
            t.save(generator.state_dict(), 'saved_models/trained_generator_' + args.model_name + '_' + iteration//1000)
            t.save(discriminator.state_dict(), 'saved_models/trained_discrminator_' + args.model_name + '_' +  iteration//1000)
            np.save('logs/{}/ce_result_valid.npy'.format(args.model_name), np.array(ce_result_valid))
            np.save('logs/{}/ce_result_train.npy'.format(args.model_name), np.array(ce_result_train))
            np.save('logs/{}/kld_result_valid'.format(args.model_name), np.array(kld_result_valid))
            np.save('logs/{}/kld_result_train'.format(args.model_name), np.array(kld_result_train))
            np.save('logs/{}/ce2_result_valid.npy'.format(args.model_name), np.array(ce2_result_valid))
            np.save('logs/{}/ce2_result_train.npy'.format(args.model_name), np.array(ce2_result_train))
            np.save('logs/{}/dg_result_valid.npy'.format(args.model_name), np.array(dg_result_valid))
            np.save('logs/{}/dg_result_train.npy'.format(args.model_name), np.array(dg_result_train))
            np.save('logs/{}/d_result_valid.npy'.format(args.model_name), np.array(d_result_valid))
            np.save('logs/{}/d_result_train.npy'.format(args.model_name), np.array(d_result_train))

        #interm sampling
        if (iteration % 10000 == 0 and iteration != 0) or iteration == (args.num_iterations - 1):
            if args.interm_sampling:
                args.seq_len = 30

                result, target, source = sample.sample_with_input_file(batch_loader, generator, args)

                sampled_file_dst = 'logs/{}/intermediate/sampled_{}k.txt'.format(args.model_name, iteration//1000)
                target_file_dst = 'logs/{}/intermediate/target_{}k.txt'.format(args.model_name, iteration//1000)
                source_file_dst = 'logs/{}/intermediate/source_{}k.txt'.format(args.model_name, iteration//1000)

                np.savetxt(sampled_file_dst, np.array(result), delimiter='\n', fmt='%s')
                np.savetxt(target_file_dst, np.array(target), delimiter='\n', fmt='%s')
                np.savetxt(source_file_dst, np.array(source), delimiter='\n', fmt='%s')

                print('------------------------------')
                print('results saved to: ')
                print(sampled_file_dst)
                print(target_file_dst)
                print(source_file_dst)

# End
