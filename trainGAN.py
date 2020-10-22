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

iterations = 200000
use_trained = False
converged = False
lr = 0.0001
warmup_step = 10000
batch_size = 32
use_cuda = False
lambdas = [0.5, 0.5, 0.01]
dropout = 0.3

if __name__ == "__main__":

    batch_loader = BatchLoader()
    parameters = Parameters(batch_loader.max_seq_len,
                            batch_loader.vocab_size)

    print(f'Initiate generator...')
    generator = Generator(parameters)
    print(f'Initiate discriminator...')
    discriminator = Discriminator(parameters)

    if use_trained:
        generator.load_state_dict(t.load('saved_models/trained_generator_' + args.model_name))
        discriminator.load_state_dict(t.load('saved_models/trained_discriminator_' + args.model_name))

    print(f'Initiate optimizers...')
    g_optim = Adam(generator.learnable_parameters(), lr)
    d_optim = Adam(discriminator.learnable_parameters(), lr)

    print(f'Initiate rollout...')
    rollout = Rollout(generator, discriminator, 0.8)

    # g_criterion = nn.CrossEntropyLoss()
    # d_criterion = nn.BCEWithLogitsLoss()

    print(f'Start adversarial training...')
    for iter in range(iterations):
        # Warmup training
        if iter <= warmup_step:
            lambda3 = (iter / warmup_step) * lambdas[2]
            lambda2 = (iter / warmup_step) * lambdas[1]
            lambda1 = 1 - lambda2

        # Sample a batch of {s_o, s_p} from dataset
        input = batch_loader.next_batch(batch_size, 'train')
        input = [var.cuda() if use_cuda else var for var in input]

        [encoder_input_source,
         encoder_input_target,
         decoder_input_source,
         decoder_input_target, target] = input

        # print(f'Encoder shape: {encoder_input_source.size()}')
        # for i in range(encoder_input_source.size(0)):
        #     print(f'Encoder[{i}] shape: {encoder_input_source[i].size()}')
        #     print(f'{sen[0][i]}')
        #     print(f'Encoder[{i}]: {encoder_input_source[i]}')

        # Train Generator
        logits, _, kld = generator(dropout,
                (encoder_input_source, encoder_input_target),
                (decoder_input_source, decoder_input_target),
                z=None, use_cuda=use_cuda)
        # print(f'Logits shape: {logits.shape}')

        logits = logits.view(2, -1, parameters.vocab_size)
        target = target.view(-1)
        ce_1 = F.cross_entropy(logits[0], target)
        ce_2 = F.cross_entropy(logits[1], target)

        # print(f'Logits1 shape: {logits[0].shape}')

        # Sample a sequence to feed into discriminator
        # prediction = [batch_size, seq_len, vocab_size]
        prediction = F.softmax(logits[0].view(batch_size, -1, parameters.vocab_size), dim=-1).data.cpu().numpy()
        # print(f'Prediction shape: {prediction.shape}')
        # print(f'Prediction shape 0: {prediction.shape[0]}')
        # prediction = t.Tensor(prediction).view(batch_size, -1, parameters.vocab_size)
        samples = []
        for i in range(prediction.shape[0]):
            samples.append([batch_loader.sample_word_from_distribution(d)
                for d in prediction[i]])
        # samples = [batch_size, seq_len]
        # gen_samples = [batch_size, max_seq_len, 300]
        # print(f'Samples shape: {np.shape(samples)}')

        gen_samples = batch_loader.embed_batch(samples)
        gen_samples = t.Tensor(gen_samples)
        # print(f'Gen_samples shape: {np.shape(gen_samples)}')

        rewards = rollout.reward(gen_samples, 2, input, use_cuda, batch_loader)
        rewards = Variable(t.tensor(rewards))
        neg_lik = F.cross_entropy(logits[0], target, size_average=False, reduce=False)

        print(f'Reward shape: {rewards.size()}, neg_lik shape: {neg_lik.size()}')
        loss_dg = t.mean(neg_lik * rewards.flatten())

        g_loss = lambda1 * ce_1 \
                + lambda2 * ce_2 \
                + lambda3 * loss_dg \
                + lambda1 * kld

        print(f'Update generator parameters... (loss: {g_loss})')
        g_optim.zero_grad()
        g_loss.backward()
        g_optim.step()

        # Train discriminator with real and fake data
        data = t.cat([encoder_input_target, gen_samples], dim=0)

        labels = t.zeros(2*batch_size)
        labels[:batch_size] = 1
        # labels[batch_size:, 0] = 1
        # print(f'Labels: {labels}')

        d_logits = discriminator(data)
        d_loss = F.binary_cross_entropy_with_logits(d_logits, labels)

        print(f'Update discriminator parameters... (loss: {d_loss})')
        # Update discriminator
        d_optim.zero_grad()
        d_loss.backward()
        d_optim.step()













# End
