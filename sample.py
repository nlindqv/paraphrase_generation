import argparse
import os
import time
import numpy as np
import torch as t
from torch.optim import Adam

from utils.batch_loader import BatchLoader
from model.parameters import Parameters
from model.parametersGAN import Parameters as ParametersGAN

from model.paraphraser import Paraphraser
from model.generator import Generator

def sample_with_input_file(batch_loader, paraphraser, args):
    result, target, source, i = [], [] , [],  0
    while True:
        next_batch = batch_loader.next_batch_from_file(batch_size=1, return_sentences=True)

        if next_batch is None:
            break

        input, sentences = next_batch
        input = [var.cuda() if args.use_cuda else var for var in input]

        if paraphraser.params.use_two_path_loss:
            result += [paraphraser.sample_with_input(batch_loader,
                                args.seq_len, args.use_cuda, input)]
        else:
            result += [paraphraser.sample_from_normal(batch_loader,
                                args.seq_len, args.use_cuda, input)]

        target += [' '.join(sentences[1][0])]
        source += [' '.join(sentences[0][0])]
        if i % 1000 == 0:
            print(i)
            print('source : ', source[-1])
            print('target : ', target[-1])
            print('sampled : ', result[-1])

        i += 1
    return result, target, source

def sample_with_beam(batch_loader, paraphraser, args, decoder_only, beam_size=5):
    results, target, source, i = [] , [], [] , 0
    while True:
        start = time.time()
        next_batch = batch_loader.next_batch_from_file(batch_size=1,
         file_name='quora_test', return_sentences=True)

        if next_batch is None:
            break

        input, sentences = next_batch
        input = [var.cuda() if args.use_cuda else var for var in input]

        results += [paraphraser.beam_search(batch_loader, args.seq_len, args.use_cuda, input, beam_size, decoder_only)]

        target += [' '.join(sentences[1][0])]
        source += [' '.join(sentences[0][0])]
        if i % 1000 == 0:
            print(i)
            print('source : ', ' '.join(sentences[0][0]))
            print('target : ', ' '.join(sentences[1][0]))
            for j in range(beam_size):
                print('sampled : ', results[-1][j])
        i += 1
        print(f'Iteration {i}/4000, elapsed time: {(time.time()-start):.0f}s')
    return results, target, source


def sample_with_input(batch_loader, paraphraser, args, decoder_only, num_samples=1):
    result, target, source, i = [] , [], [] , 0
    for j in range(num_samples):
        result.append([])
    while True:
        next_batch = batch_loader.next_batch_from_file(batch_size=1,
         file_name='quora_test', return_sentences=True)

        if next_batch is None:
            break

        input, sentences = next_batch
        input = [var.cuda() if args.use_cuda else var for var in input]

        for j in range(num_samples):
            if decoder_only:
                result[j] += [paraphraser.sample_from_normal(batch_loader, args.seq_len, args.use_cuda, input)]
            else:
                result[j] += [paraphraser.sample_with_input(batch_loader, args.seq_len, args.use_cuda, input)]


        target += [' '.join(sentences[1][0])]
        source += [' '.join(sentences[0][0])]
        if i % 1000 == 0:
            print(i)
            print('source : ', ' '.join(sentences[0][0]))
            print('target : ', ' '.join(sentences[1][0]))
            for j in range(num_samples):
                print('sampled : ', result[j][-1])
        i += 1
    return result, target, source

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Paraphraser')
    parser.add_argument('--use-cuda', type=bool, default=False, metavar='CUDA', help='use cuda (default: False)')
    parser.add_argument('--model-name', default='', metavar='MN', help='name of model to save (default: "")')
    parser.add_argument('--seq-len', default=30, metavar='SL', help='max length of sequence (default: 30)')
    parser.add_argument('--model', default='C-VAE', metavar='M', help='Model to use (default: C-VAE)')
    args = parser.parse_args()

    batch_loader = BatchLoader()
    if args.model == 'C-VAE':
        parameters = Parameters(batch_loader.max_seq_len, batch_loader.vocab_size)
        paraphraser = Paraphraser(parameters)
        paraphraser.load_state_dict(t.load('saved_models/trained_paraphraser_' + args.model_name, map_location=t.device('cpu')))
    elif args.model == 'C-VAE*':
        parameters = Parameters(batch_loader.max_seq_len, batch_loader.vocab_size, use_two_path_loss=True)
        paraphraser = Paraphraser(parameters)
        paraphraser.load_state_dict(t.load('saved_models/trained_paraphraser_' + args.model_name, map_location=t.device('cpu')))
    elif args.model == 'GAN':
        parameters = ParametersGAN(batch_loader.max_seq_len, batch_loader.vocab_size)
        paraphraser = Generator(parameters)
        paraphraser.load_state_dict(t.load('saved_models/trained_generator_' + args.model_name, map_location=t.device('cpu')))

    if args.use_cuda:
        paraphraser = paraphraser.cuda()

    result, target, source = sample_with_input(batch_loader, paraphraser, args, decoder_only=(args.model == 'C-VAE'))


    sampled_file_dst = 'logs/sampled_out_{}.txt'.format(args.model_name)
    target_file_dst = 'logs/target_out_{}.txt'.format(args.model_name)
    source_file_dst = 'logs/source_out_{}.txt'.format(args.model_name)

    np.savetxt(sampled_file_dst, np.array(result), delimiter='\n', fmt='%s')
    np.savetxt(target_file_dst, np.array(target), delimiter='\n', fmt='%s')
    np.savetxt(source_file_dst, np.array(source), delimiter='\n', fmt='%s')

    print('------------------------------')
    print('results saved to: ')
    print(sampled_file_dst)
    print(target_file_dst)
    print(source_file_dst)
    print('END')
