import argparse
import os

import numpy as np
import torch as t
from torch.optim import Adam

from sample import sample_with_input, sample_with_beam
from utils.batch_loader import BatchLoader
from model.paraphraser import Paraphraser
from model.generator import Generator

def main():
    parser = argparse.ArgumentParser(description='Paraphraser')
    parser.add_argument('--model-name', default='', metavar='MN', help='name of model to save (default: "")')
    parser.add_argument('--iteration', default='', metavar='I', help='name of model to save (default: "")')
    parser.add_argument('--sample', default=False, metavar='S', help='name of model to save (default: "")')
    parser.add_argument('--print', default=False, metavar='P', help='name of model to save (default: "")')
    parser.add_argument('--num-samples', type=int, default=1, metavar='NS', help='name of model to save (default: "")')
    parser.add_argument('--beam', type=bool, default=False, metavar='B', help='name of model to save (default: "")')
    parser.add_argument('--use-cuda', type=bool, default=False, metavar='CUDA', help='use cuda (default: False)')
    parser.add_argument('--seq-len', default=30, metavar='SL', help='max length of sequence (default: 30)')
    parser.add_argument('--ml', type=bool, default=False, metavar='ML', help='sample by maximum likelihood')


    args = parser.parse_args()

    if args.sample:
        sample(args)

    if args.print:
        print_samples(args)


def sample(args):
    # Create locations to store samples
    if not os.path.isdir('logs/'+ args.model_name + '/samples'):
        os.mkdir('logs/'+ args.model_name + '/samples')

    batch_loader = BatchLoader()
    # Load model...
    if 'ori' in args.model_name.lower() and not 'gan' in args.model_name.lower() or 'tpl' in args.model_name.lower():
        from model.parameters import Parameters
        parameters = Parameters(batch_loader.max_seq_len, batch_loader.vocab_size, use_two_path_loss=('tpl' in args.model_name.lower()))
        paraphraser = Paraphraser(parameters)
        if args.use_cuda:
            paraphraser.load_state_dict(t.load('saved_models/trained_paraphraser_' + args.model_name))
        else:
            paraphraser.load_state_dict(t.load('saved_models/trained_paraphraser_' + args.model_name, map_location=t.device('cpu')))
    elif 'gan' in args.model_name.lower():
        from model.parametersGAN import Parameters
        parameters = Parameters(batch_loader.max_seq_len, batch_loader.vocab_size)
        paraphraser = Generator(parameters)
        if args.use_cuda:
            paraphraser.load_state_dict(t.load('saved_models/trained_generator_' + args.model_name))
        else:
            paraphraser.load_state_dict(t.load('saved_models/trained_generator_' + args.model_name, map_location=t.device('cpu')))
    if args.beam:
        samples, target, source = sample_with_beam(batch_loader, paraphraser, args,
                                    decoder_only=('ori' in args.model_name.lower()),
                                    beam_size=(args.num_samples if args.num_samples != 1 else 5))
        for i in range(args.num_samples):
            np.savetxt(f'logs/{args.model_name}/samples/sampled_beam_{i}.txt', np.array(samples[i]), delimiter='\n', fmt='%s')
        np.savetxt(f'logs/{args.model_name}/samples/target_beam.txt', np.array(target), delimiter='\n', fmt='%s')
        np.savetxt(f'logs/{args.model_name}/samples/source_beam.txt', np.array(source), delimiter='\n', fmt='%s')
    else:
        samples, target, source = sample_with_input(batch_loader, paraphraser, args,
                                    decoder_only=('ori' in args.model_name.lower() and not 'gan' in args.model_name.lower()),
                                    num_samples=args.num_samples,
                                    ml=args.ml)
        for i in range(args.num_samples):
            np.savetxt(f'logs/{args.model_name}/samples/sampled' + ('_ml' if args.ml else '_s') + f'_{i}.txt', np.array(samples[i]), delimiter='\n', fmt='%s')
        np.savetxt(f'logs/{args.model_name}/samples/target' + ('_ml' if args.ml else '_s') + '.txt', np.array(target), delimiter='\n', fmt='%s')
        np.savetxt(f'logs/{args.model_name}/samples/source' + ('_ml' if args.ml else '_s') + '.txt', np.array(source), delimiter='\n', fmt='%s')

def print_samples(args):
    input_file = 'quora_test'

    if args.iteration == '':
        sampled_file_dst = []
        for i in range(args.num_samples):
            sampled_file_dst.append(f'logs/{args.model_name}/samples/sampled_{i}.txt')
        target_file_dst = f'logs/{args.model_name}/samples/target.txt'
        source_file_dst = f'logs/{args.model_name}/samples/source.txt'

        sampled = [list(np.loadtxt(sampled_file_dst[i], dtype='U', delimiter='\n')) for i in range(args.num_samples)]
    else:
        sampled_file_dst = f'logs/{args.model_name}/intermediate/sampledML_{args.iteration}k.txt'
        target_file_dst = f'logs/{args.model_name}/intermediate/targetML_{args.iteration}k.txt'
        source_file_dst = f'logs/{args.model_name}/intermediate/sourceML_{args.iteration}k.txt'

        sampled = [list(np.loadtxt(sampled_file_dst, dtype='U', delimiter='\n'))]
    target = list(np.loadtxt(target_file_dst, dtype='U', delimiter='\n'))
    source = list(np.loadtxt(source_file_dst, dtype='U', delimiter='\n'))


    for i in range(len(source)):
        print(f'Sentence pair {i}:')
        print('source : ', source[i])
        print('target : ', target[i])
        for j in range(args.num_samples):
            print('sampled : ', sampled[j][i])
        print('\n')

if __name__ == "__main__":
    main()
