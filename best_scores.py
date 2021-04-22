import argparse, re
import os, os.path, subprocess
from subprocess import *
from nlgeval import NLGEval

import numpy as np
import torch as t
from torch.optim import Adam

from sample import sample_with_input, sample_with_beam
from utils.batch_loader import BatchLoader
from model.paraphraser import Paraphraser
from model.generator import Generator

TEST_DATA_SIZE = 4000

def create_files(model_name):
    path = f'logs/{model_name}/samples'
    # Create locations to store samples
    if not os.path.isdir(path + '/tmp/'):
        os.mkdir(path + '/tmp/')

    for file_name in os.listdir(path):
        if os.path.isdir(path + '/' + file_name):
            continue
        sentences = list(np.loadtxt(path + '/' + file_name, dtype='U', delimiter='\n'))
        for idx, sentence in enumerate(sentences):
            if not os.path.isdir(path + '/tmp/' + str(idx) + '/'):
                os.mkdir(path + '/tmp/' + str(idx) + '/')

            if file_name.startswith("sampled"):
                sample_nr = re.findall('\d*\.?\d+', file_name)[0]
                np.savetxt(f'{path}/tmp/{idx}/sampled_{sample_nr}', np.array([sentence]), delimiter='\n', fmt='%s')
            elif file_name.startswith("source"):
                np.savetxt(f'{path}/tmp/{idx}/source', np.array([sentence]), delimiter='\n', fmt='%s')

def avg_meteor(model_name, mode, num_samples):
    sampled_file_dst = []
    for i in range(num_samples):
        sampled_file_dst.append(f'logs/{model_name}/samples/sampled_{mode}_{i}.txt')

    target_file_dst = f'logs/{model_name}/samples/target_{mode}.txt'

    scores = []
    for i in range(num_samples):
        args = ['multeval-0.5.1/lib/meteor-1.4/meteor-1.4.jar', sampled_file_dst[i], target_file_dst, '-l', 'se']
        scores.append(jarWrapper(*args))

    avg_score = sum(scores) / len(scores)

    print(f'Model: {model_name}, score: {avg_score}')


def meteor(model_name, mode, num_samples):
    sampled_file_dst = []
    for i in range(num_samples):
        sampled_file_dst.append(f'logs/{model_name}/samples/sampled_{mode}_{i}.txt')

    source_file_dst = f'logs/{model_name}/samples/source_{mode}.txt'

    sampled = [list(np.loadtxt(sampled_file_dst[i], dtype='U', delimiter='\n')) for i in range(num_samples)]
    source = list(np.loadtxt(source_file_dst, dtype='U', delimiter='\n'))

    if not os.path.isdir(f'logs/{model_name}/samples/tmp/'):
        os.mkdir(f'logs/{model_name}/samples/tmp/')
    print(len(source), len(sampled), [len(sampled[i]) for i in range(len(sampled))])

    best_meteor = []
    for i in range(len(source)):
        curr_best = ('', 0.0)
        np.savetxt(f'logs/{model_name}/samples/tmp/source', np.array([source[i]]), delimiter='\n', fmt='%s')
        for j in range(num_samples):
            np.savetxt(f'logs/{model_name}/samples/tmp/sample', np.array([sampled[j][i]]), delimiter='\n', fmt='%s')
            args = ['multeval-0.5.1/lib/meteor-1.4/meteor-1.4.jar', f'logs/{model_name}/samples/tmp/sample', f'logs/{model_name}/samples/tmp/source', '-l', 'se']
            score = jarWrapper(*args)
            if score > curr_best[1]:
                curr_best = (sampled[j][i], score)
            print(f'Sentence pair {i}.{j}:')
            print('source : ', source[i])
            print('sampled : ', sampled[j][i])
            print('\n')
        best_meteor.append(curr_best[0])
    np.savetxt(f'logs/{model_name}/samples/best_meteor_{mode}', np.array(best_meteor), delimiter='\n', fmt='%s')

def jarWrapper(*args):
    process = Popen(['java', '-jar']+list(args), stdout=PIPE, stderr=PIPE)
    ret = []
    while process.poll() is None:
        line = process.stdout.readline()
        if line != '' and line.endswith(b'\n'):
            ret.append(line[:-1])
    stdout, stderr = process.communicate()
    ret += stdout.split(b'\n')
    ret.remove(b'')
    for i in range(len(ret)):
        ret[i] = ret[i].decode()
    ref = ''
    i = -1
    while 'Final score' not in ref:
        i += 1
        ref = ret[i]
    score = float(re.findall('\d*\.?\d+', ref)[0])

    return score

def bleu(model_name, mode, num_samples):
    sampled_file_dst = []
    for i in range(num_samples):
        sampled_file_dst.append(f'logs/{model_name}/samples/sampled_{mode}_{i}.txt')

    source_file_dst = f'logs/{model_name}/samples/source_{mode}.txt'

    sampled = [list(np.loadtxt(sampled_file_dst[i], dtype='U', delimiter='\n')) for i in range(num_samples)]
    source = list(np.loadtxt(source_file_dst, dtype='U', delimiter='\n'))

    nlgeval = NLGEval(metrics_to_omit=['METEOR', 'ROUGE_L', 'CIDEr', 'SkipThoughtCS'])
    best_bleu = []
    print(len(source), len(sampled), [len(sampled[i]) for i in range(len(sampled))])
    for i in range(len(source)):
        curr_best = ('', 0.0)
        for j in range(num_samples):
            score = nlgeval.compute_individual_metrics([source[i]], sampled[j][i])['Bleu_4']
            if score > curr_best[1]:
                curr_best = (sampled[j][i], score)
        if i % 100 == 0:
            print(f'Sentence pair {i}.{j}:')
            print('source : ', source[i])
            print('sampled : ', curr_best[0])
            print('score : ', curr_best[1])
            print('\n')
        best_bleu.append(curr_best[0])
    np.savetxt(f'logs/{model_name}/samples/best_bleu_{mode}', np.array(best_bleu), delimiter='\n', fmt='%s')



def main():
    parser = argparse.ArgumentParser(description='Paraphraser')
    parser.add_argument('--model-name', default='', metavar='MN', help='name of model to save (default: "")')
    parser.add_argument('--metric', default='meteor', metavar='M', help='sample by maximum likelihood')
    parser.add_argument('--mode', default='ml', metavar='MD', help='sample by maximum likelihood')
    parser.add_argument('--avg', type=bool, default=False, metavar='A', help='sample by maximum likelihood')

    args = parser.parse_args()

    if not args.avg:
        if args.metric == 'meteor':
            meteor(args.model_name, args.mode, (10 if args.mode == 'ml' else 5))
        elif args.metric == 'bleu':
            bleu(args.model_name, args.mode, (10 if args.mode == 'ml' else 5))
    else:
        if args.model_name == '':
            models = ['ori_32_50k', 'ori_32_100k', 'ori_32', 'tpl_16_32_50k', 'tpl_16_32_100k', 'tpl_16_32', 'gan_50k', 'gan_100k', 'gan_140k']
            for model in models:
                avg_meteor(model, args.mode, (10 if args.mode == 'ml' else 5))
        else:
            avg_meteor(args.model_name, args.mode, (10 if args.mode == 'ml' else 5))

if __name__ == "__main__":
    main()
