import argparse
import os

import pandas as pd
import numpy as np
import torch as t
from torch.optim import Adam
import pickle5 as pickle
import json
import random

from sample import sample_with_input, sample_with_beam
from utils.batch_loader import BatchLoader, clean_str
from model.paraphraser import Paraphraser
from model.generator import Generator
from synonym_paraphraser import SynonymParaphraser

def main():
    parser = argparse.ArgumentParser(description='Paraphraser')
    parser.add_argument('--use-cuda', type=bool, default=False, metavar='CUDA', help='use cuda (default: False)')
    parser.add_argument('--seq-len', default=30, metavar='SL', help='max length of sequence (default: 30)')
    parser.add_argument('--ml', type=bool, default=True, metavar='ML', help='sample by maximum likelihood')

    args = parser.parse_args()

    # Read data
    if not os.path.exists('datasets/human_test.csv'):
        source_file = 'datasets/test.csv'
        source_data = pd.read_csv(source_file)[['question1', 'question2']]
        sentence_categories = [[] for _ in range(5)]
        for i in range(len(source_data)):

            sent = clean_str(source_data['question1'][i])
            sent_len = len(sent.split())
            if sent_len < 6:
                j = 0
            elif sent_len < 11:
                j = 1
            elif sent_len < 16:
                j = 2
            elif sent_len < 21:
                j = 3
            else:
                j = 4
            sentence_categories[j].append([source_data['question1'][i], source_data['question2'][i]])

        sample_data = []
        for category in sentence_categories:
            sample_data += random.sample(category, 20)
        source_data = pd.DataFrame(sample_data, columns=['question1', 'question2'])
        source_data.to_csv('datasets/human_test.csv')
    else:
        source_data = pd.read_csv('datasets/human_test_1.csv')[['question1', 'question2']]


    # Sample from Guptas original model
    batch_loader = BatchLoader()
    from model.parameters import Parameters
    parameters = Parameters(batch_loader.max_seq_len, batch_loader.vocab_size)
    paraphraser = Paraphraser(parameters)
    paraphraser.load_state_dict(t.load('saved_models/trained_paraphraser_ori_32', map_location=t.device('cpu')))

    samples_ori, target, source_ori = sample_with_input(batch_loader, paraphraser, args,
                                decoder_only=True,
                                file_name='datasets/human_test.csv')

    ref_items = generate_items(source_ori, target, 'ref')
    ori_items = generate_items(source_ori, samples_ori[0], 'ori')

    # Sample from Guptas model with two-path-loss
    batch_loader = BatchLoader()
    parameters = Parameters(batch_loader.max_seq_len, batch_loader.vocab_size, use_two_path_loss=True)
    paraphraser = Paraphraser(parameters)
    paraphraser.load_state_dict(t.load('saved_models/trained_paraphraser_tpl_16_32', map_location=t.device('cpu')))

    samples_tpl, target, source_tpl = sample_with_input(batch_loader, paraphraser, args,
                                decoder_only=False,
                                file_name='datasets/human_test.csv')
    tpl_items = generate_items(source_tpl, samples_tpl[0], 'tpl')

    # Sample from GAN model
    batch_loader = BatchLoader()
    from model.parametersGAN import Parameters
    parameters = Parameters(batch_loader.max_seq_len, batch_loader.vocab_size)
    paraphraser = Generator(parameters)
    paraphraser.load_state_dict(t.load('saved_models/trained_generator_gan_140k', map_location=t.device('cpu')))
    samples_gan, target, source_gan = sample_with_input(batch_loader, paraphraser, args,
                                decoder_only=False,
                                file_name='datasets/human_test.csv')
    gan_items = generate_items(source_gan, samples_gan[0], 'gan')

    # Sample from synonym model
    paraphraser = SynonymParaphraser()
    samples_synonym = paraphraser.generate_paraphrases('datasets/human_test.csv')
    base_items = generate_items(source_data['question1'], samples_synonym, 'base')

    all_items = ref_items + ori_items + tpl_items + gan_items + base_items

    eval_results = {'name' : 'Paraphrase Survey Full Ordered', 'items' : all_items}
    res = json.dumps(eval_results, ensure_ascii=False)
    with open('datasets/human_test_ordered.json', 'w') as f:
        f.write(res)

    random.shuffle(all_items)

    eval_results = {'name' : 'Paraphrase Survey Full Shuffled', 'items' : all_items}
    res = json.dumps(eval_results, ensure_ascii=False)
    with open('datasets/human_test_shuffled.json', 'w') as f:
        f.write(res)

    for i in range(10):
        eval_results = {'name' : f'Paraphrase Survey Part {i+1}/{10}', 'items' : all_items[i*50:((i+1)*50)-1]}
        res = json.dumps(eval_results, ensure_ascii=False)
        with open(f'datasets/human_test_p_{i}_{10}.json', 'w') as f:
            f.write(res)

def generate_items(original, paraphrase, model):
    items = []
    for i in range(len(original)):

        questions = 'Fråga 1: ' + original[i] + '?<br>Fråga 2: ' + paraphrase[i] + '?'
        item = {
            'question' : questions,
            'required' : True,
            'extra' : {'model' : model},
            'order': -1,
            'answer_sets' : [
                    {
                        "type": "radio",
                        "name": "Fråga 1 är grammatiskt korrekt: ",
                        "choices": [ "0", "1", "2", "3"]
                    },
                    {
                        "type": "radio",
                        "name": "Fråga 2 är grammatiskt korrekt: ",
                        "choices": [ "0", "1", "2", "3"]
                    },
                    {
                        "type": "radio",
                        "name": "Fråga 2 är betyder samma sak som Fråga 1: ",
                        "choices": [ "0", "1", "2", "3"]
                    }]
        }
        items.append(item)
    return items





if __name__ == '__main__':
    main()
