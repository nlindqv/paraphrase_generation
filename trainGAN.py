import argparse
import os

import numpy as np
import torch as t
from torch.optim import Adam

import sample
from utils.batch_loader import BatchLoader
from model.parameters import Parameters
from model.generator import Generator
from model.discriminator import Discriminator

EPOCHS = 200
use_trained = False
converged = False
lr = 0.0001
warmup_step = 10000

if __name__ == "__main__":

    batch_loader = BatchLoader()
    parameters = Parameters(batch_loader.max_seq_len,
                            batch_loader.vocab_size)

    generator = Generator(parameters)
    discriminator = Discriminator()

    if use_trained:
        generator.load_state_dict(t.load('saved_models/trained_generator_' + args.model_name))
        discriminator.load_state_dict(t.load('saved_models/trained_discriminator_' + args.model_name))

    g_optimizer = Adam(generator.learnable_parameters(), lr)
    d_optimizer = Adam(discriminator.learnable_parameters(), lr)

    while not converged:
