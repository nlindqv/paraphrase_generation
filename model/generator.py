import math
import time
import torch as t
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from .decoder import Decoder
from .encoder import Encoder
from .highway import Highway

class Generator(nn.Module):
    def __init__(self, params):
        super(Generator, self).__init__()
        self.params = params
        self.highway = Highway(self.params.word_embed_size, 2, F.relu)
        self.encoder = Encoder(self.params, self.highway)
        self.decoder = Decoder(self.params, self.highway)

    def forward(self, drop_prob, encoder_input=None, decoder_input=None,
        z=None, initial_state=None, use_cuda=True):
        """
        :param encoder_word_input: An list of 2 tensors with shape of [batch_size, seq_len] of Long type
        :param decoder_word_input: An An list of 2 tensors with shape of [batch_size, max_seq_len + 1] of Long type
        :param initial_state: initial state of decoder rnn in order to perform sampling

        :param drop_prob: probability of an element of decoder input to be zeroed in sense of dropout

        :param z: context if sampling is performing

        :return: unnormalized logits of sentence words distribution probabilities
                    with shape of [batch_size, seq_len, word_vocab_size]
                 final rnn state with shape of [num_layers, batch_size, decoder_rnn_size]
        """

        if z is None:
            ''' Get context from encoder and sample z ~ N(mu, std)
            '''
            [batch_size, _, _] = encoder_input[0].size()

            mu, logvar = self.encoder(encoder_input[0], encoder_input[1])
            std = t.exp(0.5 * logvar)

            z1 = Variable(t.randn([batch_size, self.params.latent_variable_size]))
            if use_cuda:
                z1 = z1.cuda()
            z1 = z1 * std + mu

            kld = (-0.5 * t.sum(logvar - t.pow(mu, 2) - t.exp(logvar) + 1, 1)).mean().squeeze()

            mu, logvar = self.encoder(encoder_input[0], None)
            std = t.exp(0.5 * logvar)

            z2 = Variable(t.randn([batch_size, self.params.latent_variable_size]))
            if use_cuda:
                z2 = z2.cuda()
            z2 = z2 * std + mu
        else:
            kld = None

        out1, final_state = self.decoder(decoder_input[0], decoder_input[1],
                                        z1, drop_prob, initial_state)

        out2, _ = self.decoder(decoder_input[0], decoder_input[1],
                                        z2, drop_prob, initial_state)

        return (out1, out2), final_state, kld

    def learnable_parameters(self):
        return [p for p in self.parameters() if p.requires_grad]

    def sample_with_input(self, batch_loader, seq_len, use_cuda, use_mean, input, input_only=False):
        [encoder_input_source, encoder_input_target, decoder_input_source, _, _] = input

        encoder_input = [encoder_input_source, encoder_input_target]

        # encode
        [batch_size, _, _] = encoder_input[0].size()

        if input_only:
            mu, logvar = self.encoder(encoder_input[0], None)
        else:
            mu, logvar = self.encoder(encoder_input[0], encoder_input[1])
        std = t.exp(0.5 * logvar)

        if use_mean:
            z = mu
        else:
            z = Variable(t.randn([batch_size, self.params.latent_variable_size]))
            if use_cuda:
                z = z.cuda()
            z = z * std + mu

        initial_state = self.decoder.build_initial_state(decoder_input_source)

        decoder_input = batch_loader.get_raw_input_from_sentences([batch_loader.go_label])

        result = ''
        for i in range(seq_len):
            if use_cuda:
                decoder_input = decoder_input.cuda()

            logits, initial_state = self.decoder(None, decoder_input, z, 0.0, initial_state)
            logits = logits.view(-1, self.params.vocab_size)
            # prediction = F.softmax(logits)
            prediction = F.softmax(logits, dim=-1)
            word = batch_loader.likely_word_from_distribution(prediction.data.cpu().numpy()[-1])
            # word = batch_loader.sample_word_from_distribution(prediction.data.cpu().numpy()[-1])
            if word == batch_loader.end_label:
                break
            result += ' ' + word

            decoder_input = batch_loader.get_raw_input_from_sentences([word])

        return result

    def sample(self, x, seq_len, z, initial_state, use_cuda, batch_loader):

        given_len = x.size(1)
        decoder_input = x[:, 0, :].unsqueeze(1) #given[0]
        result = []

        # Dynamic programming approach
        result = list(x[:, :-1, :].chunk(given_len, 1))
        decoder_input = x[:, -1, :].unsqueeze(1)

        for i in range(given_len, seq_len):
            if use_cuda:
                decoder_input = decoder_input.cuda()

            logits, initial_state = self.decoder(None, decoder_input, z, 0.0, initial_state)
            logits = logits.view(-1, self.params.vocab_size)

            # Save next inital state for next part of rollout...
            if i == given_len:
                next_initial_state = initial_state

            prediction = F.softmax(logits, dim=-1)
            words = batch_loader.likely_words_from_distribution(prediction.data.cpu().numpy())

            all_end_labels = True
            for word in words:
                if word != batch_loader.end_label:
                    all_end_labels = False
                    break
            if all_end_labels:
                print(f'Words last generated: {words}')
                break

            decoder_input = batch_loader.get_raw_input_from_sentences(words)

            if use_cuda:
                decoder_input = decoder_input.cuda()
            result.append(decoder_input)

        result = t.cat(result, dim=1)

        return result, next_initial_state



    def sample_with_pair(self, batch_loader, seq_len, use_cuda, source_sent, target_sent, input_only=False):
        input = batch_loader.input_from_sentences([[source_sent], [target_sent]])
        input = [var.cuda() if use_cuda else var for var in input]
        return self.sample_with_input(batch_loader, seq_len, use_cuda, False, input, input_only)

    def sample_with_seed(self, batch_loader, seq_len, use_cuda, seed):
        pass

    def sample_with_phrase(self, batch_loader, seq_len, use_cuda, source_sent):
        pass
