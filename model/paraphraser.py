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

class Paraphraser(nn.Module):
    def __init__(self, params):
        super(Paraphraser, self).__init__()
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

            if self.params.use_two_path_loss:
                mu2, logvar2 = self.encoder(encoder_input[0], input_target=None)
                std2 = t.exp(0.5 * logvar2)

                z2 = Variable(t.randn([batch_size, self.params.latent_variable_size]))
                if use_cuda:
                    z2 = z2.cuda()
                z2 = z2 * std2 + mu2

        else:
            kld = None

        out1, final_state = self.decoder(decoder_input[0], decoder_input[1],
                                        z1, drop_prob, initial_state)
        if self.params.use_two_path_loss:
            out2, final_state2 = self.decoder(decoder_input[0], decoder_input[1],
                                            z2, drop_prob, initial_state)
        else:
            out2 = None

        return (out1, out2), final_state, kld

    def learnable_parameters(self):
        return [p for p in self.parameters() if p.requires_grad]

    def trainer(self, optimizer, batch_loader):
        def train(i, batch_size, use_cuda, dropout):
            input = batch_loader.next_batch(batch_size, 'train')
            input = [var.cuda() if use_cuda else var for var in input]

            [encoder_input_source,
             encoder_input_target,
             decoder_input_source,
             decoder_input_target, target] = input

            (logits, logits2), _, kld = self(dropout,
                    (encoder_input_source, encoder_input_target),
                    (decoder_input_source, decoder_input_target),
                    z=None, use_cuda=use_cuda)

            target = target.view(-1)
            cross_entropy, cross_entropy2 = [], []


            logits = logits.view(-1, self.params.vocab_size)
            cross_entropy = F.cross_entropy(logits, target)

            if self.params.use_two_path_loss:
                logits2 = logits2.view(-1, self.params.vocab_size)
                cross_entropy2 = F.cross_entropy(logits2, target)
            else:
                cross_entropy2 = 0

            loss = self.params.ce_weight * cross_entropy \
                 + self.params.ce2_weight * cross_entropy2) \
                 + self.params.get_kld_coef(i) * kld

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            return (cross_entropy, cross_entropy2), kld, self.params.get_kld_coef(i)

        return train

    def validater(self, batch_loader):
        def get_samples(logits, target):
            '''
            logits: [batch, seq_len, vocab_size]
            targets: [batch, seq_len]
            '''

            ## for version > 0.4
            prediction = F.softmax(logits, dim=-1).data.cpu().numpy()


            ## for version < 0.3
            # seq_len = logits.size()[1]
            # prediction = F.softmax(
            #     logits.view(-1, self.params.vocab_size)).view(-1, seq_len, self.params.vocab_size)
            # prediction = prediction.data.cpu().numpy()


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

            (logits, logits2), _, kld = self(0., (encoder_input_source, encoder_input_target),
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

            logits = logits.view(-1, self.params.vocab_size)
            cross_entropy = F.cross_entropy(logits, target)

            if self.params.use_two_path_loss:
                logits2 = logits2.view(-1, self.params.vocab_size)
                cross_entropy2 = F.cross_entropy(logits2, target)
            else:
                cross_entropy2 = None


            return (cross_entropy, cross_entropy2), kld, (sampled, s1, s2)

        return validate

    def sample_with_input(self, batch_loader, seq_len, use_cuda, input, ml=True):
        [encoder_input_source, encoder_input_target, decoder_input_source, _, _] = input

        encoder_input = [encoder_input_source, encoder_input_target]

        # encode
        [batch_size, _, _] = encoder_input[0].size()

        mu, logvar = self.encoder(encoder_input[0], None)

        std = t.exp(0.5 * logvar)


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
            prediction = F.softmax(logits, dim=-1)
            if ml:
                word = batch_loader.likely_word_from_distribution(prediction.data.cpu().numpy()[-1])
            else:
                word = batch_loader.sample_word_from_distribution(prediction.data.cpu().numpy()[-1])
            if word == batch_loader.end_label:
                break
            result += ' ' + word

            decoder_input = batch_loader.get_raw_input_from_sentences([word])

        return result

    def sample_with_pair(self, batch_loader, seq_len, use_cuda, source_sent, target_sent):
        input = batch_loader.input_from_sentences([[source_sent], [target_sent]])
        input = [var.cuda() if use_cuda else var for var in input]
        return self.sample_with_input(batch_loader, seq_len, use_cuda, input)

    """ Should only be used with a batch size of 1 """
    def sample_from_normal(self, batch_loader, seq_len, use_cuda, input, ml=True):
        [_, _, decoder_input_source, _, _] = input
        [batch_size, _, _] = decoder_input_source.size()

        z = Variable(t.randn([batch_size, self.params.latent_variable_size]))
        if use_cuda:
            z = z.cuda()

        initial_state = self.decoder.build_initial_state(decoder_input_source)
        decoder_input = batch_loader.get_raw_input_from_sentences([batch_loader.go_label])

        result = ''
        for i in range(seq_len):
            if use_cuda:
                decoder_input = decoder_input.cuda()

            logits, initial_state = self.decoder(None, decoder_input, z, 0.0, initial_state)
            logits = logits.view(-1, self.params.vocab_size)
            prediction = F.softmax(logits, dim=-1)
            if ml:
                word = batch_loader.likely_word_from_distribution(prediction.data.cpu().numpy()[-1])
            else:
                word = batch_loader.sample_word_from_distribution(prediction.data.cpu().numpy()[-1])
            if word == batch_loader.end_label:
                break
            result += ' ' + word

            decoder_input = batch_loader.get_raw_input_from_sentences([word])

        return result

    def beam_search(self, batch_loader, seq_len, use_cuda, input, k, sample_from_normal):
        [encoder_input_source, _, decoder_input_source, _, _] = input

        # encode
        [batch_size, _, _] = decoder_input_source.size()

        z = Variable(t.randn([batch_size, self.params.latent_variable_size]))
        if use_cuda:
            z = z.cuda()

        if not sample_from_normal:
            mu, logvar = self.encoder(encoder_input_source, None)
            std = t.exp(0.5 * logvar)
            z = z * std + mu

        initial_state = self.decoder.build_initial_state(decoder_input_source)
        decoder_input = batch_loader.get_raw_input_from_sentences([batch_loader.go_label])
        if use_cuda:
            decoder_input = decoder_input.cuda()

        logits, initial_state = self.decoder(None, decoder_input, z, 0.0, initial_state)
        logits = logits.view(-1, self.params.vocab_size)
        predictions = F.softmax(logits, dim=-1)

        # sequences = [[list(), 0.0]]
        sequences = [[list(), 0.0, initial_state, decoder_input, False]]

        # walk over each step in sequence
        for seq in range(seq_len):
            all_candidates = list()
            # expand each current candidate
            for i in range(len(sequences)):
                seq, score, initial_state, decoder_input, complete = sequences[i]
                if complete:
                    all_candidates.append(sequences[i])
                    continue
                if use_cuda:
                    decoder_input = decoder_input.cuda()

                logits, initial_state = self.decoder(None, decoder_input, z, 0.0, initial_state)
                logits = logits.view(-1, self.params.vocab_size)
                prediction = F.softmax(logits, dim=-1).data.cpu().numpy()[-1]
                for j in range(prediction.shape[0]):
                    word = batch_loader.get_word_by_idx(j)
                    if word == batch_loader.unk_label:
                        continue
                    decoder_input = batch_loader.get_raw_input_from_sentences([word])
                    if word == batch_loader.end_label:
                        candidate = [seq, score - math.log(prediction[j]), initial_state, decoder_input, True]
                    else:
                        candidate = [seq + [word], score - math.log(prediction[j]), initial_state, decoder_input, False]
                    all_candidates.append(candidate)
            # order all candidates by score
            ordered = sorted(all_candidates, key=lambda tup:tup[1])
            # select k best
            sequences = ordered[:k]
        results = []
        for sequence in sequences:
            results.append(' '.join(sequence[0]))
        return results

    def sample_with_phrase(self, batch_loader, seq_len, use_cuda, source_sent):
        pass
