# -*- coding:utf-8 -*-

import copy
import numpy as np
import torch as t
import torch.nn.functional as F
from torch.autograd import Variable

import time
"""
	Code base taken from: https://github.com/HeroKillerEver/SeqGAN-Pytorch
"""
class Rollout(object):
	"""Rollout policy"""
	def __init__(self, generator, discriminator, update_rate, rollout_num):
		super(Rollout, self).__init__()
		self.generator = generator
		self.generator_copy = copy.deepcopy(generator)
		self.discriminator = discriminator
		self.update_rate = update_rate
		self.rollout_num = rollout_num


	def reward(self, x, encoder_input, decoder_input_source, use_cuda, batch_loader):
		"""
		Args:
			x : (batch_size, seq_len, embed_size) generated data
			rollout_num : roll-out number
			input : (batch_size, seq_len, embed_size) input data
		"""
		with t.no_grad():
			[batch_size, seq_len, embed_size] = x.size()

			mu, logvar = self.generator_copy.encoder(encoder_input[0], None)
			std = t.exp(0.5 * logvar)

			z = Variable(t.randn([batch_size, self.generator_copy.params.latent_variable_size]))
			if use_cuda:
				z = z.cuda()
			z = z * std + mu

			initial_states = [self.generator_copy.decoder.build_initial_state(decoder_input_source)]
			# rewards = []
			rewards = t.zeros([seq_len, batch_size])
			if use_cuda:
				rewards.cuda()
			# time_s = 0
			# time_d = 0
			idx = 0
			for i in range(self.rollout_num):
				for l in range(1, seq_len):
					samples, next_initial_state = self.generator_copy.sample(x[:, 0:l, :], seq_len, z, initial_states[l-1], use_cuda, batch_loader) # (batch_size, sequence_len)
					# if use_cuda:
					# 	samples = samples.cuda()

					reward = t.sigmoid(self.discriminator(samples)) # (batch_size, 1)
					# reward = reward.data.cpu().numpy()
					rewards[l] = reward
					if i == 0:
						initial_states.append(next_initial_state)

					# idx += 1
						# rewards.append(reward)
					# else:
						# rewards[l-1] += reward
					# time_d += (time.time_ns() - t0)

				# if use_cuda:
					# x = x.cuda()

				reward = t.sigmoid(self.discriminator(x))
				rewards[-1] = reward
				# reward = reward.data.cpu().numpy() # Detach from computational graph
				# if i == 0:
					# rewards.append(reward)
				# else:
					# rewards[seq_len-1] += reward

			rewards = (np.array(rewards).squeeze().T) / (1. * self.rollout_num) # (batch_size, sequence_len)
			rewards = t.mean(rewards.view(batch_size, seq_len, rollout_num), dim=-1)
		# print(f'Time spent sampling: {time_s /(10**6)}ms (avg. {time_s /(10**6)/(self.rollout_num)}/rollout)')
		# print(f'Time spent in discriminator: {time_d /(10**6)}ms (avg. {time_d /(10**6)/(self.rollout_num)}/rollout)')

		return rewards

	def update_params(self):
		dic = {}
		for name, param in self.generator.named_parameters():
			dic[name] = param.data
		for name, param in self.generator_copy.named_parameters():
			if name.startswith('emb'):
				param.data = dic[name]
			else:
				param.data = self.update_rate * param.data + (1 - self.update_rate) * dic[name]
