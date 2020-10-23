# -*- coding:utf-8 -*-

import copy
import numpy as np
import torch
import torch.nn.functional as F

"""
	Code base taken from: https://github.com/HeroKillerEver/SeqGAN-Pytorch
"""
class Rollout(object):
	"""Rollout policy"""
	def __init__(self, generator, discriminator, update_rate):
		super(Rollout, self).__init__()
		self.generator_theta = generator
		self.generator_beta = copy.deepcopy(generator)
		self.discriminator = discriminator
		self.update_rate = update_rate


	def reward(self, x, rollout_num, input, use_cuda, batch_loader):

		batch_size, seq_len, embed_size = x.size()

		rewards = []
		# print(f'batch_size: {batch_size}')
		# print(f'seq_len: {seq_len}')
		# print(f'embed_size: {embed_size}')

		for i in range(rollout_num):
			# print(f'Rollout {i+1}/{rollout_num}')
			for l in range(1, seq_len):
				data = x[:, 0:l, :]
				# samples = [batch_size, seq_len, embed_size(300)]
				samples = self.generator_beta.sample_with_seq(data, seq_len, input, use_cuda, batch_loader) # (batch_size, sequence_len)
				if use_cuda:
					samples = samples.cuda()

				# reward = F.sigmoid(self.discriminator(samples)) # (batch_size, 1)
				reward = self.discriminator(samples) # (batch_size, 1)
				reward = reward.data.cpu().numpy()
				if i == 0:
					rewards.append(reward)
				else:
					rewards[l-1] += reward

			# reward = F.sigmoid(self.discriminator(x))
			if use_cuda:
				x = x.cuda()
				
			reward = self.discriminator(x)
			reward = reward.data.cpu().numpy()
			if i == 0:
				rewards.append(reward)
			else:
				rewards[seq_len-1] += reward

		rewards = (np.array(rewards).squeeze().T) / (1. * rollout_num) # (batch_size, sequence_len)

		return rewards


	def update_params(self):
		dic = {}
		for name, param in self.generator_theta.named_parameters():
			dic[name] = param.data
		for name, param in self.generator_beta.named_parameters():
			if name.startswith('emb'):
				param.data = dic[name]
			else:
				param.data = self.update_rate * param.data + (1 - self.update_rate) * dic[name]
