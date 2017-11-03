from __future__ import division

from datetime import datetime
import os

import numpy as np
import torch
from torch import optim
from torch.autograd import Variable

from parlai.core.params import ParlaiParser

import options
from bots import Questioner, Answerer
from dataloader import ShapesQADataset
from world import QAWorld


opt = options.read()

#-------------------------------------------------------------------------------------------------
# setup dataset and opts
#-------------------------------------------------------------------------------------------------
dataset = ShapesQADataset(opt)
# pull out few attributes from dataset in main opts for other bots to use
opt['props'] = dataset.properties
opt['task_vocab'] = len(dataset.task_defn)

# make a directory to save checkpoints
timestamp = datetime.strftime(datetime.utcnow(), '%a-%d-%b-%Y-%X')
opt['save_path'] = os.path.join(opt['save_path'], 'world-{}'.format(timestamp))
os.makedirs(opt['save_path'])

#-------------------------------------------------------------------------------------------------
# setup experiment
#-------------------------------------------------------------------------------------------------
questioner = Questioner(opt)
answerer = Answerer(opt)
# this reward tensor is re-used every iteration
reward = torch.Tensor(opt['batch_size'], 1).fill_(- 10 * opt['rl_scale'])
if opt['use_gpu']:
    questioner, answerer, reward = questioner.cuda(), answerer.cuda(), reward.cuda()
print('Questioner and Answerer Bots: ')
print(questioner)
print(answerer)
world = QAWorld(opt, questioner, answerer)

optimizer = optim.Adam([{'params': world.abot.parameters(),
                         'lr': opt['learning_rate']},
                        {'params': world.qbot.parameters(),
                         'lr': opt['learning_rate']}])

#-------------------------------------------------------------------------------------------------
# train agents
#-------------------------------------------------------------------------------------------------
NUM_ITER_PER_EPOCH = max(0, int(np.ceil(len(dataset) / opt['batch_size'])))

"""``matches`` will have a tensor of booleans as values. i-th true value represents i-th example's
ground truth matching prediction in previous iteration. This dict is useful for sampling negative
examples for next iteration training."""
matches = {'train': None, 'val': None}

"""``accuracy`` dict will have training and validation accuracies updated every epoch. This dict
is useful for early stopping mechanism. Training stops if training accuracy hits 1."""
accuracy = {'train': 0.0, 'val': 0.0}

for epoch_id in range(opt['num_epochs']):
    for iter_id in range(NUM_ITER_PER_EPOCH):
        optimizer.zero_grad()

        #-----------------------------------------------------------------------------------------
        # episode batch retrieval and dialog
        #-----------------------------------------------------------------------------------------
        if matches.get('train') is not None:
            batch = dataset.get_batch('train', matches['train'])
        else:
            batch = dataset.get_batch('train')
        batch['image'], batch['task'] = Variable(batch['image']), Variable(batch['task'])
        world.qbot.observe({'batch': batch, 'episode_done': True})

        for _ in range(opt['num_rounds']):
            world.parley()
        #-----------------------------------------------------------------------------------------
        # reward formulation and reinforcement
        #-----------------------------------------------------------------------------------------
        guess_token, guess_distr = world.qbot.predict(batch['task'], 2)
        reward.fill_(- 10 * opt['rl_scale'])

        # both attributes need to match
        first_match = guess_token[0].data == batch['labels'][:, 0:1]
        second_match = guess_token[1].data == batch['labels'][:, 1:2]
        reward[first_match & second_match] = opt['rl_scale']

        # record cumulative reward in world
        batch_reward = torch.mean(reward) / opt['rl_scale']
        if not world.cumulative_reward:
            world.cumulative_reward = batch_reward
        world.cumulative_reward = 0.95 * world.cumulative_reward + 0.05 * batch_reward

        # qbot and abot observe rewards at end of episode
        world.qbot.observe({'reward': reward, 'episode_done': True})
        world.abot.observe({'reward': reward, 'episode_done': True})

        optimizer.step()

    #---------------------------------------------------------------------------------------------
    # training and validation metrics
    #---------------------------------------------------------------------------------------------

    # switch to evaluation mode
    world.qbot.eval()
    world.abot.eval()

    for dtype in ['train', 'val']:
        batch = dataset.get_complete_data(dtype)
        batch['image'], batch['task'] = Variable(batch['image']), Variable(batch['task'])
        world.qbot.observe({'batch': batch, 'episode_done': True})

        for round in range(opt['num_rounds']):
            world.parley()
        # compute accuracy for color, shape, and both
        guess_token, guess_distr = world.qbot.predict(batch['task'], 2)
        first_match = guess_token[0].data == batch['labels'][:, 0].long()
        second_match = guess_token[1].data == batch['labels'][:, 1].long()
        matches[dtype] = first_match & second_match
        accuracy[dtype] = 100 * torch.sum(matches[dtype]) / float(matches[dtype].size(0))

    # switch back to training mode
    world.qbot.train()
    world.abot.train()

    # break if train accuracy reaches 100%
    if accuracy['train'] == 100:
        break

    #---------------------------------------------------------------------------------------------
    # saving checkpoints
    #---------------------------------------------------------------------------------------------
    if epoch_id % opt['save_epoch'] == 0:
        save_path = os.path.join(opt['save_path'], 'world_epoch_{}.pth'.format(epoch_id))
        world.save_agents(save_path)

#-------------------------------------------------------------------------------------------------
# save final world checkpoint with a time stamp
#-------------------------------------------------------------------------------------------------
timestamp = datetime.strftime(datetime.utcnow(), '%a-%d-%b-%Y-%X')
final_save_path = os.path.join(opt['self_path'], 'final_world_{}.pth'.format(timestamp))
print('Saving at final world at: {}'.format(final_save_path))
world.save_agents(final_save_path)
