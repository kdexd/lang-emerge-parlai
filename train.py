#!/usr/bin/python3
"""Script for training the questioner and answerer agents in dialog world. Both agents hold
multiple rounds of dialoues per episode, after which qbot makes a prediction about the attributes
of image,  according to the assigned task.

Few global variables defined here are explained:

Global Variables
----------------
OPT : dict
    Command-line arguments. Refer ``options.py``

matches : dict
    Has keys 'train' and 'val'. Contains tensor of booleans as values. i-th true value represents
    i-th example's ground truth matching prediction in previous iteration. This dict is useful
    for sampling negative examples for next iteration training.
accuracy : dict
    Has keys 'train' and 'val'. Will have training and validation accuracies updated every epoch.
    This dict is useful for early stopping mechanism. Training stops if training accuracy hits 1.

reward : torch.FloatTensor or torch.cuda.FloatTensor
    Tensor of length equal to batch size, sets reward 1 for correctly classified example and -10
    for negatively classified sample. Re-used every episode.
cumulative_reward : float
    Scalar reward for both the bots. Same for both bots as the game is perfectly cooperative.

dataset : ShapesQADataset (torch.utils.data.Dataset)
questioner : Questioner (parlai.core.agents.Agent, nn.Module)
answerer : Answerer (parlai.core.agents.Agent, nn.Module)
world : QAWorld (parlai.core.worlds.DialogPartnerWorld)
optimizer : optim.Adam
"""
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


OPT = options.read()

# seed random for reproducibility
if OPT.get('use_gpu'):
    torch.cuda.manual_seed_all(1337)
else:
    torch.manual_seed(1337)

#-------------------------------------------------------------------------------------------------
# setup dataset and opts
#-------------------------------------------------------------------------------------------------
dataset = ShapesQADataset(OPT)
# pull out few attributes from dataset in main opts for other bots to use
OPT['props'] = dataset.properties
OPT['task_vocab'] = len(dataset.task_defn)

# make a directory to save checkpoints
timestamp = datetime.strftime(datetime.utcnow(), '%d-%b-%Y-%X')
OPT['save_path'] = os.path.join(OPT['save_path'], 'world-{}'.format(timestamp))
os.makedirs(OPT['save_path'])

#-------------------------------------------------------------------------------------------------
# setup experiment
#-------------------------------------------------------------------------------------------------
questioner = Questioner(OPT)
answerer = Answerer(OPT)
# this reward tensor is re-used every iteration
reward = torch.Tensor(OPT['batch_size'], 1).fill_(- 10 * OPT['rl_scale'])
if OPT.get('use_gpu'):
    questioner, answerer, reward = questioner.cuda(), answerer.cuda(), reward.cuda()
print('Questioner and Answerer Bots: ')
print(questioner)
print(answerer)
world = QAWorld(OPT, questioner, answerer)

optimizer = optim.Adam([{'params': world.abot.parameters(),
                         'lr': OPT['learning_rate']},
                        {'params': world.qbot.parameters(),
                         'lr': OPT['learning_rate']}])

#-------------------------------------------------------------------------------------------------
# train agents
#-------------------------------------------------------------------------------------------------
NUM_ITER_PER_EPOCH = max(0, int(np.ceil(len(dataset) / OPT['batch_size'])))

matches = {'train': None, 'val': None}
accuracy = {'train': 0.0, 'val': 0.0}

for epoch_id in range(OPT['num_epochs']):
    for iter_id in range(NUM_ITER_PER_EPOCH):
        optimizer.zero_grad()

        #-----------------------------------------------------------------------------------------
        # episode batch retrieval and dialog
        #-----------------------------------------------------------------------------------------
        if matches.get('train') is not None:
            batch = dataset.random_batch('train', matches['train'])
        else:
            batch = dataset.random_batch('train')
        batch['image'], batch['task'] = Variable(batch['image']), Variable(batch['task'])
        world.qbot.observe({'batch': batch, 'episode_done': True})

        for _ in range(OPT['num_rounds']):
            world.parley()
        #-----------------------------------------------------------------------------------------
        # reward formulation and reinforcement
        #-----------------------------------------------------------------------------------------
        guess_token, guess_distr = world.qbot.predict(batch['task'], 2)
        reward.fill_(- 10 * OPT['rl_scale'])

        # both attributes need to match
        first_match = guess_token[0].data == batch['labels'][:, 0:1]
        second_match = guess_token[1].data == batch['labels'][:, 1:2]
        reward[first_match & second_match] = OPT['rl_scale']

        # record cumulative reward in world
        batch_reward = torch.mean(reward) / OPT['rl_scale']
        if not world.cumulative_reward:
            world.cumulative_reward = batch_reward
        world.cumulative_reward = 0.95 * world.cumulative_reward + 0.05 * batch_reward

        # qbot and abot observe rewards at end of episode
        world.qbot.observe({'reward': reward, 'episode_done': True})
        world.abot.observe({'reward': reward, 'episode_done': True})

        optimizer.step()

        #-----------------------------------------------------------------------------------------
        # logging metrics
        #-----------------------------------------------------------------------------------------
        if (NUM_ITER_PER_EPOCH * epoch_id + iter_id) % 100 == 0:
            timestamp = datetime.strftime(datetime.utcnow(), '%a, %d %b %Y %X')
            print('[%s][Iter: %d][Epoch: %.2f][Reward: %.4f][Train Acc.: %.2f Val Acc.: %.2f]' % \
                  (timestamp, NUM_ITER_PER_EPOCH * epoch_id + iter_id, epoch_id,
                   world.cumulative_reward, accuracy['train'], accuracy['val']))
    #---------------------------------------------------------------------------------------------
    # training and validation metrics
    #---------------------------------------------------------------------------------------------
    world.qbot.eval()
    world.abot.eval()
    for dtype in ['train', 'val']:
        batch = dataset.complete_data(dtype)
        # make variables volatile because graph construction is not required for eval
        batch['image'] = Variable(batch['image'], volatile=True)
        batch['task'] = Variable(batch['task'], volatile=True)
        world.qbot.observe({'batch': batch, 'episode_done': True})

        for _ in range(OPT['num_rounds']):
            world.parley()
        # compute accuracy for color, shape, and both
        guess_token, guess_distr = world.qbot.predict(batch['task'], 2)
        first_match = guess_token[0].data == batch['labels'][:, 0].long()
        second_match = guess_token[1].data == batch['labels'][:, 1].long()
        matches[dtype] = first_match & second_match
        accuracy[dtype] = 100 * torch.sum(matches[dtype]) / float(matches[dtype].size(0))
    world.qbot.train()
    world.abot.train()

    # break if train accuracy reaches 100%
    if accuracy['train'] == 100:
        break

    #---------------------------------------------------------------------------------------------
    # saving checkpoints
    #---------------------------------------------------------------------------------------------
    if epoch_id % OPT['save_epoch'] == 0:
        save_path = os.path.join(OPT['save_path'], 'world_epoch_%s.pth' % str(epoch_id).zfill(5))
        world.save_agents(save_path)

#-------------------------------------------------------------------------------------------------
# save final world checkpoint with a time stamp
#-------------------------------------------------------------------------------------------------
timestamp = datetime.strftime(datetime.utcnow(), '%d-%b-%Y-%X')
final_save_path = os.path.join(OPT['save_path'], 'final_world_{}.pth'.format(timestamp))
print('Saving at final world at: {}'.format(final_save_path))
world.save_agents(final_save_path)
