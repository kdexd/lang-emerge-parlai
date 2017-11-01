from __future__ import division

import numpy as np
import torch
from torch import autograd, optim

from parlai.core.params import ParlaiParser

import options
from bots import Questioner, Answerer
from dataloader import ShapesQADataset
from world import QAWorld


opt = options.read()

#------------------------------------------------------------------------
# setup dataset
#------------------------------------------------------------------------
qa = {
    'train': ShapesQADataset(opt, 'train'),
    'val': ShapesQADataset(opt, 'val')
}
# pull out few attributes from dataset in main opts for other bots to use
opt['props'] = qa['train'].properties
opt['task_vocab'] = len(qa['train'].task_defn)

#------------------------------------------------------------------------
# setup experiment
#------------------------------------------------------------------------
questioner = Questioner(opt)
answerer = Answerer(opt)
print(questioner)
print(answerer)
world = QAWorld(opt, questioner, answerer)

optimizer = optim.Adam([{'params': world.abot.parameters(),
                         'lr': opt['learning_rate']},
                        {'params': world.qbot.parameters(),
                         'lr': opt['learning_rate']}])

#------------------------------------------------------------------------
# train agents
#------------------------------------------------------------------------
num_iter_per_epoch = int(np.ceil(len(qa['train']) / opt['batch_size']))
num_iter_per_epoch = max(1, num_iter_per_epoch)

matches = {}
accuracy = {}

for epoch_id in range(opt['num_epochs']):
    for iter_id in range(num_iter_per_epoch):
        optimizer.zero_grad()

        if 'train' in matches:
            batch = qa['train'].get_batch(matches['train'])
        else:
            batch = qa['train'].get_batch()

        batch['image'], batch['task'] = autograd.Variable(batch['image']), \
                                        autograd.Variable(batch['task'])

        # previous episode was done, pass a new batch to questioner
        world.qbot.observe({'batch': batch, 'episode_done': True})

        for round in range(opt['num_rounds']):
            world.parley()

        # predict image attributes
        guess_token, guess_distr = world.qbot.predict(batch['task'], 2)
        # compute reward for this batch
        reward = torch.Tensor(opt['batch_size'], 1).fill_(- 10 * opt['rl_scale'])

        qa['train'].pretty_print(world.acts, guess_token, batch)
        # both attributes need to match
        first_match = guess_token[0].data == batch['labels'][:, 0:1]
        second_match = guess_token[1].data == batch['labels'][:, 1:2]
        reward[first_match & second_match] = opt['rl_scale']

        # record cumulative reward in world
        batch_reward = torch.mean(reward) / opt['rl_scale']
        if not world.cumulative_reward:
            world.cumulative_reward = batch_reward
        world.cumulative_reward = 0.95 * world.cumulative_reward + 0.05 * batch_reward

        # reinforce all actions for qbot and abot
        world.qbot.reinforce(reward)
        world.abot.reinforce(reward)

        # backward pass on actions
        autograd.backward(world.qbot.actions + world.abot.actions,
                          [None for _ in world.qbot.actions + world.abot.actions],
                          retain_graph=True)

        # clamp all gradients between (-5, 5)
        for parameter in world.qbot.parameters():
            parameter.grad.data.clamp_(min=-5, max=5)
        for parameter in world.abot.parameters():
            parameter.grad.data.clamp_(min=-5, max=5)

        optimizer.step()

    #--------------------------------------------------------------------
    # training and validation metrics
    #--------------------------------------------------------------------
    world.qbot.eval()
    world.abot.eval()
    for dtype in ['train', 'val']:
        batch = qa['train'].get_complete_data()
        batch['image'], batch['task'] = autograd.Variable(batch['image']), \
                                        autograd.Variable(batch['task'])
        world.qbot.observe({'batch': batch, 'episode_done': True})

        for round in range(opt['num_rounds']):
            world.parley()
        # compute accuracy for color, shape, and both
        guess_token, guess_distr = world.qbot.predict(batch['task'], 2)
        first_match = guess_token[0].data == batch['labels'][:, 0].long()
        second_match = guess_token[1].data == batch['labels'][:, 1].long()
        matches[dtype] = first_match & second_match
        accuracy[dtype] = 100 * torch.sum(matches[dtype]) / float(matches[dtype].size(0))

    # switch to train
    world.qbot.train()
    world.abot.train()

    # break if train accuracy reaches 100%
    if accuracy['train'] == 100:
        break
