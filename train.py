from __future__ import division

import numpy as np
import torch
from torch import autograd, optim

from parlai.core.params import ParlaiParser

from bots import Questioner, Answerer
from dataloader import ShapesQADataset
from world import QAWorld


parser = ParlaiParser()
parser.add_argument_group('Dataset Parameters')
parser.add_argument('--data-path', default='data/synthetic_dataset.json', type=str,
                    help='Path to the training/val dataset file')
parser.add_argument('--neg-fraction', default=0.8, type=float,
                    help='Fraction of negative examples in batch')

parser.add_argument_group('Model Parameters')
parser.add_argument('--hidden-size', default=100, type=int,
                    help='Hidden Size for the language models')
parser.add_argument('--embed-size', default=20, type=int,
                    help='Embed size for words')
parser.add_argument('--img-feat-size', default=20, type=int,
                    help='Image feature size for each attribute')
parser.add_argument('--q-out-vocab', default=3, type=int,
                    help='Output vocabulary for questioner')
parser.add_argument('--a-out-vocab', default=4, type=int,
                    help='Output vocabulary for answerer')

parser.add_argument('--rl-scale', default=100.0, type=float,
                    help='Weight given to rl gradients')
parser.add_argument('--num-rounds', default=2, type=int,
                    help='Number of rounds between Q and A')
parser.add_argument('--remember', dest='remember', action='store_true',
                    help='Turn on/off for ABot with memory')

parser.add_argument_group('Optimization Hyperparameters')
parser.add_argument('--batch-size', default=1, type=int,
                    help='Batch size during training')
parser.add_argument('--num-epochs', default=10000, type=int,
                    help='Max number of epochs to run')
parser.add_argument('--learning-rate', default=1e-3, type=float,
                    help='Initial learning rate')
parser.add_argument('--use-gpu', dest='use_gpu', action='store_true')

opt = parser.parse_args()

#------------------------------------------------------------------------
# setup dataset
#------------------------------------------------------------------------
qa_train = ShapesQADataset(opt, 'train')
# pull out few attributes from dataset in main opts for other bots to use
opt['props'] = qa_train.props
opt['task_vocab'] = qa_train.task_select.shape[0]

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
num_iter_per_epoch = int(np.ceil(len(qa_train) / opt['batch_size']))
num_iter_per_epoch = max(1, num_iter_per_epoch)


for epoch_id in range(opt['num_epochs']):
    for iter_id in range(num_iter_per_epoch):
        optimizer.zero_grad()

        batch = qa_train.get_batch(opt['batch_size'])
        batch['image'], batch['task'] = autograd.Variable(batch['image']), \
                                        autograd.Variable(batch['task'])

        # previous episode was done, pass a new batch to questioner
        world.qbot.observe({'batch': batch, 'episode_done': True})

        for round in range(opt['num_rounds']):
            world.parley()
        # predict image attributes, compute reward
        guess_token, guess_distr = world.qbot.predict(batch['task'], 2)

        # compute reward for this batch
        reward = torch.Tensor(opt['batch_size'], 1).fill_(- 10 * opt['rl_scale'])

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
        print("STEP DONE: " + str(world.cumulative_reward))
