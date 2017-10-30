from __future__ import division

import numpy as np
import torch
from torch import autograd, optim

from parlai.core.params import ParlaiParser

from bots import Questioner, Answerer
from dataloader import DataLoaderAgent
from world import QAWorld


parser = ParlaiParser()
DataLoaderAgent.add_cmdline_args(parser)
QAWorld.add_cmdline_args(parser)
parser.add_argument_group('Optimization Hyperparameters')
parser.add_argument('--num-epochs', default=10000, type=int, help='Max number of epochs to run')
parser.add_argument('--learning-rate', default=1e-3, type=float, help='Initial learning rate')
parser.add_argument('--use-gpu', dest='use_gpu', action='store_true')

opt = parser.parse_args()

#------------------------------------------------------------------------
# setup dataset
#------------------------------------------------------------------------
data_fetcher = DataLoaderAgent(opt)
# pull out props in main opts for other bots to use
opt['props'] = data_fetcher.dataset.props
opt['task_vocab'] = len(data_fetcher.dataset.task_defn)

#------------------------------------------------------------------------
# setup experiment
#------------------------------------------------------------------------
questioner = Questioner(opt)
answerer = Answerer(opt)
print(questioner)
print(answerer)
world = QAWorld(opt, questioner, answerer, data_fetcher)

optimizer = optim.Adam([{'params': world.abot.parameters(),
                         'lr': opt['learning_rate']},
                        {'params': world.qbot.parameters(),
                         'lr': opt['learning_rate']}])

#------------------------------------------------------------------------
# train agents
#------------------------------------------------------------------------
# begin training
num_iter_per_epoch = int(np.ceil(len(data_fetcher.dataset) / opt['batchsize']))
num_iter_per_epoch = max(1, num_iter_per_epoch)

for iter_id in range(opt['num_epochs'] * num_iter_per_epoch):
    epoch = iter_id / num_iter_per_epoch

    optimizer.zero_grad()
    world.parley()
    optimizer.step()
    print("STEP DONE")
