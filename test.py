import itertools
import json
import random

import torch
from torch.autograd import Variable
from parlai.core.params import ParlaiParser

from bots import Questioner, Answerer
from dataloader import ShapesQADataset
from world import QAWorld


parser = ParlaiParser()
parser.add_argument('--load-path', type=str,
                    help='path to pth file of the world checkpoint')

OPT = parser.parse_args()

#-------------------------------------------------------------------------------------------------
# setup dataset and world from checkpoint
#-------------------------------------------------------------------------------------------------
world_dict = torch.load(OPT['load_path'])

dataset = ShapesQADataset(world_dict['opt'])
questioner = Questioner(world_dict['opt'])
answerer = Answerer(world_dict['opt'])
if world_dict['opt'].get('use_gpu'):
    questioner, answerer = questioner.cuda(), answerer.cuda()

questioner.load_state_dict(world_dict['qbot'])
answerer.load_state_dict(world_dict['abot'])

world = QAWorld(world_dict['opt'], questioner, answerer)

print('Loaded world from checkpoint: {}',format(OPT['load_path']))
print('Questioner and Answerer Bots: ')
print(world.qbot)
print(world.abot)

world.qbot.eval()
world.abot.eval()

#-------------------------------------------------------------------------------------------------
# test agents
#-------------------------------------------------------------------------------------------------
for dtype in ['train', 'val']:
    batch = dataset.complete_data(dtype)
    # make variables volatile because graph construction is not required for eval
    batch['image'] = Variable(batch['image'], volatile=True)
    batch['task'] = Variable(batch['task'], volatile=True)
    world.qbot.observe({'batch': batch, 'episode_done': True})

    for _ in range(world.opt['num_rounds']):
        world.parley()
    guess_token, guess_distr = world.qbot.predict(batch['task'], 2)

    # check how much do first attribute, second attribute, both and at least one match
    first_match = guess_token[0].data == batch['labels'][:, 0].long()
    second_match = guess_token[1].data == batch['labels'][:, 1].long()
    both_matches = first_match & second_match
    atleast_match = first_match | second_match

    # compute accuracy according to matches
    first_accuracy = 100 * torch.mean(first_match.float())
    second_accuracy = 100 * torch.mean(second_match.float())
    atleast_accuracy = 100 * torch.mean(atleast_match.float())
    accuracy = 100 * torch.mean(both_matches.float())
    print('Overall accuracy [%s]: %.2f (first: %.2f, second: %.2f, atleast_one: %.2f)'
                    % (dtype, accuracy, first_accuracy, second_accuracy, atleast_accuracy))

