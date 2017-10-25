from __future__ import absolute_import
import math

import torch
from torch import nn

from parlai.core.agents import Agent


class ChatBotAgent(Agent):
    """Parent class for both, questioner and answerer bots."""
    def __init__(self, opt, shared=None):
        super(ChatBotAgent, self).__init__(opt, shared)
        self.id = 'ChatBotAgent'

        # standard initializations
        self.h_state = torch.Tensor()
        self.c_state = torch.Tensor()
        self.eval_flag = False

        # modules (common)
        self.in_net = nn.Embedding(self.opt['in_vocab_size'], self.opt['embed_size'])
        self.out_net = nn.Linear(self.opt['hidden_size'], self.opt['out_vocab_size'])
        self.softmax = nn.Softmax()

        # xavier init of in_net and out_net
        for module in {self.in_net, self.out_net}:
            fan_in = module.weight.data.size(0)
            fan_out = module.weight.data.size(1)
            module.weight.data.normal_(0, math.sqrt(2 / (fan_in + fan_out)))

    def reinforce(self):
        """Reinforce each state wth reward."""
        for action in self.actions:
            action.reinforce(self.observation['reward'])

    def reset(self, batch_size, retain_actions=False):
        """Reset model and actions."""
        self.model.reset_states(batch_size)
        if not retain_actions:
            self.actions = []
        super(ChatBotAgent, self).reset()
