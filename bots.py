from __future__ import absolute_import, division
import math

import torch
from torch import nn
from torch.autograd import Variable

from parlai.core.agents import Agent


class ChatBotAgent(Agent, nn.Module):
    """Parent class for both, questioner and answerer bots."""
    def __init__(self, opt, shared=None):
        super(ChatBotAgent, self).__init__(opt, shared)
        self.id = 'ChatBotAgent'

        # standard initializations
        self.h_state = torch.Tensor()
        self.c_state = torch.Tensor()
        self.eval_flag = False
        self.actions = []

        # modules (common)
        self.in_net = nn.Embedding(self.opt['in_vocab_size'], self.opt['embed_size'])
        self.out_net = nn.Linear(self.opt['hidden_size'], self.opt['out_vocab_size'])
        self.softmax = nn.Softmax()

        # xavier init of in_net and out_net
        for module in {self.in_net, self.out_net}:
            fan_in = module.weight.data.size(0)
            fan_out = module.weight.data.size(1)
            module.weight.data.normal_(0, math.sqrt(2 / (fan_in + fan_out)))

    def observe(self, observation):
        """Given an input token, interact for next round."""
        self.observation = observation
        # embed and pass through LSTM
        token_embeds = self.model.in_net(observation['text'])

        # concat with image representation
        if 'image' in observation:
            token_embeds = torch.cat((token_embeds, observation['image']), 1)

        # now pass it through rnn
        self.model.h_state, self.model.c_state = self.model.rnn(
            token_embeds, (self.model.h_state, self.model.c_state))

    def act(self):
        """Speak a token."""
        # compute softmax and choose a token
        out_distr = self.model.softmax(self.model.out_net(self.model.h_state))

        # if evaluating
        if self.model.eval_flag:
            _, actions = out_distr.max(1)
            actions = actions.unsqueeze(1)
        else:
            actions = out_distr.multinomial()
            # record actions
            self.actions.append(actions)
        return actions.squeeze(1)

    def reinforce(self):
        """Reinforce each state wth reward."""
        for action in self.actions:
            action.reinforce(self.observation['reward'])

    def reset(self, batch_size, retain_actions=False):
        """Reset model and actions."""
        self.h_state.resize_(batch_size, self.hidden_size).fill_(0)
        self.h_state = Variable(self.h_state)
        self.c_state.resize_(batch_size, self.hidden_size).fill_(0)
        self.c_state = Variable(self.c_state)

        if not retain_actions:
            self.actions = []
        super(ChatBotAgent, self).reset()

    def train(self):
        """Switch to training mode."""
        self.eval_flag = False

    def eval(self):
        """Switch to evaluation mode."""
        self.eval_flag = True

    def forward(self, *inputs):
        """Dummy forward pass."""
        pass


class Questioner(ChatBotAgent):
    def __init__(self, opt, shared=None):
        opt['in_vocab_size'] = opt['q_in_vocab']
        opt['out_vocab_size'] = opt['q_out_vocab']
        super(Questioner, self).__init__(opt, shared)
        self.id = 'QBot'

        # always condition on task
        self.rnn = nn.LSTMCell(self.opts['embed_size'], self.opts['hidden_size'])

        # additional prediction network
        # start token included
        num_preds = sum([len(ii) for ii in self.opt['props'].values()])
        # network for predicting
        self.predict_rnn = nn.LSTMCell(self.embed_size, self.hidden_size)
        self.predict_net = nn.Linear(self.hidden_size, num_preds)

        # xavier init of rnn, predict_rnn, predict_net
        for module in {self.rnn, self.predict_rnn, self.predict_net}:
            fan_in = module.weight.data.size(0)
            fan_out = module.weight.data.size(1)
            module.weight.data.normal_(0, math.sqrt(2 / (fan_in + fan_out)))

        # setting offset
        self.task_offset = opt['a_out_vocab'] + opt['q_out_vocab']
        self.listen_offset = opt['a_out_Vocab']

    def predict(self, tasks, num_tokens):
        """Return an answer from the task."""
        guess_tokens = []
        guess_distr = []

        for _ in xrange(num_tokens):
            # explicit task dependence
            task_embeds = self.in_net(tasks)
            # compute softmax and choose a token
            self.h_state, self.c_state = self.predict_rnn(
                task_embeds, (self.h_state, self.c_state))
            out_distr = self.softmax(self.predict_net(self.h_state))

            # if evaluating
            if self.eval_flag:
                _, actions = out_distr.max(1)
            else:
                actions = out_distr.multinomial()
                # record actions
                self.actions.append(actions)

            # record the guess and distribution
            guess_tokens.append(actions)
            guess_distr.append(out_distr)

        # return prediction
        return guess_tokens, guess_distr

    def embed_task(self, tasks):
        """Embed the image."""
        return self.in_net(tasks + self.task_offset)


class Answerer(ChatBotAgent):
    def __init__(self, opt, shared=None):
        opt['in_vocab_size'] = opt['a_in_vocab']
        opt['out_vocab_size'] = opt['a_out_vocab']
        super(Answerer, self).__init__(opt, shared)

        # number of attribute values
        num_attrs = sum([len(ii) for ii in self.opt['props'].values()])
        # number of unique attributes
        num_unique_attrs = len(self.opt['props'])

        # rnn input size
        rnn_input_size = num_unique_attrs * self.opt['img_feat_size'] + self.opt['embed_size']

        self.img_net = nn.Embedding(num_attrs, self.opt['img_feat_size'])
        self.rnn = nn.LSTMCell(rnn_input_size, self.opt['hidden_size'])

        # xavier init of in_net and out_net
        for module in {self.img_net, self.rnn}:
            fan_in = module.weight.data.size(0)
            fan_out = module.weight.data.size(1)
            module.weight.data.normal_(0, math.sqrt(2 / (fan_in + fan_out)))

        # set offset
        self.listen_offset = opt['q_out_vocab']

    # Embedding the image
    def embed_image(self, batch):
        embeds = self.img_net(batch)
        # concat instead of add
        features = torch.cat(embeds.transpose(0, 1), 1)
        # add features
        #features = torch.sum(embeds, 1).squeeze(1)
        return features
