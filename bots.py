from __future__ import absolute_import, division
import math

import torch
from torch import nn
from torch.autograd import Variable
from torch.autograd import backward as autograd_backward

from parlai.core.agents import Agent


def xavier_init(module):
    for parameter in module.parameters():
        if len(parameter.data.shape) == 1:
            # 1D vector means bias
            parameter.data.fill_(0)
        else:
            fan_in = parameter.data.size(0)
            fan_out = parameter.data.size(1)
            parameter.data.normal_(0, math.sqrt(2 / (fan_in + fan_out)))
    return module


class ChatBotAgent(Agent, nn.Module):
    """Parent class for both, questioner and answerer bots."""
    def __init__(self, opt, shared=None):
        super(ChatBotAgent, self).__init__(opt, shared)
        nn.Module.__init__(self)
        self.id = 'ChatBotAgent'
        self.observation = None
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
            module = xavier_init(module)

    def observe(self, observation):
        """Given an input token, interact for next round."""
        self.observation = observation

        # if reward received, then reinforce it and backward pass on actions
        if observation.get('episode_done'):
            if observation.get('reward') is not None:
                for action in self.actions:
                    action.reinforce(observation['reward'])
                autograd_backward(self.actions, [None for _ in self.actions], retain_graph=True)

                # clamp all gradients between (-5, 5)
                for parameter in self.parameters():
                    parameter.grad.data.clamp_(min=-5, max=5)
            return
        # embed and pass through LSTM
        token_embeds = self.in_net(observation['text'])

        # concat with image representation
        if 'image' in observation:
            token_embeds = torch.cat((token_embeds, observation['image']), 1)
        # remove all dimensions with size one
        token_embeds = token_embeds.squeeze(1)
        # now pass it through rnn
        self.h_state, self.c_state = self.rnn(token_embeds, (self.h_state, self.c_state))

    def act(self):
        """Speak a token."""
        # compute softmax and choose a token
        out_distr = self.softmax(self.out_net(self.h_state))

        # if evaluating
        if self.eval_flag:
            _, actions = out_distr.max(1)
            actions = actions.unsqueeze(1)
        else:
            actions = out_distr.multinomial()
            # record actions
            self.actions.append(actions)
        return {'text': actions.squeeze(1), 'id': self.id}

    def reset(self, batch_size=None, retain_actions=False):
        """Reset model and actions. opt.batch_size is not used because batch_size is different
        when complete data is passed on."""
        if batch_size is None:
            batch_size = self.opt['batch_size']
        self.h_state = Variable(torch.zeros(batch_size, self.opt['hidden_size']))
        self.c_state = Variable(torch.zeros(batch_size, self.opt['hidden_size']))
        if self.opt.get('use_gpu'):
            self.h_state, self.c_state = self.h_state.cuda(), self.c_state.cuda()

        if not retain_actions:
            self.actions = []

    def train(self):
        """Switch to training mode."""
        self.eval_flag = False

    def eval(self):
        """Switch to evaluation mode."""
        self.eval_flag = True

    def forward(self):
        """Dummy forward pass."""
        pass


class Questioner(ChatBotAgent):
    def __init__(self, opt, shared=None):
        opt['in_vocab_size'] = opt['q_out_vocab'] + opt['a_out_vocab'] + opt['task_vocab']
        opt['out_vocab_size'] = opt['q_out_vocab']
        super(Questioner, self).__init__(opt, shared)
        self.id = 'QBot'

        # always condition on task
        self.rnn = nn.LSTMCell(self.opt['embed_size'], self.opt['hidden_size'])

        # additional prediction network
        # start token included
        num_preds = sum([len(ii) for ii in self.opt['props'].values()])
        # network for predicting
        self.predict_rnn = nn.LSTMCell(self.opt['embed_size'], self.opt['hidden_size'])
        self.predict_net = nn.Linear(self.opt['hidden_size'], num_preds)

        # xavier init of rnn, predict_rnn, predict_net
        for module in {self.rnn, self.predict_rnn, self.predict_net}:
            module = xavier_init(module)

        # setting offset
        self.task_offset = opt['q_out_vocab'] + opt['a_out_vocab']
        self.listen_offset = opt['a_out_vocab']

    def predict(self, tasks, num_tokens):
        """Return an answer from the task."""
        guess_tokens = []
        guess_distr = []

        for _ in range(num_tokens):
            # explicit task dependence
            task_embeds = self.in_net(tasks)
            task_embeds = task_embeds.squeeze()
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


class Answerer(ChatBotAgent):
    def __init__(self, opt, shared=None):
        opt['in_vocab_size'] = opt['q_out_vocab'] + opt['a_out_vocab']
        opt['out_vocab_size'] = opt['a_out_vocab']
        super(Answerer, self).__init__(opt, shared)
        self.id = 'ABot'

        # number of attribute values
        num_attrs = sum([len(ii) for ii in self.opt['props'].values()])
        # number of unique attributes
        num_unique_attrs = len(self.opt['props'])

        # rnn input size
        rnn_input_size = num_unique_attrs * self.opt['img_feat_size'] + self.opt['embed_size']

        self.img_net = nn.Embedding(num_attrs, self.opt['img_feat_size'])
        self.rnn = nn.LSTMCell(rnn_input_size, self.opt['hidden_size'])

        # xavier init of img_net and rnn
        for module in {self.img_net, self.rnn}:
            module = xavier_init(module)

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
