"""Module with classes for questioner and answerer bots. A generic ChatBotAgent is defined,
which is extended separate by Questioner and Answerer bot classes.

Refer ParlAI docs on general semantics of a ParlAI Agent:
    * http://parl.ai/static/docs/basic_tutorial.html#agents
    * http://parl.ai/static/docs/agents.html#parlai.core.agents.Agent
"""
from __future__ import absolute_import, division
import math

import torch
from torch import nn
from torch.autograd import Variable
from torch.autograd import backward as autograd_backward

from parlai.core.agents import Agent


def xavier_init(module):
    """Xavier initializer for module parameters."""
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
    """Parent class for both, questioner and answerer bots. Extends a ParlAI Agent and PyTorch
    module. This class is generic implementation of how a bot should look like / act and observe
    in a generic ParlAI dialog world. It comprises of state tensor, actions list and
    observation dict.

    This parent class provides a ``listen_net`` embedding module to embed the text tokens. Also,
    ``speak_net`` module takes out a token to be given as action, based on state of agent. Both
    questioner and answerer agents observe and act in a generic way.

    Attributes
    ----------
    opt : dict
        Command-line opts passed into constructor from the world.
    observation : dict
        Observations dict exchanged during dialogs and on starting fresh episode. Has keys as
        described in ParlAI docs ('text', 'image', 'episode_done', 'reward').
    actions : list
        List of action tensors by agent, acted in the current dialog episode.
    h_state, c_state : torch.autograd.Variable
        State of the agent.
    eval_flag : boolean
        Flag indicating whether agent in training or evaluation mode.
    listen_net : nn.Embedding
    speak_net : nn.Linear
    softmax : nn.Softmax
    """
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
        self.listen_net = nn.Embedding(self.opt['in_vocab_size'], self.opt['embed_size'])
        self.speak_net = nn.Linear(self.opt['hidden_size'], self.opt['out_vocab_size'])
        self.softmax = nn.Softmax()

        # xavier init of listen_net and speak_net
        for module in {self.listen_net, self.speak_net}:
            module = xavier_init(module)

    def observe(self, observation):
        """Given an input token, interact for next round."""
        self.observation = observation
        if not observation.get('episode_done'):
            # embed and pass through LSTM
            token_embeds = self.listen_net(observation['text'])

            # concat with image representation (valid for abot)
            if 'image' in observation:
                token_embeds = torch.cat((token_embeds, observation['image']), 1)
            # remove all dimensions with size one
            token_embeds = token_embeds.squeeze(1)
            # update agent state using these tokens
            self.h_state, self.c_state = self.rnn(token_embeds, (self.h_state, self.c_state))
        else:
            if observation.get('reward') is not None:
                for action in self.actions:
                    action.reinforce(observation['reward'])
                autograd_backward(self.actions, [None for _ in self.actions], retain_graph=True)

                # clamp all gradients between (-5, 5)
                for parameter in self.parameters():
                    parameter.grad.data.clamp_(min=-5, max=5)

    def act(self):
        """Speak a token."""
        # compute softmax and choose a token
        out_distr = self.softmax(self.speak_net(self.h_state))

        if self.eval_flag:
            _, actions = out_distr.max(1)
            actions = actions.unsqueeze(1)
        else:
            actions = out_distr.multinomial()
            self.actions.append(actions)
        return {'text': actions.squeeze(1), 'id': self.id}

    def reset(self, batch_size=None, retain_actions=False):
        """Reset state and actions. ``opt.batch_size`` is not always used because batch_size
        changes when complete data is passed (for validation)."""
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
    """Questioner bot - extending a ParlAI Agent as well as a PyTorch module. Answerer is modeled
    as a combination of a speaker network, a listener LSTM, and a prediction network.

    At the start of new episode of dialog, a task is observed by questioner bot, which is embedded
    via listener LSTM. At each round, questioner observes the answer and acts by modelling the
    probability of output utterances based on previous state. After observing the reply from
    answerer, the listener LSTM updates the state by processing both tokens (question/answer) of
    the dialog exchange. In the final round, the prediction LSTM is unrolled twice to produce
    questioner's prediction based on the final state and assigned task.

    Attributes
    ----------
    rnn : nn.LSTMCell
        Listener LSTM module. Embedding module before listener is provided by base class.
    predict_rnn, predict_net : nn.LSTMCell, nn.Linear
        Collectively form the prediction network module.
    task_offset : int
        Offset in terms of one-hot encoding, task vectors come after width equal to question and
        answer vocabulary.
    listen_offset : int
        Offset due to listening response of answer bot. Answer token one-hot vectors would
        require width equal to answer vocabulary - next question vectors would be after that.
    """
    def __init__(self, opt, shared=None):
        opt['in_vocab_size'] = opt['q_out_vocab'] + opt['a_out_vocab'] + opt['task_vocab']
        opt['out_vocab_size'] = opt['q_out_vocab']
        super(Questioner, self).__init__(opt, shared)
        self.id = 'QBot'

        # always condition on task
        self.rnn = nn.LSTMCell(self.opt['embed_size'], self.opt['hidden_size'])

        # additional prediction network, start token included
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
            task_embeds = self.listen_net(tasks)
            task_embeds = task_embeds.squeeze()
            # unroll twice, compute softmax and choose a token
            self.h_state, self.c_state = self.predict_rnn(task_embeds,
                                                          (self.h_state, self.c_state))
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
    """Answerer bot - extending a ParlAI Agent as well as a PyTorch module. Answerer is modeled
    as a combination of a speaker network, a listener LSTM, and an image encoder.

    While observing, it embeds the received question tokens and concatenates them with image
    embeds, using them to update its state by listener LSTM. Answerer bot acts by choosing a
    token based on softmax probabilities obtained after passing the state through speak net. The
    image encoder embeds each one-hot attribute vector via a linear layer and concatenates all
    three encodings to obtain a unified image instance representation.

    Attributes
    ----------
    rnn : nn.LSTMCell
        Listener LSTM module. Embedding module before listener is provided by base class.
    img_net : nn.Embedding
        Image Instance Encoder module.
    listen_offset : int
        Offset due to listening response of question bot. Question token one-hot vectors would
        require width equal to question vocabulary - answer vectors would be after that.
    """
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

    def embed_image(self, image):
        """Embed the image attributes color, shape and style into vectors of length 20 each, and
        concatenate them to make a feature vector representing the image.
        """
        embeds = self.img_net(image)
        features = torch.cat(embeds.transpose(0, 1), 1)
        return features
