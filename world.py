from __future__ import division

import torch
from torch.autograd import Variable
from parlai.core.worlds import World


class QAWorld(World):
    @staticmethod
    def add_cmdline_args(parser):
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
        parser.add_argument('--neg-fraction', default=0.8, type=float,
                            help='Fraction of negative examples in batch')
        return parser

    def __init__(self, opt, questioner, answerer, data_fetcher, shared=None):
        self.id = 'QAWorld'
        self.qbot = questioner
        self.abot = answerer
        self.dbot = data_fetcher
        self.acts = []
        self.reward = torch.Tensor(opt['batchsize'], 1)
        self.cumulative_reward = 0
        super(QAWorld, self).__init__(opt, [self.qbot, self.abot], shared)

    def parley(self):
        batch = self.dbot.act()
        batch['image'], batch['task'] = Variable(batch['image']), Variable(batch['task'])

        self.qbot.reset()
        self.abot.reset()

        # get image representation
        img_embed = self.abot.embed_image(batch['image'])

        # answer first question, which was from the dataset
        abot_ans = {
            'text': batch['task'] + self.qbot.task_offset,
            'id': self.abot.id
        }

        # ask multiple rounds of questions and record conversation
        self.acts = []
        for _ in range(self.opt['num_rounds']):
            # observe answer, ask q_r and observe q_r as well
            self.qbot.observe(abot_ans)
            qbot_ques = self.qbot.act()

            # clone and randomize a bit
            qbot_ques['text'] = qbot_ques['text'].detach()
            self.qbot.observe({
                'text': qbot_ques['text'] + self.qbot.listen_offset,
                'id': self.qbot.id
            })

            # forget answer if abot is memory-less
            if not self.opt['remember']:
                self.abot.reset(retain_actions=True)

            # observe question and image, also observe answer
            self.abot.observe({
                'text': qbot_ques['text'],
                'id': self.qbot.id,
                'image': img_embed
            })
            abot_ans = self.abot.act()

            # clone and randomize a bit
            abot_ans['text'] = abot_ans['text'].detach()
            self.abot.observe({
                'text': abot_ans['text'] + self.abot.listen_offset,
                'id': self.abot.id,
                'image': img_embed
            })

            self.acts.extend([qbot_ques, abot_ans])

        # observe last answer
        self.qbot.observe(abot_ans)

        # predict image attributes, compute reward
        guess_token, guess_distr = self.qbot.predict(batch['task'], 2)

        # compute reward
        self.reward.fill_(- 10 * self.opt['rl_scale'])

        # both attributes need to match
        first_match = guess_token[0].data == batch['labels'][:, 0:1]
        second_match = guess_token[1].data == batch['labels'][:, 1:2]
        self.reward[first_match & second_match] = self.opt['rl_scale']

        # reinforce all actions for qbot and abot
        self.qbot.reinforce(self.reward)
        self.abot.reinforce(self.reward)

        # backward pass on actions
        for action in self.qbot.actions + self.abot.actions:
            action.backward(retain_graph=True)

        # clamp all gradients between (-5, 5)
        for parameter in self.qbot.parameters():
            parameter.grad.data.clamp_(min=-5, max=5)
        for parameter in self.abot.parameters():
            parameter.grad.data.clamp_(min=-5, max=5)

        # cumulative reward
        batch_reward = torch.mean(self.reward) / self.opt['rl_scale']
        self.cumulative_reward = 0.95 * self.cumulative_reward + 0.05 * batch_reward

