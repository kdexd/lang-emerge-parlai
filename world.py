from __future__ import division

import torch
from torch.autograd import Variable
from parlai.core.worlds import World


class QAWorld(World):
    def __init__(self, opt, questioner, answerer, data_fetcher, shared=None):
        self.id = 'QAWorld'
        self.qbot = questioner
        self.abot = answerer
        self.data_fetcher = data_fetcher
        self.acts = []
        self.reward = torch.Tensor(self.opt['batch_size'], 1)
        self.cumulative_reward = 0
        super(QAWorld, self).__init__(opt, [self.qbot, self.abot], shared)

    def parley(self):
        batch_img, batch_task, batch_labels = self.data_fetcher.get_batch(self.opt['batch_size'])
        batch_img, batch_task = Variable(batch_img), Variable(batch_task)

        self.qbot.reset(self.opt['batch_size'])
        self.abot.reset(self.opt['batch_size'])

        # get image representation
        img_embed = self.abot.embed_image(batch_img)

        # answer first question, which was from the dataset
        abot_ans = batch_task + self.qbot.task_offset

        # ask multiple rounds of questions and record conversation
        self.acts = []
        for _ in self.opt['num_rounds']:
            # observe answer, ask q_r and observe q_r as well
            self.qbot.observe(abot_ans)
            qbot_ques = self.qbot.act()

            # clone and randomize a bit
            qbot_ques = qbot_ques.detach()
            self.qbot.observe(self.qbot.listen_offset + qbot_ques)

            # forget answer if abot is memory-less
            if not self.opt['remember']:
                self.abot.reset(self.opt['batch_size'], retain_actions=True)

            # observe question and image, also observe answer
            self.abot.observe(qbot_ques, img_embed)
            abot_ans = self.abot.act()

            # clone and randomize a bit
            abot_ans = abot_ans.detach()
            self.abot.observe(abot_ans + self.abot.listen_offset, img_embed)

            self.acts.extend([qbot_ques, abot_ans])

        # observe last answer
        self.qbot.observe(abot_ans)

        # predict image attributes, compute reward
        guess_token, guess_distr = self.qbot.predict(batch_task, 2)

        # compute reward
        self.reward.fill_(self.opt['rl_negative_reward'])

        # both attributes need to match
        first_match = guess_token[0].data == batch_labels[:, 0:1]
        second_match = guess_token[1].data == batch_labels[:, 1:2]
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

