from __future__ import division

import torch
from torch.autograd import Variable
from parlai.core.worlds import DialogPartnerWorld


class QAWorld(DialogPartnerWorld):
    def __init__(self, opt, questioner, answerer, shared=None):
        self.id = 'QAWorld'
        self.qbot = questioner
        self.abot = answerer
        self.acts = []
        self.episode_batch = None    # episode specific batch
        self.cumulative_reward = 0
        super(QAWorld, self).__init__(opt, [self.qbot, self.abot], shared)

    def parley(self):
        if self.qbot.observation.get('episode_done'):
            self.episode_batch = self.qbot.observation['batch']
            self.qbot.reset(retain_actions=False)
            self.abot.reset(retain_actions=False)

            # get image representation
            self.episode_batch['image_embed'] = self.abot.embed_image(self.episode_batch['image'])

            # ask multiple rounds of questions and record conversation
            self.acts = []

            # answer first question, which was from the dataset
            abot_ans = {
                'text': self.episode_batch['task'] + self.qbot.task_offset,
                'id': self.abot.id
            }

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
            'image': self.episode_batch['image_embed']
        })
        abot_ans = self.abot.act()

        # clone and randomize a bit
        abot_ans['text'] = abot_ans['text'].detach()
        self.abot.observe({
            'text': abot_ans['text'] + self.abot.listen_offset,
            'id': self.abot.id,
            'image': self.episode_batch['image_embed']
        })

        self.acts.extend([qbot_ques, abot_ans])
        self.qbot.observe(abot_ans)

