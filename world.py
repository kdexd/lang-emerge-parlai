"""Module with class for world. This world has been extended from ``DialogPartnerWorld``, provided
 by ParlAI, which involves alternating dialog between two agents in the world.

Refer ParlAI docs on general semantics of a ParlAI World:
    * http://parl.ai/static/docs/basic_tutorial.html#worlds
    * http://parl.ai/static/docs/agents.html#parlai.core.worlds.DialogPartnerWorld
"""
import torch
from parlai.core.worlds import DialogPartnerWorld


class QAWorld(DialogPartnerWorld):
    """Custom Parlai world with two agents - questioner and answerer bots, having multi round
    dialog. As per semantics, the overidden method ``world.parley()`` hold one round of question
    and answer exchange of dialog between both agents.

    Attributes
    ----------
    opt : dict
        All options needed to set up the world and its agents. These are the command line
        arguments defined in ``options.py``.
    acts : list
        List of actions (dialogs) by the agents of this world. List contains dicts with keys
        'id' and 'text' - which agent spkoe what.
    qbot : Questioner
    abot : Answerer
    """
    def __init__(self, opt, questioner, answerer, shared=None):
        self.id = 'QAWorld'
        self.qbot = questioner
        self.abot = answerer
        self.acts = []
        self.episode_batch = None    # episode specific batch
        self.cumulative_reward = 0
        super(QAWorld, self).__init__(opt, [self.qbot, self.abot], shared)

    def parley(self):
        """Conduct one round of dialog. QBot asks question and observes it later. ABot answers the
        question and observes it later too. Dialog between QBot and ABot is totally cooperative,
        hence both can observe their own questions and answers respectively, and both receive the
        same reward later as well.
        """
        if self.qbot.observation.get('episode_done'):
            self.episode_batch = self.qbot.observation['batch']
            self.qbot.reset(batch_size=self.episode_batch['task'].size(0), retain_actions=False)
            self.abot.reset(batch_size=self.episode_batch['task'].size(0), retain_actions=False)

            # get task embedding and image representation
            self.episode_batch['image_embed'] = self.abot.embed_image(self.episode_batch['image'])
            # ask multiple rounds of questions and record conversation
            self.acts = []

            # qbot start with task embedding
            self.qbot.observe({
                'text': self.episode_batch['task'] + self.qbot.task_offset,
                'id': self.id
            })

        # qbot ask a question and observe it as well
        qbot_ques = self.qbot.act()
        qbot_ques['text'] = qbot_ques['text'].detach()
        self.qbot.observe({
            'text': qbot_ques['text'] + self.qbot.listen_offset,
            'id': self.qbot.id
        })

        # forget answer if abot is memory-less
        if self.opt['memoryless_abot']:
            self.abot.reset(batch_size=self.episode_batch['task'].size(0), retain_actions=True)

        # observe question and image embedding, also observe answer
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
        self.qbot.observe(abot_ans)
        self.acts.extend([qbot_ques, abot_ans])

    def save_agents(self, save_path):
        """Save complete world with all the agents, saves everything required to reload later."""
        qbot_state_dict = self.qbot.state_dict()
        abot_state_dict = self.abot.state_dict()
        torch.save({'qbot': qbot_state_dict, 'abot': abot_state_dict, 'opt': self.opt}, save_path)
