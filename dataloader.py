from __future__ import print_function
from functools import reduce
import itertools
import json
import random

import torch
from torch.utils.data import Dataset, DataLoader

from parlai.core.agents import Agent


class ShapesQADataset(Dataset):
    def __init__(self, opt, type='train'):
        self.q_out_vocab = opt['q_out_vocab']
        self.a_out_vocab = opt['a_out_vocab']

        # load dataset from file
        with open(opt['data_path'], 'r') as infile:
            loaded = json.load(infile)
            self.props = loaded['props']
            self.task_select = torch.LongTensor(loaded['task_defn'])
            self.data = loaded['split_data'][type]

        # number of single and pair wise tasks
        self.num_pair_tasks = 6
        self.num_single_tasks = 3
        # create a vocab map for field values (associate each attribute value with a number)
        attr_vals = reduce(lambda x, y: x + y, self.props.values())
        self.vocab_attr = {index: value for index, value in enumerate(attr_vals)}
        inv_vocab_attr = {value: index for index, value in self.vocab_attr.items()}

        # get encoding for attribute pairs
        attr_pair = itertools.product(attr_vals, repeat=2)
        self.vocab_attr_pair = {index: value for index, value in enumerate(attr_pair)}

        ddata = torch.LongTensor(len(self.data), len(self.props))
        for index, attr_set in enumerate(self.data):
            ddata[index] = torch.LongTensor([inv_vocab_attr[attr] for attr in attr_set])
        self.data = ddata

        self.range_indices = torch.arange(0, len(self.data)).long()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        task = torch.Tensor([random.randint(0, self.num_pair_tasks - 1)]).long()
        example = self.data[index]

        # now sample predictions based on task
        select_index = self.task_select[task[0]].long()
        labels = example.gather(0, select_index)

        return {'image': example, 'task': task, 'labels': labels}

    def get_batch(self, batch_size, current_pred=None, neg_fraction=0.8):
        """Get a batch."""
        tasks = torch.LongTensor(batch_size).random_(0, self.num_pair_tasks - 1)
        indices = torch.LongTensor(batch_size).random_(0, len(self.data) - 1)

        if current_pred is not None:
            # fill the first batch_size / 2 based on previously misclassified examples
            neg_indices = current_pred.view(-1, self.num_pair_tasks).sum(1) < self.num_pair_tasks
            neg_indices = self.range_indices.masked_select(neg_indices)
            neg_batch_size = int(batch_size * neg_fraction)
            # sample from this
            neg_samples = torch.LongTensor(neg_batch_size).fill_(0)
            if neg_indices.size(0) > 1:
                neg_samples.random_(0, neg_indices.size(0) - 1)
            neg_indices = neg_indices[neg_samples]
            indices[:neg_batch_size] = neg_indices
        images = self.data[indices]

        # now sample predictions based on task
        select_indices = self.task_select[tasks]
        labels = images.gather(1, select_indices)

        return {'image': images, 'task': tasks, 'labels': labels}

    def get_complete_data(self):
        """Get all configurations."""
        # expand self.data three folds, along with labels
        images = self.data.unsqueeze(0).repeat(1, 1, self.num_pair_tasks)
        images = images.view(-1, len(self.props))
        tasks = torch.arange(0, self.num_pair_tasks).long()
        tasks = tasks.unsqueeze(0).repeat(1, len(self.data)).view(-1)

        # now sample predictions based on task
        select_indices = self.task_select[tasks]
        labels = images.gather(1, select_indices)

        return {'image': images, 'task': tasks, 'labels': labels}

    def reformat_talk(self, talk, preds, images, tasks, labels):
        """Convert to text."""
        script = []
        if self.q_out_vocab < 4:
            a_vocab = [str(ii) for ii in xrange(self.a_out_vocab)]
            q_vocab = [chr(ii + 88) for ii in xrange(self.q_out_vocab)]
        else:
            a_vocab = ['a-%d' % ii for ii in xrange(self.a_out_vocab)]
            q_vocab = ['q-%d' % ii for ii in xrange(self.q_out_vocab)]

        attr_name_task_defn = {0: 'color', 1: 'shape', 2: 'style'}
        for i in xrange(images.size(0)):
            # conversation
            conv = {}
            conv['image'] = [self.vocab_attr[j] for j in images[i]]
            conv['gt'] = [self.vocab_attr[labels[i, j]] for j in xrange(2)]
            conv['task'] = [attr_name_task_defn[j] for j in self.task_select[tasks[i]]]
            conv['pred'] = [self.vocab_attr[preds[j].data[i, 0]]
                            for j in xrange(2)]
            conv['chat'] = [q_vocab[talk[0].data[i]],
                            a_vocab[talk[1].data[i]]]
            if len(talk) > 3:
                conv['chat'].extend([q_vocab[talk[2].data[i]],
                                     a_vocab[talk[3].data[i]]])
            script.append(conv)

        # re-arrange such that negative examples are on the top
        wrong_ex = []
        for i in script:
            if i['gt'] != i['pred']:
                wrong_ex.append(i)

        # remove wrong Ex from script
        for ex in wrong_ex:
            script.remove(ex)
        script = wrong_ex + script
        return script

    @staticmethod
    def pretty_print(talk):
        """Pretty print result."""
        for conv in talk:
            # first print image, task
            print('Im: %s -  Task: %s' % (conv['image'], conv['task']))
            # print conversation
            print('\tQ1 : %s \t A1: %s' % (conv['chat'][0], conv['chat'][1]))
            print('\tQ2 : %s \t A2: %s' % (conv['chat'][2], conv['chat'][3]))
            # print GT and prediction
            print('\tGT: %s\tPred: %s' % (conv['gt'], conv['pred']))
            print('--------------------\n')
