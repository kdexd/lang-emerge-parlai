from __future__ import print_function
from functools import reduce
import json
import random

import torch
from torch.utils.data import Dataset, DataLoader

from parlai.core.agents import Agent


class ShapesQADataset(Dataset):
    """Custom dataset to represent synthetic training examples (shape,
    color, style). This dataset inherits from ``torch.utils.data.Dataset``.
    Apart from list-style indexng, it has features to pull out a batch of
    examples randomly as well as retrieve all the examples.

    Attributes
    ----------
    opt : dict
    attributes : list
    properties : dict
    task_defn : torch.LongTensor or torch.cuda.LongTensor
    data : torch.LongTensor or torch.cuda.LongTensor
    vocab_task : dict
    vocab_attr_val : dict
    """

    def __init__(self, opt):
        self.opt = opt

        # load dataset from file
        with open(opt['data_path'], 'r') as infile:
            loaded = json.load(infile)
            self.attributes = loaded['attributes']
            self.properties = loaded['properties']
            self.task_defn = torch.LongTensor(loaded['task_defn'])
            self.data = loaded['split_data']

        # create a vocab map for field values (associate each attribute value with a number)
        attr_vals = reduce(lambda x, y: x + y, [self.properties[attr] for attr in self.attributes])
        self.vocab_task = {index: value for index, value in enumerate(self.attributes)}
        self.vocab_attr_val = {index: value for index, value in enumerate(attr_vals)}
        inv_vocab_attr_val = {value: index for index, value in self.vocab_attr_val.items()}

        for dtype in ['train', 'val']:
            data_t = torch.LongTensor(len(self.data[dtype]), len(self.properties))
            for index, attr_set in enumerate(self.data[dtype]):
                data_t[index] = torch.LongTensor([inv_vocab_attr_val[attr] for attr in attr_set])
            self.data[dtype] = data_t

    def __len__(self):
        return len(self.data['train'])

    def __getitem__(self, index):
        image = self.data['train'][index]
        task = random.randint(0, len(self.task_defn) - 1)

        # now sample predictions based on task
        select_index = torch.LongTensor(self.task_defn[task])
        labels = image.gather(0, torch.LongTensor(select_index))
        task = torch.LongTensor([task])
        if self.opt.get('use_gpu'):
            image, task, labels = image.cuda(), task.cuda(), labels.cuda()
        return {'image': image, 'task': task, 'labels': labels}

    def random_batch(self, dtype, current_pred=None):
        """Get a batch randomly sampled from data."""
        indices = [random.randint(0, len(self.data[dtype]) - 1)
                   for _ in range(self.opt['batch_size'])]
        indices = torch.LongTensor(indices)

        if current_pred is not None:
            # fill the first batch_size / 2 based on previously misclassified examples
            current_pred = current_pred.cpu()
            neg_indices = current_pred.view(
                -1, len(self.task_defn)).sum(1) < len(self.task_defn)
            neg_indices = torch.arange(0, len(self.data[dtype])).masked_select(neg_indices).long()
            neg_batch_size = int(self.opt['batch_size'] * self.opt['neg_fraction'])
            # sample from this
            if neg_batch_size > 0:
                neg_samples = torch.zeros(neg_batch_size).long()
                if neg_indices.size(0) > 1:
                    neg_samples.random_(0, neg_indices.size(0) - 1)
                neg_indices = neg_indices[neg_samples]
                indices[:neg_batch_size] = neg_indices
        images = self.data[dtype][indices]

        tasks = torch.Tensor([random.randint(0, len(self.task_defn) - 1)
                              for _ in range(self.opt['batch_size'])]).long()
        # now sample predictions based on task
        select_indices = self.task_defn[tasks]
        labels = images.gather(1, select_indices)
        if self.opt.get('use_gpu'):
            images, tasks, labels = images.cuda(), tasks.cuda(), labels.cuda()
        return {'image': images, 'task': tasks, 'labels': labels}

    def complete_data(self, dtype):
        """Get all configurations."""
        # expand self.data three folds, along with labels
        images = self.data[dtype].unsqueeze(0).repeat(1, 1, len(self.task_defn))
        images = images.view(-1, len(self.properties))
        tasks = torch.arange(0, len(self.task_defn)).long()
        tasks = tasks.unsqueeze(0).repeat(1, len(self.data[dtype])).view(-1)

        # now sample predictions based on task
        select_indices = self.task_defn[tasks]
        labels = images.gather(1, select_indices)
        if self.opt.get('use_gpu'):
            images, tasks, labels = images.cuda(), tasks.cuda(), labels.cuda()
        return {'image': images, 'task': tasks, 'labels': labels}

    def talk_to_script(self, talk, preds, batch):
        """COnverts talk of agents to a list of json objects, useful for saving and printing."""
        images, tasks, labels = batch['image'].data, batch['task'].data, batch['labels']
        script = []
        if self.opt['q_out_vocab'] < 4:
            q_vocab = [chr(i + 88) for i in range(self.opt['q_out_vocab'])]  # X, Y, Z
            a_vocab = [str(i) for i in range(self.opt['a_out_vocab'])]       # 1, 2, 3
        else:
            q_vocab = ['Q%d' % i for i in range(self.opt['q_out_vocab'])]    # Q1, Q2, Q3, Q4...
            a_vocab = ['A%d' % i for i in range(self.opt['a_out_vocab'])]    # A1, A2, A3, A4...

        for i in range(images.size(0)):
            # conversation
            conv = {}
            conv['image'] = [self.vocab_attr_val[j] for j in images[i]]
            conv['gt'] = [self.vocab_attr_val[labels[i, j]] for j in range(2)]
            conv['task'] = [self.vocab_task[j] for j in self.task_defn[tasks[i]].squeeze()]
            conv['pred'] = [self.vocab_attr_val[preds[j].data[i]] for j in range(2)]
            conv['chat'] = [q_vocab[talk[0]['text'].data[i]],
                            a_vocab[talk[1]['text'].data[i]]]
            for j in range(2, len(talk), 2):
                conv['chat'].extend([q_vocab[talk[j]['text'].data[i]],
                                     a_vocab[talk[j + 1]['text'].data[i]]])
            script.append(conv)

        # re-arrange such that negative examples are on the top
        wrong_ex = [conv for conv in script if conv['gt'] != conv['pred']]
        for ex in wrong_ex:
            script.remove(ex)
        script = wrong_ex + script
        return script

    @staticmethod
    def pretty_print(script):
        """Pretty print as conversation."""
        for conv in script:
            print('Im: %s -  Task: %s' % (conv['image'], conv['task']))
            print('\tQ1: %s\t A1: %s' % (conv['chat'][0], conv['chat'][1]))
            print('\tQ2: %s\t A2: %s' % (conv['chat'][2], conv['chat'][3]))
            print('\tGT: %s\tPred: %s' % (conv['gt'], conv['pred']))
            print('-' * 59)
