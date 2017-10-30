from __future__ import print_function
from functools import reduce
import itertools
import json
import os
import random

import torch
from torch.utils.data import Dataset, DataLoader

from parlai.core.agents import Agent


class ShapesQADataset(Dataset):

    def __init__(self, data_path, q_out_vocab, a_out_vocab, train_split=None):
        self.path = data_path
        self.q_out_vocab = q_out_vocab
        self.a_out_vocab = a_out_vocab
        self.train_split = train_split

        if os.path.exists(self.path):
            # load dataset from file
            with open(self.path, 'r') as infile:
                loaded = json.load(infile)
                for key, value in loaded.items():
                    # inject props, task_defn, split_data into ``self``
                    setattr(self, key, value)
        else:
            # create dataset if not loaded, set default train_split if not set yet
            if not self.train_split:
                self.train_split = 0.8

            self.props = {
                'colors': ['red', 'green', 'blue', 'purple'],
                'shape': ['square', 'triangle', 'circle', 'star'],
                'style': ['dotted', 'solid', 'filled', 'dashed']
            }
            data_verbose = list(itertools.product(*self.props.values()))

            # randomly select train and rest of it is test
            self.split_data = {}
            self.split_data['train'] = random.sample(data_verbose,
                                                     int(self.train_split * len(data_verbose)))
            self.split_data['test'] = list(set(data_verbose) - set(self.split_data['train']))

            self.task_defn = [[0, 1], [1, 0], [0, 2],
                              [2, 0], [1, 2], [2, 1],
                              [0, 0], [1, 1], [2, 2]]

            to_save = {
                'props': self.props,
                'task_defn': self.task_defn,
                'split_data': self.split_data
            }

            with open(self.path, 'w') as outfile:
                json.dump(to_save, outfile, indent=4, separators=(',', ': '), sort_keys=True)

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

        # Separate data loading for test/train
        self.data = {}
        for dtype in ['train', 'test']:
            ddata = torch.LongTensor(len(self.split_data[dtype]), len(self.props))
            for index, attr_set in enumerate(self.split_data[dtype]):
                ddata[index] = torch.LongTensor([inv_vocab_attr[attr] for attr in attr_set])
            self.data[dtype] = ddata
        del self.split_data

        self.range_inds = torch.arange(0, len(self.data['train'])).long()

    def __len__(self):
        return len(self.data['train'])

    def __getitem__(self, index):
        task = torch.Tensor([random.randint(0, self.num_pair_tasks - 1)]).long()
        example = self.data['train'][index]

        # now sample predictions based on task
        select_index = torch.Tensor(self.task_defn[task[0]]).long()
        labels = example.gather(0, select_index)

        return {'image': example, 'task': task, 'labels': labels}


class DataLoaderAgent(Agent):

    @staticmethod
    def add_cmdline_args(argparser):
        dictionary = argparser.add_argument_group('Dataset Arguments')
        dictionary.add_argument('--data-path', help='path of dataset file to save/load dataset')
        dictionary.add_argument('--train-split', type=float, default=None,
                                help='fraction of examples to be used for training (0 to 1)')
        return dictionary

    def __init__(self, opt, shared=None):
        super(DataLoaderAgent, self).__init__(opt, shared)
        self.id = 'DataLoaderAgent'
        self.dataset = ShapesQADataset(
            opt['data_path'], opt['q_out_vocab'], opt['a_out_vocab'])
        self.dataloader = DataLoader(self.dataset, shuffle=True, batch_size=opt['batchsize'])
        self.iter_dataloader = itertools.cycle(self.dataloader)

    def act(self):
        batch = next(self.iter_dataloader)
        if self.opt['use_gpu']:
            for key in batch:
                batch[key] = batch[key].cuda()
        return batch

    def observe(self, observation=None):
        pass

    def reformat_talk(self, talk, preds, images, tasks, labels):
        """Convert to text."""
        script = []
        if self.dataset.q_out_vocab < 4:
            a_vocab = [str(ii) for ii in xrange(self.dataset.a_out_vocab)]
            q_vocab = [chr(ii + 88) for ii in xrange(self.dataset.q_out_vocab)]
        else:
            a_vocab = ['a-%d' % ii for ii in xrange(self.dataset.a_out_vocab)]
            q_vocab = ['q-%d' % ii for ii in xrange(self.dataset.q_out_vocab)]

        attr_name_task_defn = {0: 'color', 1: 'shape', 2: 'style'}
        for i in xrange(images.size(0)):
            # conversation
            conv = {}
            conv['image'] = [self.dataset.vocab_attr[j] for j in images[i]]
            conv['gt'] = [self.dataset.vocab_attr[labels[i, j]] for j in xrange(2)]
            conv['task'] = [attr_name_task_defn[j] for j in self.dataset.task_defn[tasks[i]]]
            conv['pred'] = [self.dataset.vocab_attr[preds[j].data[i, 0]]
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


if __name__ == '__main__':
    a = DataLoaderAgent({'data_path': 'data/toy64_split_0.8.json', 'q_out_vocab': 3, 'a_out_vocab': 4, 'batchsize': 2})
    a.act()

