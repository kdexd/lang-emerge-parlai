from __future__ import print_function
import itertools
import json
import random

import torch
from torch.utils.data import Dataset


class ShapesQADataset(Dataset):
    def __init__(self, opt):
        self.opt = opt
        if 'load_path' in opt:
            # load dataset from file
            with open(opt['load_path'], 'r') as file_id:
                loaded = json.load(file_id)
                for key, value in loaded.iteritems():
                    setattr(self, key, value)
        else:
            # create dataset if not loaded
            self.props = {
                'colors': ['red', 'green', 'blue', 'purple'],
                'shape': ['square', 'triangle', 'circle', 'star'],
                'style': ['dotted', 'solid', 'filled', 'dashed']
            }
            data_verbose = list(itertools.product(*self.props.values()))

            # select train_size for train
            num_inst = len(data_verbose)
            num_inst_train = int(opt['train_split'] * num_inst)
            num_inst_test = num_inst - num_inst_train

            # randomly select test
            self.split_data = {}
            self.split_data['test'] = random.sample(data_verbose, num_inst_test)
            self.split_data['train'] = list(set(data_verbose) - set(self.split_data['test']))

            self.task_defn = [[0, 1], [1, 0], [0, 2],
                              [2, 0], [1, 2], [2, 1],
                              [0, 0], [1, 1], [2, 2]]

            to_save = {
                'props': self.props,
                'task_defn': self.task_defn,
                'split': self.split_data
            }

            if opt['save_path']:
                with open(opt['save_path'], 'w') as outfile:
                    json.dump(to_save, outfile, indent=4, separators=(',', ': '), sort_keys=True)

        self.attr_val_vocab = reduce(lambda x, y: x + y,
                                     [self.props[attr] for attr in self.props])
        self.num_tasks = len(self.task_defn)

        self.task_select = torch.LongTensor(self.task_defn)

        # number of single and pair wise tasks
        self.num_pair_tasks = 6
        self.num_single_tasks = 3

        # create a vocab map for field values
        attr_vals = reduce(lambda x, y: x + y,
                           [self.props[attr] for attr in self.props])
        self.attr_vocab = {value: ii for ii, value in enumerate(attr_vals)}
        self.inv_attr_vocab = {index: attr for attr, index in self.attr_vocab.items()}

        # get encoding for attribute pairs
        self.attr_pair = itertools.product(attr_vals, repeat=2)
        self.attr_pair_vocab = {value: ii for ii, value in enumerate(self.attr_pair)}
        self.inv_attr_pair_vocab = {index: value for value, index
                                    in self.attr_pair_vocab.items()}

        # Separate data loading for test/train
        self.data = {}
        for dtype in ['train', 'test']:
            ddata = torch.LongTensor(len(self.split_data[dtype]), len(self.props))
            for ii, attr_set in enumerate(self.split_data[dtype]):
                ddata[ii] = torch.LongTensor([self.attr_vocab[at] for at in attr_set])
            self.data[dtype] = ddata

        self.range_inds = torch.arange(0, len(self.split_data['train'])).long()
        # ship to gpu if needed
        if opt['use_gpu']:
            for key, value in self.data.iteritems():
                self.data[key] = value.cuda()
            self.range_inds = self.range_inds.cuda()

    def __len__(self):
        return len(self.data['train']) + len(self.data['test'])

    # get a batch
    def __getitem__(self, index):
        task = random.nextint(0, self.num_pair_tasks - 1)
        example = self.data['train'][index]

        # now sample predictions based on task
        select_index = self.task_select[task]
        labels = example.gather(1, select_index)

        return example, task, labels

    # converting to text
    def reformat_talk(self, talk, preds, images, tasks, labels):
        script = []
        if self.opt['q_out_vocab'] < 4:
            a_vocab = [str(ii) for ii in xrange(self.opt['a_out_vocab'])]
            q_vocab = [chr(ii + 88) for ii in xrange(self.opt['q_out_vocab'])]
        else:
            a_vocab = ['a-%d' % ii for ii in xrange(self.opt['a_out_vocab'])]
            q_vocab = ['q-%d' % ii for ii in xrange(self.opt['q_out_vocab'])]

        attr_name_task_defn = {0: 'color', 1: 'shape', 2: 'style'}
        for ii in xrange(images.size(0)):
            # conversation
            conv = {}
            conv['image'] = [self.inv_attr_vocab[jj] for jj in images[ii]]
            conv['gt'] = [self.inv_attr_vocab[labels[ii, jj]] for jj in xrange(2)]
            conv['task'] = [attr_name_task_defn[jj] for jj in self.task_defn[tasks[ii]]]
            conv['pred'] = [self.inv_attr_vocab[preds[jj].data[ii, 0]]
                            for jj in xrange(2)]
            conv['chat'] = [q_vocab[talk[0].data[ii]],
                            a_vocab[talk[1].data[ii]]]
            if len(talk) > 3:
                conv['chat'].extend([q_vocab[talk[2].data[ii]],
                                     a_vocab[talk[3].data[ii]]])
            script.append(conv)

        # re-arrange such that negative examples are on the top
        wrong_ex = []
        for ii in script:
            if ii['gt'] != ii['pred']:
                wrong_ex.append(ii)

        # remove wrong Ex from script
        for ex in wrong_ex:
            script.remove(ex)
        script = wrong_ex + script
        return script

    def pretty_print(self, talk):
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
    options = {
        'save_path': 'data/toy64_split_0.8.json',
        'train_split': 0.8,
        'q_out_vocab': 3,
        'a_out_vocab': 4,
        'use_gpu': True
    }
    dataset = ShapesQADataset(options)
