from __future__ import print_function
import itertools
import json
import os
import random

import torch
from torch.utils.data import Dataset


class ShapesQADataset(Dataset):
    def __init__(self, data_path, train_split, q_out_vocab, a_out_vocab):
        self.path = data_path
        self.train_split = train_split
        self.q_out_vocab = q_out_vocab
        self.a_out_vocab = a_out_vocab

        if os.path.exists(self.path):
            # load dataset from file
            with open(self.path, 'r') as infile:
                loaded = json.load(infile)
                for key, value in loaded.iteritems():
                    # inject props, task_defn, split_data into ``self``
                    setattr(self, key, value)
        else:
            # create dataset if not loaded
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
        return len(self.data['train']) + len(self.data['test'])

    def __getitem__(self, index):
        task = torch.Tensor([random.randint(0, self.num_pair_tasks - 1)]).long()
        example = self.data['train'][index]

        # now sample predictions based on task
        select_index = torch.Tensor(self.task_defn[task[0]]).long()
        labels = example.gather(0, select_index)

        return {'example': example, 'task': task, 'labels': labels}

    # converting to text
    def reformat_talk(self, talk, preds, images, tasks, labels):
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
            conv['task'] = [attr_name_task_defn[j] for j in self.task_defn[tasks[i]]]
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
    dataset = ShapesQADataset('data/toy64_split_0.8.json', 0.8, 3, 4)
    print(random.choice(dataset))
