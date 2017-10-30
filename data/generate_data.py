import itertools
import json
import random

from parlai.core.params import ParlaiParser


parser = ParlaiParser()
parser.add_argument('--save-path', type=str, default='synthetic_dataset.json',
                    help='path to save the generated dataset')
parser.add_argument('--size', type=str, default='normal', help='size of dataset: normal|large')
parser.add_argument('--holdout', type=float, default=0.2,
                    help='fraction of data to be held out as validation data')


if __name__ == '__main__':
    opt = parser.parse_args()

    if opt['size'] == 'normal':
        PROPS = {
            'color': ['red', 'green', 'blue', 'purple'],
            'shape': ['square', 'triangle', 'circle', 'star'],
            'style': ['dotted', 'solid', 'filled', 'dashed']
        }
    else:
        PROPS = {
            'color': ['red', 'green', 'blue', 'purple', 'yellow', 'cyan', 'orange', 'teal'],
            'shape': ['square', 'triangle', 'circle', 'star', 'heart', 'pentagon', 'hexagon',
                      'ring'],
            'style': ['dotted', 'solid', 'filled', 'dashed', 'hstripe', 'vstripe', 'hgradient',
                      'vgradient']
        }

    data_verbose = list(itertools.product(*PROPS.values()))

    # randomly select train and rest of it is test
    SPLIT_DATA = {}
    SPLIT_DATA['val'] = random.sample(data_verbose, int(opt['holdout'] * len(data_verbose)))
    SPLIT_DATA['train'] = [sample for sample in data_verbose if sample not in SPLIT_DATA['val']]

    TASK_DEFN = [[0, 1], [1, 0], [0, 2],
                 [2, 0], [1, 2], [2, 1],
                 [0, 0], [1, 1], [2, 2]]

    TO_SAVE = {
        'props': PROPS,
        'task_defn': TASK_DEFN,
        'split_data': SPLIT_DATA
    }

    with open(opt['save_path'], 'w') as outfile:
        json.dump(TO_SAVE, outfile, indent=4, separators=(',', ': '), sort_keys=True)
