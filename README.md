Language Emergence in Multi Agent Dialog
========================================

Code for the Paper

**[Natural Language Does Not Emerge 'Naturally' in Multi-Agent Dialog][1]**
Satwik Kottur, José M. F. Moura, Stefan Lee, Dhruv Batra
[EMNLP 2017 (Best Short Paper)](https://www.facebook.com/emnlp2017/posts/1935449656693734)

If you find this code useful, please consider citing the original work by authors:

```
@inproceedings{visdial,
  title = {{N}atural {L}anguage {D}oes {N}ot {E}merge '{N}aturally' in {M}ulti-{A}gent {D}ialog},
  author = {Satwik Kottur and Jos\'e M.F. Moura and Stefan Lee and Dhruv Batra},
  journal = {CoRR},
  volume = {abs/1706.08502},
  year = {2017}
}
```

Introduction
------------

This paper focuses on proving that the emergence of language by agent-dialogs is not necessarily
compositional and human interpretable. To demonstrate this fact, the paper uses a Image Guessing
Game "Task and Talk" as a testbed. The game comprises of two bots, a questioner and answerer.

Answerer has an image attributes, as shown in figure. Questioner cannot see the image, and has
a task of finding two attributes of the image (color, shape, style). Answerer does not know the
task. Multiple rounds of q/a dialogs occur, after which the questioner has to guess the
attributes. Reward to both bots is given on basis of prediction of questioner.

![Task And Talk](https://i.imgur.com/9YCM3Gp.png)

Further, the paper discusses the ways to make the grounded language more compositional and human
 interpretable by restrictions on how two agents may communicate.


Setup
-----

This repository is only compatible with Python3, as ParlAI imposes this restriction; it requires
Python3.

1. Follow instructions under **Installing ParlAI** section from [ParlAI site][6].
2. Follow instructions outlined on [PyTorch Homepage][7] for installing PyTorch (Python3).
3. [tqdm][8] is used for providing progress bars, which can be downloaded via pip3.


Dataset Generation
------------------

Described in Section 2 and Figure 1 of paper. Synthetic dataset of shape attributes is generated
using **`data/generate_data.py`** script. To generate the dataset, simply execute:

```sh
cd data
python3 generate_data.py
cd ..
```

This will create **`data/synthetic_dataset.json`**, with 80% training data (312 samples) and rest
validation data (72 samples). Save path, size of dataset and split ratio can be changed through
command line. For more information:

```sh
python3 generate_data.py --help
```

#### Dataset Schema


```json
{
    "attributes": ["color", "shape", "style"],
    "properties": {
        "color": ["red", "green", "blue", "purple"],
        "shape": ["square", "triangle", "circle", "star"],
        "style": ["dotted", "solid", "filled", "dashed"]
    },
    "split_data": {
        "train": [ ["red", "square", "solid"], ["color2", "shape2", "style2"] ],
        "val": [ ["green", "star", "dashed"], ["color2", "shape2", "style2"] ]
    },
    "task_defn": [ [0, 1], [1, 0], [0, 2], [2, 0], [1, 2], [2, 1] ]
}
```

A custom Pytorch Dataset class is written in **`dataloader.py`** which ingests this dataset and
provides random batch / complete data while training and validation.


Training
--------

Training happens through **`train.py`**, which iteratively carries out multiple rounds of dialog
in each episode, between our ParlAI Agents - **QBot** and **ABot**, both placed in a ParlAI
World. The dialog is completely cooperative - both bots receive same reward after each episode.

This script prints the cumulative reward, training accuracy and validation accuracy after fixed
number of iterations. World checkpoints are saved after regular intervals as well.

Training is controlled by various options, which can be passed through command line. All of them
have suitable default values set in **`options.py`**, although they can be tinkered easily.
They can also be viewed as:

```sh
python3 train.py --help   # view command line args (you need not change "Main ParlAI Arguments")
```

Questioner and Answerer bot classes are defined in **`bots.py`** and World is defined in
**`world.py`**. Paper describes three configurations for training:

#### Overcomplete Vocabulary

Described in Section 4.1 of paper. Both QBot and Abot will have vocabulary size equal to
number of possible objects (64).

```sh
python3 train.py --data-path /path/to/json --q-out-vocab 64 --a-out-vocab 64
```

#### Attribute-Value Vocabulary

Described in Section 4.2 of paper. Both QBot will have vocab size 3 (color, shape, style) and
Abot will have vocabulary size equal to number of possible attribute values (4 * 3).

```sh
python3 train.py --data-path /path/to/json --q-out-vocab 3 --a-out-vocab 12
```

#### Memoryless ABot, Minimal Vocabulary (best)

Described in Section 4.3 of paper. Both QBot will have vocab size 3 (color, shape, style) and
Abot will have vocabulary size equal to number of possible values per attribute (4).

```sh
python3 train.py --q-out-vocab 3 --a-out-vocab 4 --data-path /path/to/json --memoryless-abot
```

Checkpoints would be saved by default in **`checkpoints`** directory every 100 epochs. Be default,
 CPU is used for training. Include `--use-gpu` in command-line to train using GPU.

Refer script docstring and inline comments in **`train.py`** for understanding of execution.


Evaluation
----------

Saved world checkpoints can be evaluated using the **`evaluate.py`** script. Besides evaluation,
the dialog between QBot and ABot for all examples can be saved in JSON format. For evaluation:

```sh
python3 evaluate.py --load-path /path/to/pth/checkpoint
```

Save the conversation of bots by providing `--save-conv-path` argument. For more information:
```sh
python3 evaluate.py --help
```

Evaluation script reports training and validation accuracies of the world. Separate accuracies
for first attribute match, second attribute match, both match and atleast one match are reported.

#### Sample Conversation

```text
Im: ['purple', 'triangle', 'filled'] -  Task: ['shape', 'color']
    Q1: X    A1: 2
    Q2: Y    A2: 0
    GT: ['triangle', 'purple']  Pred: ['triangle', 'purple']
```


Pretrained World Checkpoint
---------------------------

Best performing world checkpoint has been released [here](https://github.com/karandesai-96/lang-emerge-parlai/releases/tag/v1.0), along with details to reconstruct the world object using
this checkpoint.

**Reported metrics:**
```text
Overall accuracy [train]: 96.47 (first: 97.76, second: 98.72, atleast_one: 100.00)
Overall accuracy [val]: 98.61 (first: 98.61, second: 100.00, atleast_one: 100.00)
```

**TODO: Visualizing evolution chart - showing emergence of grounded language.**

References
----------

1. **Sattvik Kottur, José M.F.Moura, Stefan Lee, Dhruv Batra.** _Natural Language Does Not Emerge
   Naturally in Multi-Agent Dialog._ EMNLP 2017. [[arxiv][1]]
1. **Alexander H. Miller, Will Feng, Adam Fisch, Jiasen Lu, Dhruv Batra, Antoine Bordes, Devi
   Parikh, Jason Weston.** _ParlAI: A Dialog Research Software Platform._ 2017. [[arxiv][2]]
1. **Abhishek Das, Satwik Kottur, Khushi Gupta, Avi Singh, Deshraj Yadav, José M.F. Moura, Devi
   Parikh and Dhruv Batra.** _Visual Dialog._ CVPR 2017. [[arxiv][3]]
1. **Abhishek Das, Satwik Kottur, José M.F. Moura, Stefan Lee, and Dhruv Batra.** _Learning
   Cooperative Visual Dialog Agents with Deep Reinforcement Learning._ ICCV 2017. [[arxiv][4]]
1. ParlAI Docs. [[http://parl.ai/static/docs/index.html](http://parl.ai/static/docs/index.html)]
1. PyTorch Docs. [[http://pytorch.org/docs/master](http://pytorch.org/docs/master)]


Standing on the Shoulders of Giants
-----------------------------------

The ease of implementing this paper using ParlAI framework is heavy accredited to the original
source code released by authors of this paper. [[batra-mlp-lab/lang-emerge][5]]

License
-------

BSD

[1]: https://arxiv.org/abs/1706.08502
[2]: https://arxiv.org/abs/1705.06476
[3]: https://arxiv.org/abs/1611.08669
[4]: https://arXiv.org/abs/1703.06585
[5]: https://www.github.com/batra-mlp-lab/lang-emerge
[6]: https://parl.ai/about
[7]: https://pytorch.org
[8]: https://pypi.python.org/pypi/tqdm
