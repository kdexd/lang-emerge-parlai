from parlai.core.params import ParlaiParser


def read():
    parser = ParlaiParser()
    parser.add_argument_group('Dataset Parameters')
    parser.add_argument('--data-path', default='data/synthetic_dataset.json', type=str,
                        help='Path to the training/val dataset file')
    parser.add_argument('--neg-fraction', default=0.8, type=float,
                        help='Fraction of negative examples in batch')

    parser.add_argument_group('Model Parameters')
    parser.add_argument('--hidden-size', default=100, type=int,
                        help='Hidden Size for the language models')
    parser.add_argument('--embed-size', default=20, type=int,
                        help='Embed size for words')
    parser.add_argument('--img-feat-size', default=20, type=int,
                        help='Image feature size for each attribute')
    parser.add_argument('--q-out-vocab', default=3, type=int,
                        help='Output vocabulary for questioner')
    parser.add_argument('--a-out-vocab', default=4, type=int,
                        help='Output vocabulary for answerer')

    parser.add_argument('--rl-scale', default=100.0, type=float,
                        help='Weight given to rl gradients')
    parser.add_argument('--num-rounds', default=2, type=int,
                        help='Number of rounds between Q and A')
    parser.add_argument('--remember', dest='remember', action='store_true',
                        help='Turn on/off for ABot with memory')

    parser.add_argument_group('Optimization Hyperparameters')
    parser.add_argument('--batch-size', default=1000, type=int,
                        help='Batch size during training')
    parser.add_argument('--num-epochs', default=10000, type=int,
                        help='Max number of epochs to run')
    parser.add_argument('--learning-rate', default=1e-3, type=float,
                        help='Initial learning rate')
    parser.add_argument('--save-epoch', default=100, type=int,
                        help='Save model at regular intervals of epochs.')
    parser.add_argument('--save-path', default='checkpoints', type=str,
                        help='Directory path to save checkpoints.')
    parser.add_argument('--use-gpu', dest='use_gpu', action='store_true')
    return parser.parse_args()
