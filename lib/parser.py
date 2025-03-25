import argparse


def args_parser():
    parser = argparse.ArgumentParser()
    
    # miscellaneous args
    parser.add_argument('--gpu', type=int, default=0,
                        help='GPU (default=%(default)s)')
    parser.add_argument('--results-dir', type=str, default='./results',
                        help='Results dir (default=%(default)s)')
    parser.add_argument('--exp-name', default='debug', type=str,
                        help='Experiment name (default=%(default)s) [ablation, debug]')
    parser.add_argument('--seed', type=int, default=0,
                        help='Random seed (default=%(default)s)')
    parser.add_argument('--save-models', action='store_true',
                        help='Save trained models (default=%(default)s)')
    parser.add_argument('--no-cudnn-deterministic', action='store_true',
                        help='Disable CUDNN deterministic (default=%(default)s)')

    # dataset args
    parser.add_argument('--data-dir', default='~/data', type=str)
    parser.add_argument('--dataset', default='CIFAR100', type=str, help="TinyImageNet|CIFAR100")
    parser.add_argument('--num-workers', default=0, type=int, required=False,
                        help='Number of subprocesses to use for dataloader (default=%(default)s)')
    parser.add_argument('--pin-memory', default=0, type=int, required=False,
                        help='Copy Tensors into CUDA pinned memory before returning them (default=%(default)s)')
    parser.add_argument('--batch-size', default=32, type=int, required=False,
                        help='Number of samples per batch to load (default=%(default)s)')
    parser.add_argument('--num-tasks', default=10, type=int, required=False,
                        help='Number of tasks per dataset (default=%(default)s)')
    parser.add_argument('--min-class', default=1.0, type=float, required=False,
                        help='min class percentage for each task')
    parser.add_argument('--max-class', default=1.0, type=float, required=False,
                        help='max class percentage for each task')
    parser.add_argument('--client-no-intersection', default=0, type=int, required=False)
    parser.add_argument('--nc-first-task', default=None, type=int, required=False,
                        help='Number of classes of the first task (default=%(default)s)')
    parser.add_argument('--use-valid-only', action='store_true',
                        help='Use validation split instead of test (default=%(default)s)')
    parser.add_argument('--stop-at-task', default=-1, type=int, required=False,
                        help='Stop training after specified task (default=%(default)s)')

    # model args
    parser.add_argument('--network', default='resnet18', type=str,
                        help='Network architecture used (default=%(default)s)', metavar="NETWORK")
    parser.add_argument('--keep-existing-head', action='store_true',
                        help='Disable removing classifier last layer (default=%(default)s)')
    parser.add_argument('--pretrained', action='store_true',
                        help='Use pretrained backbone (default=%(default)s)')
    
    # training args
    parser.add_argument('--approach', default='lwf_fedavg', type=str,
                        help='Learning approach used (default=%(default)s)', metavar="APPROACH")
    parser.add_argument('--nepochs', default=1, type=int, required=False,
                        help='Number of epochs per training session (default=%(default)s)')
    parser.add_argument('--lr', default=0.01, type=float, required=False,
                        help='Starting learning rate (default=%(default)s)')
    parser.add_argument('--clipping', default=10000, type=float, required=False,
                        help='Clip gradient norm (default=%(default)s)')
    parser.add_argument('--momentum', default=0.0, type=float, required=False,
                        help='Momentum factor (default=%(default)s)')
    parser.add_argument('--weight-decay', default=0.0, type=float, required=False,
                        help='Weight decay (L2 penalty) (default=%(default)s)')
    parser.add_argument('--warmup-nepochs', default=0, type=int, required=False,
                        help='Number of warm-up epochs (default=%(default)s)')
    parser.add_argument('--warmup-lr-factor', default=1.0, type=float, required=False,
                        help='Warm-up learning rate factor (default=%(default)s)')
    parser.add_argument('--multi-softmax', action='store_true',
                        help='Apply separate softmax for each task (default=%(default)s)')
    parser.add_argument('--fix-bn', action='store_true',
                        help='Fix batch normalization after first task (default=%(default)s)')
    parser.add_argument('--eval-on-train', action='store_true',
                        help='Show train loss and accuracy (default=%(default)s)')

    # federated learning args
    parser.add_argument('--num-clients', type=int, default=2,
                        help='federated learning client_num (default=%(default)s)')
    parser.add_argument('--client-percent', type=float, default=1.0)
    parser.add_argument('--fed-iters', type=int, default=1,
                        help='federated learning local epochs (default=%(default)s)')
    
    # unlearning related
    parser.add_argument('--unlearn', default=1, type=int, required=False,)

    return parser.parse_known_args()