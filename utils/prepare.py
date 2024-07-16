import argparse
import inspect
import logging
import torch
import os

import approaches
import datasets

from utils.train import create_log_dir, seed_everything


def add_base_args(parser):
    parser.add_argument('--log-dir',
                        help='directory where logs and checkpoints will be dumped',
                        type=str,
                        default='logs/')
    parser.add_argument('--log-debug', action='store_true')
    return parser


def experiment_from_args(argv):
    parser = argparse.ArgumentParser()

    apprchs = {k: v for k, v in inspect.getmembers(approaches, inspect.isclass) if
             issubclass(v, torch.nn.Module) and not inspect.isabstract(v)}
    dss = {k: v for k, v in inspect.getmembers(datasets, inspect.isclass) if
           issubclass(v, torch.utils.data.Dataset) and not inspect.isabstract(v)}

    parser.add_argument('approach', help='Model architecture', choices=apprchs.keys())
    parser.add_argument('dataset', help='Dataset to use', choices=dss.keys())
    parser.add_argument('--seed', type=int, default=0)
    main_args = parser.parse_args(argv[1:3])

    apprch_class = apprchs[main_args.approach]
    data_module_class = dss[main_args.dataset]

    parser = add_base_args(parser)
    parser = apprch_class.add_argparse_args(parser)
    parser = data_module_class.add_argparse_args(parser)
    args = parser.parse_args(argv[1:])
    seed_everything(args.seed)

    # Set up logger
    args.log_dir = create_log_dir(args.log_dir)

    logging.getLogger.propagate = False
    logging.root.handlers = []
    logging.basicConfig(
        level=logging.DEBUG if args.log_debug else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.FileHandler(os.path.join(args.log_dir, "log.txt")),
                  logging.StreamHandler()]
    )

    logging.info(args)
    data_module = data_module_class(args)
    logging.info(str(data_module))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = apprch_class(args, data_module, device)
    return data_module, model, args, device
