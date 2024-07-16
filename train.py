import logging
import os
import sys

import numpy as np

from utils.prepare import experiment_from_args


def main():
    data_module, approach, args, device = experiment_from_args(sys.argv)

    approach.to(device)
    logging.info(f"Starting training on {device}. There are {os.cpu_count()} cpu cores available.")
    training_loop(approach, data_module, args)
    logging.info(f"Log saved to {os.path.join(args.log_dir, 'log.txt')}")


def training_loop(approach, data_module, args):
    train_dataloaders_dict = data_module.train_dataloader()
    val_dataloaders_dict = data_module.val_dataloader()
    results = {}
    for t in range(args.num_tasks):
        if args.load_ckpts:
            logging.info(30*"#" + f"\tSkipping training and loading checkpoint for task: {t}\t" + 30*"#")
            approach.load_ckpt(t)
        else:
            logging.info(30*"#" + f"\tTraining task: {t}\t" + 30*"#")
            approach.train_feature_extractor(t, train_dataloaders_dict[t], val_dataloaders_dict[t])

        if args.save_ckpts:
            logging.info(30*"#" + f"\tSaving checkpoint after task: {t}\t" + 30*"#")
            approach.save_ckpt(t)

        old_prototypes, adapted_prototypes = approach.assimilate_knowledge(t, train_dataloaders_dict[t], val_dataloaders_dict[t])
        logging.info(30*"#" + f"\tEvaluating after task: {t}\t" + 30*"#")
        results[t] = approach.eval_task(train_dataloaders_dict, val_dataloaders_dict, t, old_prototypes, adapted_prototypes)
        logging.info(30*"#" + f"\tResults after task: {t}\t" + 30*"#" + "\n")
        logging.info(parse_results(results))
        approach.store_exemplars(t, train_dataloaders_dict[t])
    return results


def parse_results(results):
    s = ""
    if "upper_bound_tag_acc" in results[0]:
        s += "\n### Upper bound ###\n"
        s += f"Upper bound task aware accuracy (%):\n"
        for t in results.keys():
            s += f"Task {t}: "
            for j in results[t]["upper_bound_taw_acc"].keys():
                s += f"{100*results[t]['upper_bound_taw_acc'][j]:.1f} "
            s += f" Avg: {100*results[t]['avg_upper_bound_taw_acc']:.1f}\n"
        # avg_inc = np.mean([results[t]['avg_upper_bound_taw_acc'] for t in results.keys()])
        # s += f"Upper bound avg incremental taw accuracy: {100*avg_inc:.1f}\n\n"

        s += f"Upper bound task agnostic accuracy (%):\n"
        for t in results.keys():
            s += f"Task {t}: "
            for j in results[t]["upper_bound_tag_acc"].keys():
                s += f"{100*results[t]['upper_bound_tag_acc'][j]:.1f} "
            s += f" Avg: {100*results[t]['avg_upper_bound_tag_acc']:.1f}\n"
        # avg_inc = np.mean([results[t]['avg_upper_bound_tag_acc'] for t in results.keys()])
        # s += f"Upper bound avg incremental tag accuracy: {100*avg_inc:.1f}\n\n"

    s += "\n##### Main results #####\n"
    s += "### Known ###\n"
    s += get_acc_string(results, "aware known", "taw_known_acc")
    s += get_acc_string(results, "agnostic known", "tag_known_acc")
    s += "### Novel ###\n"
    s += get_acc_string(results, "aware novel", "taw_novel_acc")
    s += get_acc_string(results, "agnostic novel", "tag_novel_acc")
    s += "### All ###\n"
    s += get_acc_string(results, "aware", "taw_acc")
    s += get_acc_string(results, "agnostic", "tag_acc")

    return s


def get_acc_string(results, name, key):
    s = f"Task {name} accuracy (%):\n"
    for t in results.keys():
        s += f"Task {t}: "
        for j in results[t][key].keys():
            s += f"{100*results[t][key][j]:.1f} "
        s += f" Avg: {100*results[t][f'avg_{key}']:.1f}\n"
    # avg_inc = np.mean([results[t][f'avg_{key}'] for t in results.keys()])
    # s += f"Avg incremental accuracy: {100*avg_inc:.1f}\n\n"
    s += "\n"
    return s


if __name__ == "__main__":
    main()
