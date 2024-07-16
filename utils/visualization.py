import os
import torch
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


@torch.no_grad()
def get_features(dl, backbone, device):
    features, labels = [], []
    for X, Y in dl:
        X = X.to(device)
        feats = backbone(X)
        features.append(feats)
        labels.append(Y)
    features = torch.cat(features).to('cpu')
    labels = torch.cat(labels)
    return features, labels


@torch.no_grad()
def visualize_drift(old_dl, new_dl, old_backbone, new_backbone, old_protos, adapted_protos, gt_protos, device, log_dir):
    # naming convention: o -> old, n -> new, a - all; first letter -> data, second letter -> backbone
    # so 'no' means 'new data forwarded throug old backbone'
    
    torch.save(old_protos, os.path.join(log_dir, "old_protos.ckpt"))
    torch.save(adapted_protos, os.path.join(log_dir, "adapted_protos.ckpt"))
    torch.save(gt_protos, os.path.join(log_dir, "gt_protos.ckpt"))
    
    oo_feats, oo_labels = get_features(old_dl, old_backbone, device)
    on_feats, on_labels = get_features(old_dl, new_backbone, device)
    nn_feats, nn_labels = get_features(new_dl, new_backbone, device)

    # Get the unique labels and their corresponding colors
    label_colors = sns.color_palette("hls", len(gt_protos.keys()))


    plt.figure(figsize=(12, 12))

    oo_df = pd.DataFrame()
    oo_df["feat_1"] = oo_feats[:, 0]
    oo_df["feat_2"] = oo_feats[:, 1]
    oo_df["labels"] = oo_labels

    ax = sns.scatterplot(
        x="feat_1",
        y="feat_2",
        hue="labels",
        palette=label_colors,
        data=oo_df,
        legend="full",
        alpha=0.7,
        marker='.',
        s=30,
    )

    # Plot the means as points on the scatterplot with matching colors
    for label, proto in old_protos[0].items():
        color = label_colors[label]
        proto = proto.cpu()
        ax.scatter(
            x=proto[0],
            y=proto[1],
            c=[color],
            marker='X',
            s=100,
        )

    # ax.set(xlabel="", ylabel="", xticklabels=[], yticklabels=[])
    # ax.tick_params(left=False, right=False, bottom=False, top=False)

    plt.tight_layout()
    plt.savefig(os.path.join(log_dir, "oo.png"))


    plt.figure(figsize=(12, 12))

    an_df = pd.DataFrame()
    an_df["feat_1"] = torch.cat([on_feats[:, 0], nn_feats[:, 0]])
    an_df["feat_2"] = torch.cat([on_feats[:, 1], nn_feats[:, 1]])
    an_df["labels"] = torch.cat([on_labels, nn_labels])

    ax = sns.scatterplot(
        x="feat_1",
        y="feat_2",
        hue="labels",
        palette=label_colors,
        data=an_df,
        legend="full",
        alpha=0.7,
        marker='.',
        s=30,
    )

    for label, proto in old_protos[0].items():
        color = label_colors[label]
        proto = proto.cpu()
        ax.scatter(
            x=proto[0],
            y=proto[1],
            c=[color],
            marker='X',
            s=100,
            edgecolors='black',
            linewidths=1,
        )

        adapted_proto = adapted_protos[0][label].cpu()
        ax.scatter(
            x=adapted_proto[0],
            y=adapted_proto[1],
            c=[color],
            marker='s',
            s=100,
            edgecolors='black',
            linewidths=1,
        )

        arrow_props = dict(facecolor='black', arrowstyle='->')
        ax.annotate(
            '',
            xy=(adapted_proto[0], adapted_proto[1]),
            xytext=(proto[0], proto[1]),
            arrowprops=arrow_props
        )

    for label, gt_proto in gt_protos.items():
        color = label_colors[label]
        gt_proto = gt_proto.cpu()
        ax.scatter(
            x=gt_proto[0],
            y=gt_proto[1],
            c=[color],
            marker='^',
            s=100,
            edgecolors='black',
            linewidths=1,
        )

    # ax.set(xlabel="", ylabel="", xticklabels=[], yticklabels=[])
    # ax.tick_params(left=False, right=False, bottom=False, top=False)

    plt.tight_layout()
    plt.savefig(os.path.join(log_dir, "on.png"))
