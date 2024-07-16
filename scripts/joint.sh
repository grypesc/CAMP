#!/bin/bash
#SBATCH --job-name=joint
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G


for SEED in 0
do
python train.py SemiSupervised StanfordCarsx224 --network vit --data-dir data --supervised-head SupConHead --self-supervised-head SimCLRHead --distance-metric L2 --num-tasks 1 --epochs 200 --batch-size 128 --lr-scheduler CosineAnnealingLR --num-workers 4 --seed $SEED --log-dir logs/table1/cars/joint/
python train.py SemiSupervised CUB200x224 --network vit --data-dir data --supervised-head SupConHead --self-supervised-head SimCLRHead --distance-metric L2 --num-tasks 1 --epochs 200 --batch-size 128 --lr-scheduler CosineAnnealingLR --num-workers 4 --seed $SEED --log-dir logs/table1/cub200/joint/
python train.py SemiSupervised FGVCAircraftx224 --network vit --data-dir data --supervised-head SupConHead --self-supervised-head SimCLRHead --distance-metric L2 --num-tasks 1 --epochs 200 --batch-size 128 --lr-scheduler CosineAnnealingLR --num-workers 4 --seed $SEED --log-dir logs/table1/aircraft/joint/
python train.py SemiSupervised DomainNet --network vit --data-dir data --supervised-head SupConHead --self-supervised-head SimCLRHead --distance-metric L2 --num-tasks 1 --epochs 200 --batch-size 128 --lr-scheduler CosineAnnealingLR --num-workers 4 --seed $SEED --log-dir logs/table1/domainnet/joint/
python train.py SemiSupervised CIFAR100x224 --network vit --data-dir data --supervised-head SupConHead --self-supervised-head SimCLRHead --distance-metric L2 --num-tasks 1 --epochs 200 --batch-size 128 --lr-scheduler CosineAnnealingLR --num-workers 4 --seed $SEED --log-dir logs/table1/cifar100/joint/
done