#!/bin/bash
#SBATCH --job-name=GCD
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G


for SEED in 0
do
python train.py SemiSupervised CIFAR100x224 --network vit --data-dir data --supervised-head SupConHead --self-supervised-head SimCLRHead --distance-metric L2 --num-tasks 5 --batch-size 128 --lr-scheduler CosineAnnealingLR --num-workers 4 --seed $SEED --log-dir logs/table1/cifar100/GCD/
python train.py SemiSupervised StanfordCarsx224 --network vit --data-dir data --supervised-head SupConHead --self-supervised-head SimCLRHead --distance-metric L2 --num-tasks 4 --batch-size 128 --lr-scheduler CosineAnnealingLR --num-workers 4 --seed $SEED --log-dir logs/table1/cars/GCD/
python train.py SemiSupervised CUB200x224 --network vit --data-dir data --supervised-head SupConHead --self-supervised-head SimCLRHead --distance-metric L2 --num-tasks 5 --batch-size 128 --lr-scheduler CosineAnnealingLR --num-workers 4 --seed $SEED --log-dir logs/table1/cub200/GCD/
python train.py SemiSupervised FGVCAircraftx224 --network vit --data-dir data --supervised-head SupConHead --self-supervised-head SimCLRHead --distance-metric L2 --num-tasks 5 --batch-size 128 --lr-scheduler CosineAnnealingLR --num-workers 4 --seed $SEED --log-dir logs/table1/aircraft/GCD/
python train.py SemiSupervised DomainNet --network vit --data-dir data --supervised-head SupConHead --self-supervised-head SimCLRHead --distance-metric L2 --num-tasks 6 --batch-size 128 --lr-scheduler CosineAnnealingLR --num-workers 4 --seed $SEED --log-dir logs/table1/domainnet/GCD/
done