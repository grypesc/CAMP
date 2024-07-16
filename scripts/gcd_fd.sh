#!/bin/bash
#SBATCH --job-name=GCD+LWF
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G


for EXEMPLARS in 0
do
CUDA_VISIBLE_DEVICES=0 python train.py SemiSupervised StanfordCarsx224 --network vit --data-dir data --supervised-head SupConHead --self-supervised-head SimCLRHead --distance-metric L2 --alpha 0.5 --num-exemplars $EXEMPLARS --num-tasks 4 --batch-size 128 --lr-scheduler CosineAnnealingLR --distiller FeatureDistiller --num-workers 4 --seed 0 --log-dir logs/table1/cars/FD/
CUDA_VISIBLE_DEVICES=0 python train.py SemiSupervised CUB200x224 --network vit --data-dir data --supervised-head SupConHead --self-supervised-head SimCLRHead --distance-metric L2 --alpha 0.5 --num-exemplars $EXEMPLARS --num-tasks 5 --batch-size 128 --lr-scheduler CosineAnnealingLR --distiller FeatureDistiller  --num-workers 4 --seed 0 --log-dir logs/table1/cub200/FD/
CUDA_VISIBLE_DEVICES=0 python train.py SemiSupervised FGVCAircraftx224 --network vit --data-dir data --supervised-head SupConHead --self-supervised-head SimCLRHead --distance-metric L2 --alpha 0.5 --num-exemplars $EXEMPLARS --num-tasks 5 --batch-size 128 --lr-scheduler CosineAnnealingLR --distiller FeatureDistiller --num-workers 4 --seed 0 --log-dir logs/table1/aircraft/FD/
CUDA_VISIBLE_DEVICES=0 python train.py SemiSupervised DomainNet --network vit --data-dir data --supervised-head SupConHead --self-supervised-head SimCLRHead --distance-metric L2 --alpha 0.5 --num-exemplars $EXEMPLARS --num-tasks 6 --batch-size 128 --lr-scheduler CosineAnnealingLR --distiller FeatureDistiller --num-workers 4 --seed 0 --log-dir logs/table1/domainnet/FD/
CUDA_VISIBLE_DEVICES=0 python train.py SemiSupervised CIFAR100x224 --network vit --data-dir data --supervised-head SupConHead --self-supervised-head SimCLRHead --distance-metric L2 --alpha 0.5 --num-exemplars $EXEMPLARS --num-tasks 5 --batch-size 128 --lr-scheduler CosineAnnealingLR --distiller FeatureDistiller --num-workers 4 --seed 0 --log-dir logs/table1/cifar100/FD/
done