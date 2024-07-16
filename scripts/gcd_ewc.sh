#!/bin/bash
#SBATCH --job-name=GCDEWC
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G


for SEED in 0
do
python train.py GCD_EWC CIFAR100x224 --network vit --data-dir data --supervised-head CrossEntropyHead --self-supervised-head SimCLRHead --distill-loss-fn mse --distance-metric L2 --num-tasks 5 --batch-size 128 --lr 0.1 --lr-scheduler CosineAnnealingLR --num-workers 4 --clip 1.0  --seed $SEED --log-dir logs/table1/cifar100/GCD+EWC/
python train.py GCD_EWC StanfordCarsx224 --network vit --data-dir data --supervised-head CrossEntropyHead --self-supervised-head SimCLRHead --distill-loss-fn mse --distance-metric L2 --num-tasks 4 --batch-size 128 --lr 0.1 --lr-scheduler CosineAnnealingLR --num-workers 4 --clip 1.0  --seed $SEED --log-dir logs/table1/cars/GCD+EWC/
python train.py GCD_EWC CUB200x224 --network vit --data-dir data --supervised-head CrossEntropyHead --self-supervised-head SimCLRHead --distill-loss-fn mse --distance-metric L2 --num-tasks 5 --batch-size 128 --lr 0.1 --lr-scheduler CosineAnnealingLR --num-workers 4 --clip 1.0  --seed $SEED --log-dir logs/table1/cub200/GCD+EWC/
python train.py GCD_EWC FGVCAircraftx224 --network vit --data-dir data --supervised-head CrossEntropyHead --self-supervised-head SimCLRHead --distill-loss-fn mse --distance-metric L2 --num-tasks 5 --batch-size 128 --lr 0.1 --lr-scheduler CosineAnnealingLR --num-workers 4 --clip 1.0  --seed $SEED --log-dir logs/table1/aircraft/GCD+EWC/
python train.py GCD_EWC DomainNet --network vit --data-dir data --supervised-head CrossEntropyHead --self-supervised-head SimCLRHead --distill-loss-fn mse --distance-metric L2 --num-tasks 6 --batch-size 128 --lr 0.1 --lr-scheduler CosineAnnealingLR --num-workers 4 --clip 1.0  --seed $SEED --log-dir logs/table1/domainnet/GCD+EWC/
done