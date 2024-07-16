#!/bin/bash
#SBATCH --job-name=GCDEWC
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G


for SEED in 0
do
python train.py SimGCD StanfordCarsx224 --network vit --data-dir data  --num-tasks 4  --batch-size 128 --lr-scheduler CosineAnnealingLR --num-workers 4 --clip 1.0  --seed $SEED --log-dir logs/table1/cars/SimGCD/
python train.py SimGCD CUB200x224 --network vit --data-dir data --num-tasks 5 --batch-size 128 --lr-scheduler CosineAnnealingLR --num-workers 4 --clip 1.0  --seed $SEED --log-dir logs/table1/cub200/SimGCD/
python train.py SimGCD FGVCAircraftx224 --network vit --data-dir data --num-tasks 5 --batch-size 128 --lr-scheduler CosineAnnealingLR --num-workers 4 --clip 1.0  --seed $SEED --log-dir logs/table1/aircraft/SimGCD/
python train.py SimGCD DomainNet --network vit --data-dir data --num-tasks 6 --batch-size 128 --lr-scheduler CosineAnnealingLR --num-workers 4 --clip 1.0  --seed $SEED --log-dir logs/table1/domainnet/SimGCD/
python train.py SimGCD CIFAR100x224 --network vit --data-dir data --num-tasks 5 --batch-size 128 --lr-scheduler CosineAnnealingLR --num-workers 4 --clip 1.0  --seed $SEED --log-dir logs/table1/cifar100/SimGCD/
done