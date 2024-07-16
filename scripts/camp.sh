#!/bin/bash
#SBATCH --job-name=CAMP
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G


for SEED in 0
do
python train.py CAMP CIFAR100x224 --network vit --data-dir data --distance-metric L2 --num-tasks 5 --batch-size 128 --lr-scheduler CosineAnnealingLR --adapt-prototypes --distiller MLPDistiller --alpha 0.5 --num-workers 4 --num-exemplars 0 --seed $SEED --log-dir logs/table1/cifar100/CAMP/
python train.py CAMP StanfordCarsx224 --network vit --data-dir data --distance-metric L2 --num-tasks 4 --batch-size 128 --lr-scheduler CosineAnnealingLR --adapt-prototypes --distiller MLPDistiller --alpha 0.5 --num-workers 4 --num-exemplars 0 --seed $SEED --log-dir logs/table1/cars/CAMP/
python train.py CAMP CUB200x224 --network vit --data-dir data  --distance-metric L2 --num-tasks 5  --batch-size 128 --lr-scheduler CosineAnnealingLR --adapt-prototypes --distiller MLPDistiller --alpha 0.5 --num-workers 4 --num-exemplars 0 --save-ckpts --seed 0 --log-dir logs/estimating/cub200/CAMP/
python train.py CAMP FGVCAircraftx224 --network vit --data-dir data  --distance-metric L2 --num-tasks 5  --batch-size 128 --lr-scheduler CosineAnnealingLR --adapt-prototypes --distiller MLPDistiller --alpha 0.5 --num-workers 4 --num-exemplars 0 --seed $SEED --log-dir logs/table1/aircraft/CAMP/
python train.py CAMP DomainNet --network vit --data-dir data  --distance-metric L2 --num-tasks 6  --batch-size 128 --lr-scheduler CosineAnnealingLR --adapt-prototypes --distiller MLPDistiller --alpha 0.5 --num-workers 4 --num-exemplars 0 --seed $SEED --log-dir logs/table1/domainnet/CAMP/
done