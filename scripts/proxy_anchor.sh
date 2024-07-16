#!/bin/bash
#SBATCH --job-name=GCDEWC
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G


for SEED in 0
do
python train.py ProxyAnchor StanfordCarsx224 --network vit --data-dir data  --supervised-head ProxyAnchorHead --num-tasks 4 --epochs 100 --optimizer AdamW --lr 1e-4 --batch-size 256 --lr-scheduler CosineAnnealingLR --num-workers 4 --distiller FeatureDistiller --alpha 0.5 --num-exemplars 20 --clip 1.0  --seed $SEED --log-dir logs/table1/cars/PA/
python train.py ProxyAnchor CUB100x224 --network vit --data-dir data --supervised-head ProxyAnchorHead --num-tasks 5 --epochs 100 --optimizer AdamW --lr 1e-4 --batch-size 256 --lr-scheduler CosineAnnealingLR --num-workers 4 --distiller FeatureDistiller --alpha 0.5 --num-exemplars 20 --clip 1.0  --seed $SEED --log-dir logs/table1/cub100/PA/
python train.py ProxyAnchor FGVCAircraftx224 --network vit --data-dir data --supervised-head ProxyAnchorHead --num-tasks 5 --epochs 100 --optimizer AdamW --lr 1e-4 --batch-size 256 --lr-scheduler CosineAnnealingLR --num-workers 4 --distiller FeatureDistiller --alpha 0.5 --num-exemplars 20 --clip 1.0  --seed $SEED --log-dir logs/table1/aircraft/PA/
python train.py ProxyAnchor DomainNet --network vit --data-dir data --supervised-head ProxyAnchorHead --num-tasks 6 --epochs 100 --optimizer AdamW --lr 1e-4 --batch-size 256 --lr-scheduler CosineAnnealingLR --num-workers 4 --distiller FeatureDistiller --alpha 0.5 --num-exemplars 20 --clip 1.0  --seed $SEED --log-dir logs/table1/domainnet/PA/
python train.py ProxyAnchor CIFAR100x224 --network vit --data-dir data --supervised-head ProxyAnchorHead --num-tasks 5 --epochs 100 --optimizer AdamW --lr 1e-4 --batch-size 256 --lr-scheduler CosineAnnealingLR --num-workers 4 --distiller FeatureDistiller --alpha 0.5 --num-exemplars 20 --clip 1.0  --seed $SEED --log-dir logs/table1/cifar100/PA/
done