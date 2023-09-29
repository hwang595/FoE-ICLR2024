cd ../baselines

CUDA_VISIBLE_DEVICES=0 python centralized_training_baseline.py \
--lr 0.1 \
--dataset cifar100 \
--epochs 300