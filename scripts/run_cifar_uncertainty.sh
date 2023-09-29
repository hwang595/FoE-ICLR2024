cd ..

# learning based
CUDA_VISIBLE_DEVICES=0 python cifar10_uncertainty_gate.py \
--lr 0.001 \
--cutmix 0.0 \
--dataset cifar100 \
--par-strategy sample-based \
--opt_name adamw \
--num_clients 20 \
--sampled-experts 1 \
--local_ep 250 \
--caliration_type none \
--var_bs 500 \
--eval-mode learning-based \
--pred-mode label-pred \
--resume

# shortest path oracle
# NOTE: only enable `beam-search` when using nn models
CUDA_VISIBLE_DEVICES=0 python cifar10_uncertainty_gate.py \
--lr 0.001 \
--cutmix 0.0 \
--dataset cifar100 \
--par-strategy sample-based \
--opt_name adamw \
--num_clients 20 \
--sampled-experts 1 \
--local_ep 250 \
--caliration_type none \
--var_bs 1 \
--eval-mode shortest-path \
--sp-model nn \
--beam-search \
--num-apis-per-layer-bs 50 \
--k4knn 10 \
--delta-dist-threshold 2 \
--pred-mode label-pred \
--resume


CUDA_VISIBLE_DEVICES=1 python cifar10_uncertainty_gate.py \
--lr 0.001 \
--cutmix 0.0 \
--dataset cifar100 \
--par-strategy sample-based \
--opt_name adamw \
--num_clients 20 \
--sampled-experts 1 \
--local_ep 250 \
--caliration_type none \
--var_bs 1 \
--eval-mode shortest-path \
--sp-model knn \
--k4knn 10 \
--delta-dist-threshold 2.0 \
--pred-mode label-pred \
--resume

# tree oracle
CUDA_VISIBLE_DEVICES=0 python cifar10_uncertainty_gate.py \
--lr 0.001 \
--cutmix 0.0 \
--dataset cifar100 \
--par-strategy sample-based \
--opt_name adamw \
--num_clients 20 \
--sampled-experts 1 \
--local_ep 250 \
--caliration_type none \
--var_bs 500 \
--eval-mode tree-oracle \
--pred-mode label-pred \
--resume

# learning based oracle
# CUDA_VISIBLE_DEVICES=0 python cifar10_uncertainty_gate.py \
# --lr 0.001 \
# --cutmix 0.0 \
# --dataset cifar100 \
# --par-strategy sample-based \
# --opt_name adamw \
# --num_clients 20 \
# --sampled-experts 1 \
# --local_ep 250 \
# --caliration_type none \
# --var_bs 500 \
# --eval-mode learning-based-oracle \
# --pred-mode label-pred \
# --resume

# vanilla mixup
# CUDA_VISIBLE_DEVICES=0 python cifar10_uncertainty_gate.py \
# --lr 0.001 \
# --cutmix 0.0 \
# --dataset cifar100 \
# --par-strategy sample-based \
# --opt_name adamw \
# --num_clients 20 \
# --local_ep 250 \
# --caliration_type none \
# --var_bs 500 \
# --eval-mode mutual-info \
# --e-mat-score maxsoftmax \
# --dist-metric cosine \
# --save_logits \
# --use-histgram \
# --resume

# vanilla
# CUDA_VISIBLE_DEVICES=0 python cifar10_uncertainty_gate.py \
# --lr 0.001 \
# --mixup 0.0 \
# --cutmix 0.0 \
# --dataset cifar100 \
# --opt_name adamw \
# --num_clients 5 \
# --local_ep 250 \
# --caliration_type none \
# --var_bs 500 \
# --save_logits \
# --resume

# no mixup but with temp scaling
# CUDA_VISIBLE_DEVICES=1 python cifar10_uncertainty_gate.py \
# --lr 0.001 \
# --mixup 0.0 \
# --cutmix 0.0 \
# --dataset cifar10 \
# --opt_name adamw \
# --local_ep 250 \
# --caliration_type temp-scaling \
# --var_bs 500

# CUDA_VISIBLE_DEVICES=0 python cifar10_uncertainty_gate.py \
# --lr 0.1 \
# --opt_name sgd \
# --local_ep 300 \
# --caliration_type temp-scaling \
# --var_bs 2000