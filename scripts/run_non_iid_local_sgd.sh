CUDA_VISIBLE_DEVICES=0 python mnist_non_iid_local_sgd.py \
--num_clients=10 \
--client_per_round=10 \
--num_total_clients=50 \
--alpha=0.9 \
--fl_rounds=20 \
--local_ep=5 \
--local_bs=128 \
--lr=0.001 \
--momentum=0.9

# CUDA_VISIBLE_DEVICES=0 python cifar10_non_iid_local_sgd.py \
# --num_clients=10 \
# --client_per_round=10 \
# --num_total_clients=50 \
# --alpha=0.9 \
# --fl_rounds=250 \
# --local_ep=5 \
# --local_bs=64 \
# --lr=0.001 \
# --momentum=0.9