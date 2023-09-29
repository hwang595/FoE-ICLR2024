declare -a NUM_CLIENTS
NUM_CLIENTS=(4 5 10 30 50)
for IDX in $(seq 0 1 4)
do
    CUDA_VISIBLE_DEVICES=$IDX nohup python cifar10_non_iid_local_gate.py \
    --num_clients=${NUM_CLIENTS[$IDX]} \
    --num_total_clients=50 \
    --alpha=0.9 \
    --fl_rounds=250 \
    --local_ep=5 \
    --local_bs=64 \
    --lr=0.001 \
    --momentum=0.9 > nohup_${NUM_CLIENTS[$IDX]}.out &
done


CUDA_VISIBLE_DEVICES=0 python cifar10_non_iid_local_gate.py \
--num_clients=30 \
--num_total_clients=50 \
--alpha=0.9 \
--fl_rounds=250 \
--local_ep=5 \
--local_bs=64 \
--lr=0.001 \
--spreadout_regu \
--spreadout_opt_step=50 \
--spreadout_lr=0.0002 \
--momentum=0.9