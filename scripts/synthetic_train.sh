CUDA_VISIBLE_DEVICES=1,2 python -m torch.distributed.launch \
--nproc_per_node=2 --master_port=25649 \
train_avs.py \
--name "synthetic" \
--config configs/sam_avs_adapter.yaml \
--subset "synthetic" 




