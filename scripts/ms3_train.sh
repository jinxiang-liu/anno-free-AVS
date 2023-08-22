#### Trained with MS3
CUDA_VISIBLE_DEVICES=1,2 python -m torch.distributed.launch \
--nproc_per_node=2 train_avs.py \
--subset "ms3"  --name "ms3" \
--config configs/sam_avs_adapter.yaml 



#### Trained with MS3, with S4 warmup
CUDA_VISIBLE_DEVICES=1,2 python -m torch.distributed.launch \
--nproc_per_node=2 train_avs.py \
--subset "ms3"  --name "ms3" \
--config configs/sam_avs_adapter.yaml \
--pretrained_weights "ckpts/s4/model_epoch_best.pth"



#### Trained with MS3, with S4 (with AVS-Synthetic warmup) warmup
CUDA_VISIBLE_DEVICES=1,2 python -m torch.distributed.launch \
--nproc_per_node=2 train_avs.py \
--subset "ms3"  --name "ms3" \
--config configs/sam_avs_adapter.yaml \
--pretrained_weights "ckpts/s4_ptrSyn/model_epoch_best.pth"  