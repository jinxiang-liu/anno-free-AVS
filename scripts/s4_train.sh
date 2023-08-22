### Trained with S4 
CUDA_VISIBLE_DEVICES=0,4 python -m torch.distributed.launch \
--nproc_per_node=2 --master_port=25648 train_avs.py --name "s4" \
--config "configs/sam_avs_adapter.yaml"



### Trained with S4 with AVS-Synthetic warmup
CUDA_VISIBLE_DEVICES=0,4 python -m torch.distributed.launch \
--nproc_per_node=2 --master_port=25648 train_avs.py --name "s4" \
--config "configs/sam_avs_adapter.yaml" \
--pretrained_weights "ckpts/synthetic/model_epoch_best.pth"


