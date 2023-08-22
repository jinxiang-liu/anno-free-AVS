
### Trained with S4
CUDA_VISIBLE_DEVICES=3 python test_avs.py \
--subset "s4" \
--config "configs/sam_avs_adapter.yaml" \
--eval "ckpts/s4/model_epoch_best.pth"


### Trained with S4, with AVS-Synthetic warmup
CUDA_VISIBLE_DEVICES=3 python test_avs.py \
--subset "s4" \
--config "configs/sam_avs_adapter.yaml" \
--eval "ckpts/s4_ptrSyn/model_epoch_best.pth"

