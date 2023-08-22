# Trained with AVS-Synthetic
CUDA_VISIBLE_DEVICES=0 python test_avs.py \
--subset "synthetic" \
--config "configs/sam_avs_adapter.yaml" \
--eval "ckpts/synthetic/model_epoch_best.pth"