### Trained with MS3
CUDA_VISIBLE_DEVICES=0 python test_avs.py \
--subset "ms3" \
--eval "ckpts/ms3/model_epoch_best.pth"


### Trained with MS3, with S4 warmup
CUDA_VISIBLE_DEVICES=0 python test_avs.py \
--subset "ms3" \
--eval "ckpts/ms3_ptrS4/model_epoch_best.pth"


### Trained with MS3, with S4 (with AVS-Synthetic warmup) warmup
CUDA_VISIBLE_DEVICES=0 python test_avs.py \
--subset "ms3" \
--eval "ckpts/ms3_ptrS4ptrSyn/model_epoch_best.pth"

