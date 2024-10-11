# Annotation-free Audio-Visual Segmentation
Official implementation of  [Annotation-free Audio-Visual Segmentation
](https://openaccess.thecvf.com/content/WACV2024/papers/Liu_Annotation-Free_Audio-Visual_Segmentation_WACV_2024_paper.pdf).

This paper has been accepted by **WACV 2024**, the project page is [https://jinxiang-liu.github.io/anno-free-AVS/](https://jinxiang-liu.github.io/anno-free-AVS/).

![](assets/pipeline.png)

******************
## Requirements
### Installation
Create a conda environment and install dependencies:
```shell
conda create -n sama python=3.10.11
conda activate sama

pip install -r requirements.txt
```

### Dataset
#### 1. Download the datasets 
- AVSBench
    - Please refer to [https://github.com/OpenNLPLab/AVSBench](https://github.com/OpenNLPLab/AVSBench) to download the AVSBench dataset.
    - Please download re-organized split files with the [OneDrive link](https://1drv.ms/f/s!Al8pv4sl4wmygwsS5WVpIb4fhxvT?e=OGEPsp) which might be helpful.
- AVS-Synthetic
    - Please download the dataset from [https://zenodo.org/record/8125822](https://zenodo.org/record/8125822).

#### 2. Configure the dataset locations
After downloading the datasets with annotations, please declare the directory and file locations in the `configs/sam_avs_adapter.yaml` file.

*****************
## Get Started
### Evaluation
**Model weights**: All the weights including the image backbone from SAM, audio backbone for VGGish and our pretrained models are obtained with the [OneDrive link](https://1drv.ms/f/s!Al8pv4sl4wmygwsS5WVpIb4fhxvT?e=OGEPsp).
- Please place `vggish-10086976.pth` and `sam_vit_h_4b8939.pth` in `assets` sub-folder.
- Please place the pretrained model weights in `ckpts` sub-folder.

#### Test
- Test on AVS-Synthetic test set
```shell
bash scripts/synthetic_test.sh
```

- Test on AVSBench S4 test set
```shell
bash scripts/s4_test.sh
```



- Test on AVSBench MS3 test set
```shell
bash scripts/ms3_test.sh
```



### Training
- Train AVS-Synthetic
```shell
bash scripts/synthetic_train.sh
```
- Train AVSBench S4
```shell
bash scripts/s4_train.sh
```

- Train AVSBench MS3
```shell
bash scripts/ms3_train.sh
```



***********
## Citation
```txt
@inproceedings{liu2024annotation,
  title={Annotation-free audio-visual segmentation},
  author={Liu, Jinxiang and Wang, Yu and Ju, Chen and Ma, Chaofan and Zhang, Ya and Xie, Weidi},
  booktitle={Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision},
  pages={5604--5614},
  year={2024}
}
```

## Contact
If you have any question, feel free to contact `jinxliu#sjtu.edu.cn` (replace `#` with `@`).
