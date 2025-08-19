# Video Demoireing using Focused-Defocused Dual-Camera System

[Xuan Dong*](), [Xiangyuan Sun*](), [Xia Wang*](), [Jian Song](), [Ya Li](), [Weixin Li]()

*Equal contribution.

[Paper](https://arxiv.org/abs/2508.03449) [Project](https://circle11111.github.io/TPAMI_Demoireing_Webpage/) [Dataset](https://circle11111.github.io/Dual-demoireing_datasets/)

## Environment Setup

To set up your environment, follow these steps:

```
conda create -n my_env python=3.8 -y
conda activate my_env
conda install pytorch==1.10.1 torchvision==0.11.2 cudatoolkit=11.1 -c pytorch -y
pip install -r requirements.txt
```

## Test our demoireing network

Run the following command to test our demoireing network. Results are saved in the `results/ambush_4/demoired` folder.
```
python test_model.py
```
If you want to test your own pre-aligned image pairs, please replace `main_path`, `guide_path`, and `demoired_path` by your own image paths.

## Test our demoireing pipeline

Run the following command to test our demoireing pipeline. Results are saved in the `results/20230601_154119` folder.
```
bash test_pipe.sh
```
You can change `--video_name` and `--num` to check other examples.

## Train your own demoireing network

### Step-1: data preparation
You can download our [DualSynthetic](https://pan.baidu.com/s/1Sv1eva5IjY6NrjN-Ix04ww?pwd=62tn) or [DualSyntheticVideo](https://pan.baidu.com/s/1Xc5YsikoqRBbWXjmckFmPg?pwd=6prt) dataset. The files should be organized like
```
--|--train--|--main
  |         |--guide
  |         |--target
  |
  |---val---|--main
  |         |--guide
  |         |--target
  |
  |--test---|--main
            |--guide
            |--target
```
### Step-2: start training
Open `train_gan.sh` and replace `--dataset`, `--data_path` and `--save_dir` by your own paths, then run the following command
```
bash train_gan.sh
```
You can see `train.csv` and `val.csv` in the `datasets/your_dataset` folder. The contents will be organized like `example.csv`. Your models will be saved in the `your_save_dir` folder.

## Datasets

- DualReal [[Baidu Disk]](https://pan.baidu.com/s/1jV8aiL559LtwRMb_nIQu7A?pwd=ekbr)
- DualSynthetic [[Baidu Disk]](https://pan.baidu.com/s/1Sv1eva5IjY6NrjN-Ix04ww?pwd=62tn)
- DualSyntheticVideo [[Baidu Disk]](https://pan.baidu.com/s/1Xc5YsikoqRBbWXjmckFmPg?pwd=6prt) 
## Citations

If our work is useful for your research, please consider citing:

```
@ARTICLE{DuDemoire2025,
  author={Dong, Xuan and Sun, Xiangyuan and Wang, Xia and Song, Jian and Li, Ya and Li, Weixin},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence}, 
  title={Video Demoireing using Focused-Defocused Dual-Camera System}, 
  year={2025},
  volume={},
  number={},
  pages={1-15},
  keywords={Video demoireing; Focused-defocused dual camera},
  doi={10.1109/TPAMI.2025.3596700}}
```
## Acknowledgements
Special thanks to the following repositories for supporting our research:
- [UHDM](https://github.com/CVMI-Lab/UHDM)
- [GigaGAN](https://github.com/JiauZhang/GigaGAN)
- [FlowFormer](https://github.com/drinkingcoder/FlowFormer-Official)