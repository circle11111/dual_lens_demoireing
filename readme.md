# Video Demoireing using Focused-Defocused Dual-Camera System

[Xuan Dong*](), [Xiangyuan Sun*](), [Xia Wang*](), [Jian Song](), [Ya Li](), [Weixin Li]()

*Equal contribution.

This repo is the official implementation of [Video Demoireing using Focused-Defocused Dual-Camera System](https://arxiv.org/abs/2508.03449).

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
You can download our [DualSynthetic]() or [DualSyntheticVideo]() dataset. The files should be organized like
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
- DualSynthetic [[Baidu Disk]]()
- DualSyntheticVideo [[Baidu Disk]]() 
## Citations

If our work is useful for your research, please consider citing:

```
@misc{dong2025videodemoireingusingfocuseddefocused,
      title={Video Demoireing using Focused-Defocused Dual-Camera System}, 
      author={Xuan Dong and Xiangyuan Sun and Xia Wang and Jian Song and Ya Li and Weixin Li},
      year={2025},
      eprint={2508.03449},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2508.03449}, 
}
```
## Acknowledgements
Special thanks to the following repositories for supporting our research:
- [UHDM](https://github.com/CVMI-Lab/UHDM)
- [GigaGAN](https://github.com/JiauZhang/GigaGAN)
- [FlowFormer](https://github.com/drinkingcoder/FlowFormer-Official)