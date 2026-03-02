# ICDUN
Enhancing Light Field Reflection Removal through Intra and Cross-View Nonlocal Similarity Guided Deep Unfolding
The dataset and source code for ICDUN: Intra and Cross View Nonlocal Similarity Guided Deep Unfolding Network for Light Field Reflection Removal. This work is currently submitted to The Visual Computer.[Paper]() | [Bibtex]()

## 🛠️ Dependencies
Python 3.8 (Recommend to use [Anaconda](https://www.anaconda.com/download/#linux))

PyTorch 1.7.0 or higher

einops 

Numpy & Scipy

Matplotlib

OpenCV-Python

timm 

## 📂 Datasets
Our dataset consists of both synthetic and real-world light field (LF) scenes. We use 400 (synthetic) + 50 (real-world) scenes for training and 20 (synthetic) + 20 (real-world) scenes for testing.

Please download our datasets via [Baidu Drive](https://pan.baidu.com/s/1GXKF9HzT0sKhN91z0gvUHw?pwd=vida) and place them in the ./ICDUN_DATA/ folder.

You can use the provided ./read_lf_h5.py script to load the .h5 data.
  ```
Project Structure:
├── ./ICDUN_DATA/
│    ├── ICDUN_training/
│    │    ├── mixturelf_syn/
│    │    │    ├── mixturelf_syn_001.h5
│    │    │    └── ...
│    │    ├── mixturelf_real/
│    │    │    ├── mixturelf_real_001.h5
│    │    │    └── ...
│    ├── ICDUN_testing/
│    │    ├── synthetic/
│    │    │    ├── test_syn_001.h5
│    │    │    └── ...
│    │    └── realworld/
│    │         ├── test_real_001.h5
│    │         └── ...
  ```
  
## 🚀 Train & Test
1. Pretrained Models
Before training or testing, please download the checkpoints from [Baidu Drive / Google Drive Link] and put them into the ./pretrained_model/ folder.

2. Training
* To start the training process with the default settings:

```shell
python train.py
```
3. Testing
* To evaluate the model and generate reflection-removed results:

```shell
python test.py
```

## 📜 Citation
If you find this work helpful for your research, please consider citing our paper:
```latex

```



## 🔗 Related Projects
[DMI](https://github.com/Yutong2022/LFRR?tab=readme-ov-file#lfrr_tci2024)

[BasicLFSR](https://github.com/ZhengyuLiang24/BasicLFSR)
