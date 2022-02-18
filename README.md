# CSGStumpNet
The official implementation of CSG-Stump: A Learning Friendly CSG-Like Representation for Interpretable Shape Parsing

### [Paper](https://arxiv.org/abs/666.666) |   [Project page](https://kimren227.github.io/projects/CSGStump/)

## Citation
If you find our work interesting and benifits your research, please consider citing:

	@inproceedings{ren2021csg,
	  title={CSG-Stump: A Learning Friendly CSG-Like Representation for Interpretable Shape Parsing},
	  author={Ren, Daxuan and Zheng, Jianmin and Cai, Jianfei and Li, Jiatong and Jiang, Haiyong and Cai, Zhongang and Zhang, Junzhe and Pan, Liang and Zhang, Mingyuan and Zhao, Haiyu and others},
	  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
	  pages={12478--12487},
	  year={2021}
	}

## Setup
### Install envoriment:
We recommand using Anaconda to set the envoriment, once Anacodna in installed, run the following command.

```
conda create --name CSGStumpNet python=3.7
conda activate CSGStumpNet
conda install pytorch==1.7.1 torchvision==0.8.2 torchaudio==0.7.2 cudatoolkit=10.1 -c pytorch
conda install -c open3d-admin open3d=0.9
conda install numpy
conda install pymcubes
conda install tensorboard
conda install scipy
pip install tqdm
```

## Datasets and pre-trained weights
### Dataset
You can use the pre-prepared dataset from [OccNet](https://github.com/autonomousvision/occupancy_networks)(consider citing them), you can download the data by 
```
mkdir data
cd data
wget https://s3.eu-central-1.amazonaws.com/avg-projects/occupancy_networks/data/dataset_small_v1.1.zip
unzip dataset_small_v1.1.zip
```

If you want to prepare data yourself (maybe you want to generate the watertight mesh etc.), please refer to [this link](https://github.com/autonomousvision/occupancy_networks/tree/master/external/mesh-fusion).

### Pre-Train Weights
Please download pre-trained weights from this [google drive](https://drive.google.com/drive/folders/1QQIvPXreE_BECFP7Cnr3i-HV9iklm2gw?usp=sharing)

### Evaluate using pre-trian weights
```
python eval.py --config_path ./configs/plane.json
```
### Train from stratch
```
python train.py --config_path ./configs/plane.json
```
### Evaluation
```
python metrics.py --config_path ./configs/plane.json
```

## License
This project is licensed under the terms of the MIT license (see LICENSE for details).





