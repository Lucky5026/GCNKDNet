# GCNKD-Net: Multi-layer graph convolution networks for point cloud keypoints detection

Official code for paper 'GCNKD-Net:Multi-layer graph convolution networks for point cloud keypoints detection' (Unpublished). This code is an improvement proposed on the basis of RSKDD-Net, which improves the keypoints detector module in the feature extraction proposed by RSKDD-Net.

## Environment

- Python 3.7.0
- PyTorch 1.10.0
- Cuda 11.3
- Numpy 1.21.6
- Scipy 1.7.3
- PCL
- CMake



## Dataset

The datasets used in our paper: KITTI and Oxford RobotCar. Data preprocessing please see the Data preprocessing of [RSKDD](https://github.com/ispc-lab/RSKDD-Net.git).



## Training

The network should be trained in two stages.

- Firstly, train detector network.

```
python train_kitti.py --data_dir YOUR_DATA_DIR --seq SEQ --ckpt_dir SAVE_DIR --train_type det
```

Secondly, train descriptor network.

```
python train_kitti.py --data_dir YOUR_DATA_DIR --seq SEQ --ckpt_dir SAVE_DIR --train_type desc --pretrain_detector PRETRANIN_DETECTOR.pth
```



## Testing

For detector:

```
python test.py --data_dir DATA_DIR --model_path ./pretrain/rskdd.pth --test_type det --save_dir SAVE_DIR --test_seq TEST_SEQ --save_dir 
```
For descriptor:
```
python test.py --data_dir DATA_DIR --model_path ./pretrain/rskdd.pth --test_type desc --save_dir SAVE_DIR --test_seq TEST_SEQ --save_dir SAVE_DIR
```



## Visualization

`demo/demo_reg/demo_reg.m` is a matlab code to visualize registration of the sample pairs.

## Acknowledgements    

- Most of our code uses RSKDD and DGCNN-pytorch codes. Thank you very much to [Fan Lu, Guang Chen, Yinlong Liu, Zhongnan Qu, Alois Knoll](https://github.com/ispc-lab/RSKDD-Net.git), authors of RSKDD, and [Yue Wang](https://github.com/antao97/dgcnn.pytorch.git), author of DGCNN-pytorch.

