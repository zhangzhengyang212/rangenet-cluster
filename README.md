# rangenet_pp-cluster 
在语义分割rangenet上加入聚类、点云补全和边界框操作

**准备**  
--
1.下载和解压缩 libtorch

2.搭建深度学习环境,安装 nvidia-driver, CUDA, TensorRT, cuDNN

本机配置：ubuntu20.04+navida-driver535.183.01+CUDA11.6+TensorRT8.4.1.5+opencv3.4.5+cudnn8.4.1.50  

3.安装 apt 和 Python 包  
```
sudo apt install build-essential python3-dev python3-pip apt-utils git cmake libboost-all-dev libyaml-cpp-dev libopencv-dev python3-empy libfmt-dev
pip install catkin_tools trollius numpy  
```
**导入模型文件和数据集**  
--
下载[数据文件](https://pan.baidu.com/s/1lqvA5Lvo6oZ_mvrArHxaVg?pwd=1234)   

保存的目录结构  
.
├── model  
│   ├── arch_cfg.yaml  
│   ├── data_cfg.yaml  
│   └── model.onnx  
├── data  
│   ├── 000000.pcd  
│   ├── kitti_2011_09_30_drive_0027_synced  
│   └── kitti_2011_09_30_drive_0027_synced.bag  
    
**编译**  
--
```
cd ~/rangenet_pp/  
source ~/.bashrc 
rm -rf build devel 
catkin build
```  
**运行**
--
```  
source devel/setup.bash  
roslaunch rangenet_pp ros1_rangenet.launch 
source ~/rangenet_pp/devel/setup.bash  
roslaunch rangenet_pp ros1_bag.launch  
