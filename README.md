# TennisOneV4
对比V3，暂时去除网球的追踪，改为使用yolo直接检测关键帧，用精度换速度。

大幅修改最终视频UI界面，增加对姿势的打分。

利用n1n.ai接入gemini3.1-pro-preview进行AI建议生成


# 1. 安装
1. 配置cuda，不再赘述
2. 创建虚拟环境并安装所需的库
```
conda create -n tennisone python=3.11

conda activate tennisone

# 若使用CUDA 12.8或更高版本，建议先安装torch
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu130

# 安装YOLO框架、vit-pose框架
pip install -r requirements.txt
```
3. 安装大恒摄像头相关库
```
wget https://gb.daheng-imaging.com/CN/Software/Cameras/Linux/Galaxy_Linux-x86_Gige-U3_32bits-64bits_2.4.2507.9231.zip

unzip Galaxy_Linux-x86_Gige-U3_32bits-64bits_2.4.2507.9231.zip

cd Galaxy_Linux-x86_Gige-U3_32bits-64bits_2.4.2507.9231

./Galaxy_camera.run

wget https://gb.daheng-imaging.com/CN/Software/Cameras/Python/Galaxy_Linux_Python_2.4.2503.9202.zip

unzip Galaxy_Linux_Python_2.4.2503.9202.zip

cd ./Galaxy_Linux_Python_2.4.2503.9202/Galaxy_Linux_Python_2.4.2503.9202/api

python3 setup.py build
sudo python3 setup.py install
```

4. 若在WSL中使用，注意WSL无法直接读取USB设备，因此需要额外安装[usbipd工具](https://github.com/dorssel/usbipd-win/releases/download/v5.3.0/usbipd-win_5.3.0_x64.msi)


在powershell中输入`usbipd list`可以查看当前物理机中所有已连接的usb设备，通过插拔与摄像机的连接可以判断出对应的BUSID，先使用命令`usbipd bind --busid <对应的BUSID>`共享对应的USB接口，然后使用命令`usbipd attach --wsl --busid <对应的BUSID>`将该USB接口映射给WSL。

具体可参照[微软官方教程](https://learn.microsoft.com/zh-cn/windows/wsl/connect-usb)

> **注意：**  将该USB接口映射给WSL后，物理机将无法访问该USB接口

# 2. 使用
```
python plot_ui_v2.py
```
# 3. TODO
- [√] 将大恒摄像头与sport-vision对接，并将骨骼识别模型改为YOLO11
- [ ] 将整个流程与sport-vision对接，打造一个网页端实时检测系统
- [ ] 完善REDEME
- [ ] 打造本地知识库，接入本地大模型
- [ ] 加入st-gcn进行动作识别（会增加耗时）