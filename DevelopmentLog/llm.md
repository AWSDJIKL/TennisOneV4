# 网球项目大模型相关开发记录


20260326: 根据资方在中期验收中的反馈意见，目前存在以下待改进事项：
- 根据摄像头持续采集击球动作不稳定。
- 在分析中增加与上一次动作的对比，方便用户对比根据建议改进的效果。
- 大模型给出的建议过于空泛和雷同，缺乏针对性。

根据改进建议，有以下研究内容：
- 重新训练st-gcn，目前的st-gcn是根据输入的30帧骨骼序列进行分类，当识别出动作时，动作序列中开头的帧可能并不是与击球动作相关的，导致切片错误率较高。在后续训练中应该针对这一点进行优化，严格要求包含完整动作才输出识别为一个击球动作。（需要根据已有数据集进行数据增广，增加更多的负样本，具体效果未知，优先级中）
- 在系统中增加对用户击球动作的存档，对于用户使用了AI分析过的动作标记为对比动作，在接下来的AI分析中增加与之前击球动作的对比。（实现较为简单且网页前端部分更多，优先级最低）
- 目前方案为固定prompt+切片视频数据发送至Gemini3.1-pro-preview中进行分析，从模型的输出可以看出模型对网球知识了解不深入，只会输出空泛且重复的建议。因此需要一个本地微调大模型+本地语料库RAG，增强模型对网球动作理解能力和改进建议的丰富程度。（优先级最高）


20260331: 经过简单调研，目前选取GitHub上star数量最高的开源RAG项目[ragflow](https://github.com/infiniflow/ragflow)，本地大模型部署框架选用Ollama，大模型选用qwen3.5-122B，在Ollama框架下显存占用为91G左右，仍留有一定冗余为yolo骨骼识别和st-gcn推理。目前遇到以下问题：
- [✅]为ubuntu安装docker时需留意权限问题，使用docker部署ragflow时务必使用普通账号权限，否则会调用系统级的nginx服务。这里若系统中未安装nginx，则会报错找不到指定目录文件等，若以安装并开启了nginx服务，则会报错对应端口已被占用，容器启动失败。底层原因是官方容器中已准备了对应的nginx服务相关的文件，需使用普通账号权限才能正确调用。若使用sudo docker命令执行，则会调用系统级的nginx，导致报错。（已解决）
- [ ]docker容器访问宿主机的网络服务问题。由于ragflow是使用docker部署，大模型在宿主机使用ollama框架部署，正常访问模型使用http://localhost:11434/api/chat即可，但docker容器中访问需要使用http://host.docker.internal:11434/api/chat访问，目前出现位置状况仍无法在容器中正常访问。（待解决）

20260401: 针对docker容器访问ollama问题使用以下解决方案：令ollama监听0.0.0.0，以便容器可以根据宿主机ip直接访问。因此需要为ollama创建配置文件，执行以下命令：
```
# 创建配置文件夹
sudo mkdir -p /etc/systemd/system/ollama.service.d
# 创建配置文件并写入
echo -e "[Service]\nEnvironment="OLLAMA_HOST=0.0.0.0"" | sudo tee /etc/systemd/system/ollama.service.d/override.conf
# 重新加载服务配置
sudo systemctl daemon-reload
# 重新启动ollama
sudo systemctl restart ollama
```
验证：
```
# 切换到容器终端
sudo docker exec -it docker-ragflow-gpu-1 bash
# 尝试访问宿主机的ollama服务
curl http://{宿主机IP}:11434
```
若得到`Ollama is running`则说明配置成功。