# nuScenes多模态数据集构建器

基于nuScenes数据集构建用于视觉语言模型(VLM)训练的多模态对话数据集的完整工具包，专注于ShareGPT对话格式，同时保持对传统JSON格式的兼容性。

## 🎯 项目概述

本项目将原始的nuScenes自动驾驶数据集转换为适合视觉语言模型微调的多模态对话数据集。生成的数据集包含：

- **RGB图像**: 来自6个摄像头的原始图像(360°覆盖)
- **深度图**: 使用ZoeDepth/MiDaS生成的深度估计
- **语义分割图**: 使用SegFormer生成的语义标注
- **高频轨迹**: 12Hz采样的3秒未来轨迹(36个路径点)
- **历史状态**: 包含CAN总线数据的过去1秒车辆状态(12个历史点)
- **对话格式**: 支持ShareGPT标准的多模态对话数据

## ✨ 新特性

### 🚀 ShareGPT对话格式支持
- 完全兼容ShareGPT标准的对话数据格式
- 支持多模态输入(RGB + 深度图 + 语义分割图)
- 用户-助手对话结构，适合VLM对话训练

### 📊 12Hz高频轨迹
- 从nuScenes原生2Hz升级到12Hz采样
- 3秒预测时域，36个精确路径点
- 高精度轨迹插值算法

### 🚗 CAN总线数据集成
- 历史车辆状态记录(加速度、速度、转向角)
- 1秒历史数据，12Hz采样
- 自动回退到运动学估算(当CAN数据不可用时)

## 🚀 快速开始

### 环境要求

- Python 3.7+
- CUDA 11.0+ (推荐，用于GPU加速)
- 至少100GB可用磁盘空间
- 16GB+ RAM (推荐32GB)

### 安装

1. **克隆项目**
```bash
git clone <repository-url>
cd nuscenes_multimodal_dataset
```

2. **安装依赖**
```bash
pip install -r requirements.txt
```

3. **下载nuScenes数据集**

从[nuScenes官网](https://www.nuscenes.org/download)下载并解压数据集：
```
/path/to/nuscenes/
├── maps/
├── samples/
├── sweeps/
└── v1.0-trainval/
```

### 配置

1. **编辑配置文件**
```bash
cp configs/sharegpt_dataset_config.yaml configs/my_config.yaml
# 编辑 my_config.yaml，设置正确的数据路径
```

2. **关键配置项**
```yaml
dataset:
  nuscenes_dataroot: "/path/to/nuscenes"     # nuScenes数据路径
  output_directory: "/path/to/output"        # 输出目录

processing:
  num_workers: 8                             # 并行进程数
  batch_size: 4                              # 批处理大小
```

### 运行

#### ShareGPT对话格式数据集（推荐）
```bash
# 基本用法 - 生成ShareGPT格式数据集（推荐）
python scripts/build_sharegpt_dataset.py \
    --config configs/sharegpt_dataset_config.yaml

# 测试运行（只处理10个样本）
python scripts/build_sharegpt_dataset.py \
    --config configs/sharegpt_dataset_config.yaml \
    --max-samples 10

# 从特定场景开始处理
python scripts/build_sharegpt_dataset.py \
    --config configs/sharegpt_dataset_config.yaml \
    --start-scene 5
```

#### 服务器环境快速启动
```bash
# 使用提供的Shell脚本
chmod +x run_sharegpt_generation.sh
./run_sharegpt_generation.sh

# 后台运行并记录日志
nohup ./run_sharegpt_generation.sh > generation.log 2>&1 &
```

#### 传统JSON格式数据集（兼容性）
```bash
# 生成传统JSON格式（向后兼容）
python scripts/build_dataset.py \
    --config configs/dataset_config.yaml \
    --output /path/to/output \
    --max-samples 100
```

## 📁 项目结构

```
nuscenes_multimodal_dataset/
├── src/                           # 核心源代码
│   ├── core/                      # 核心数据处理模块
│   │   ├── nuscenes_reader.py     # nuScenes数据读取器(支持CAN总线)
│   │   ├── coordinate_transform.py # 坐标系变换
│   │   └── trajectory_calculator.py # 轨迹计算器(12Hz插值)
│   ├── processors/                # 数据处理器
│   │   ├── depth_processor.py     # 深度图生成(ZoeDepth/MiDaS)
│   │   ├── semantic_processor.py  # 语义分割(SegFormer)
│   │   ├── multimodal_processor.py # 多模态整合
│   │   └── modality_fallback.py   # 模态失败降级处理
│   ├── generators/                # 数据生成器
│   │   ├── conversation_prompt_generator.py # 对话式提示生成(主要)
│   │   ├── sharegpt_formatter.py # ShareGPT格式化(主要)
│   │   ├── prompt_generator.py    # 传统指令文本生成(兼容)
│   │   └── json_formatter.py     # 传统JSON格式化(兼容)
│   ├── utils/                     # 工具模块
│   │   ├── file_utils.py         # 文件操作工具
│   │   ├── math_utils.py         # 数学计算工具
│   │   └── logging_utils.py      # 日志工具
│   └── config/                    # 配置管理
│       └── config_manager.py     # 配置管理器
├── scripts/                       # 执行脚本
│   ├── build_sharegpt_dataset.py # ShareGPT格式构建脚本(主要)
│   ├── build_dataset.py          # 传统JSON格式构建脚本(兼容)
│   └── run_sharegpt_generation.sh # 服务器环境启动脚本
├── configs/                       # 配置文件
│   └── sharegpt_dataset_config.yaml # ShareGPT格式配置(主要)
├── requirements.txt               # Python依赖包
├── setup.py                       # 安装配置
└── README.md                     # 本文档
```

## 🔧 核心功能

### 1. 多格式数据生成
- **ShareGPT对话格式**: 主要功能，适用于对话式VLM训练，完全兼容ShareGPT标准
- **传统JSON格式**: 向后兼容，兼容现有VLM训练流程

### 2. 高频轨迹预测
- **12Hz高频采样**: 从nuScenes原生2Hz升级至12Hz
- **智能插值算法**: 三次样条插值确保轨迹平滑性
- **角度处理**: 专用算法处理heading角度的-π/π边界问题

### 3. nuScenes数据处理
- 自动解析nuScenes的关系型数据结构
- 提取6个摄像头的同步图像数据(360°全景覆盖)
- 获取精确的自车位姿信息
- **新增**: CAN总线数据集成，包含车辆动态状态

### 2. 坐标系变换
- 全局坐标系到自车坐标系的精确变换
- 四元数到旋转矩阵的转换
- 航向角计算和归一化

### 3. 轨迹计算
- 基于连续样本计算未来3秒轨迹
- 12Hz高频采样率，生成36个路径点
- 自动验证轨迹完整性

### 4. 多模态数据生成
- **深度图**: 使用ZoeDepth或MiDaS生成
- **语义分割**: 使用SegFormer或Mask2Former生成
- 支持批量处理和GPU加速

### 5. 指令数据格式化
- 支持多种提示模板（基础、思维链、角色扮演）
- 生成符合Qwen2.5-VL格式的JSON数据
- 自动数据验证和质量控制

## 📊 输出格式

### ShareGPT格式数据集结构（主要输出）：
```
output_directory/
├── sharegpt_format/
│   ├── conversation_xxxxx.json    # ShareGPT格式对话数据
│   ├── conversation_yyyyy.json
│   └── sharegpt_manifest.json     # 数据集清单
├── depth/                         # 深度图文件
│   └── samples/
│       └── CAM_FRONT/
│           └── xxxxx_depth.png
├── semantic/                      # 语义分割图文件
│   └── samples/
│       └── CAM_FRONT/
│           └── xxxxx_semantic.png
└── logs/                          # 处理日志
    └── sharegpt_generation_*.log
```

### 传统格式数据集结构（兼容性）：
```
output_directory/
├── processed_dataset/
│   ├── scene_xxxx/
│   │   ├── sample_yyyy/
│   │   │   ├── CAM_FRONT.jpg
│   │   │   ├── CAM_FRONT_depth.png
│   │   │   ├── CAM_FRONT_semantic.png
│   │   │   ├── ... (其他5个摄像头)
│   │   │   └── prompt.json
│   │   └── ...
│   └── ...
├── dataset_manifest.json
├── statistics.json
└── class_mapping.json
```

### ShareGPT格式示例（主要输出）
```json
{
  "id": "scene001_sample001",
  "images": [
    "samples/CAM_FRONT/xxxxx.jpg",
    "samples/CAM_FRONT_LEFT/xxxxx.jpg",
    "samples/CAM_FRONT_RIGHT/xxxxx.jpg",
    "samples/CAM_BACK/xxxxx.jpg",
    "samples/CAM_BACK_LEFT/xxxxx.jpg",
    "samples/CAM_BACK_RIGHT/xxxxx.jpg"
  ],
  "depth_maps": [
    "depth/samples/CAM_FRONT/xxxxx_depth.png",
    "depth/samples/CAM_FRONT_LEFT/xxxxx_depth.png",
    "depth/samples/CAM_FRONT_RIGHT/xxxxx_depth.png",
    "depth/samples/CAM_BACK/xxxxx_depth.png",
    "depth/samples/CAM_BACK_LEFT/xxxxx_depth.png",
    "depth/samples/CAM_BACK_RIGHT/xxxxx_depth.png"
  ],
  "semantic_maps": [
    "semantic/samples/CAM_FRONT/xxxxx_semantic.png",
    "semantic/samples/CAM_FRONT_LEFT/xxxxx_semantic.png",
    "semantic/samples/CAM_FRONT_RIGHT/xxxxx_semantic.png",
    "semantic/samples/CAM_BACK/xxxxx_semantic.png",
    "semantic/samples/CAM_BACK_LEFT/xxxxx_semantic.png",
    "semantic/samples/CAM_BACK_RIGHT/xxxxx_semantic.png"
  ],
  "messages": [
    {
      "role": "user",
      "content": "You are an autonomous driving agent. You have access to multi-modal sensory data from a vehicle's 6-camera system providing 360° coverage..."
    },
    {
      "role": "assistant",
      "content": "<PLANNING>Predicted future trajectory for the next 3 seconds (36 waypoints sampled at 12Hz, 0.083-second intervals)...</PLANNING>"
    }
  ]
}
```

### 传统JSON格式示例（兼容性）
```json
{
  "id": "scene-0001_sample-0001",
  "scene_token": "scene-token-string",
  "sample_token": "sample-token-string",
  "visual_inputs": {
    "CAM_FRONT": {
      "rgb_path": "scene_0001/sample_0001/CAM_FRONT.jpg",
      "depth_path": "scene_0001/sample_0001/CAM_FRONT_depth.png",
      "semantic_path": "scene_0001/sample_0001/CAM_FRONT_semantic.png"
    }
  },
  "text_prompt": "You are a professional autonomous driving AI...",
  "ground_truth": {
    "future_trajectory": [
      {"x": 0.5, "y": 0.01, "heading": 0.001},
      {"x": 1.0, "y": 0.02, "heading": 0.002}
    ]
  }
}
```

## ⚙️ 高级配置

### 深度估计模型配置
```yaml
depth:
  model_name: "ZoeDepth"    # 或 "MiDaS"
  device: "auto"            # "cuda", "cpu", 或 "auto"
  save_format: "png16"      # "png16", "png8", 或 "npy"
  max_depth: 100.0
```

### 语义分割模型配置
```yaml
semantic:
  model_name: "SegFormer"   # 或 "Mask2Former"
  device: "auto"
  save_colored: false       # 是否保存彩色版本
```

### 并行处理配置
```yaml
processing:
  num_workers: 8            # 并行进程数
  batch_size: 4             # 批处理大小
  enable_validation: true   # 启用数据验证
```

## 🔍 数据验证和质量控制

### 自动验证
- 图像文件完整性检查
- 轨迹数据格式验证
- JSON模式合规性检查
- 坐标边界验证

### 质量统计
```bash
# 查看生成的数据集清单
cat output_directory/sharegpt_format/sharegpt_manifest.json

# 检查生成的样本数量
find output_directory/sharegpt_format -name "conversation_*.json" | wc -l

# 查看处理日志
tail -f output_directory/logs/sharegpt_generation_*.log
```

## 🧪 测试

```bash
# 基本功能测试（生成少量样本）
python scripts/build_sharegpt_dataset.py \
    --config configs/sharegpt_dataset_config.yaml \
    --max-samples 5

# 验证组件导入
python -c "from src.core.nuscenes_reader import NuScenesReader; print('导入成功')"
python -c "from src.generators.sharegpt_formatter import ShareGPTFormatter; print('导入成功')"

# 检查配置文件格式
python -c "import yaml; yaml.safe_load(open('configs/sharegpt_dataset_config.yaml')); print('配置格式正确')"
```

## 📖 进阶用法

### 自定义提示模板
```python
from src.generators.conversation_prompt_generator import ConversationPromptGenerator

# 创建对话提示生成器
generator = ConversationPromptGenerator()

# 生成自定义对话提示
user_prompt, assistant_response = generator.generate_conversation_prompt(
    template_name="multimodal_trajectory",
    historical_states=[],  # 历史状态数据
    future_trajectory=[]   # 未来轨迹数据
)
```

### 批量处理多个场景
```bash
# 从特定场景开始处理
python scripts/build_sharegpt_dataset.py \
    --config configs/sharegpt_dataset_config.yaml \
    --start-scene 10

# 限制处理样本数量
python scripts/build_sharegpt_dataset.py \
    --config configs/sharegpt_dataset_config.yaml \
    --max-samples 1000
```

### 服务器环境部署
```bash
# 修改配置文件中的路径为服务器路径
# 然后使用服务器脚本
./run_sharegpt_generation.sh

# 或者后台运行
nohup ./run_sharegpt_generation.sh > generation.log 2>&1 &
```

## 🚨 故障排除

### 常见问题

1. **内存不足**
   - 减少 `batch_size` 和 `num_workers`
   - 使用更小的图像尺寸

2. **GPU显存不足**
   - 使用混合精度
   - 设置 `device: "cpu"` 使用CPU处理
   - 减少批处理大小

3. **磁盘空间不足**
   - 清理临时文件
   - 使用更高的图像压缩率

4. **依赖包问题**
   - 检查CUDA版本兼容性
   - 重新安装PyTorch

### 日志和调试
```bash
# ShareGPT生成过程会自动创建日志文件
# 查看实时日志
tail -f sharegpt_generation_*.log

# 检查生成状态
ls -la output_directory/sharegpt_format/

# 检查处理进度
find output_directory/sharegpt_format -name "conversation_*.json" | wc -l
```

## 🤝 贡献指南

1. Fork项目
2. 创建功能分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 开启Pull Request

## 📄 许可证

本项目采用MIT许可证 - 查看 [LICENSE](LICENSE) 文件了解详情。

## 🙏 致谢

- [nuScenes](https://www.nuscenes.org/) 数据集
- [ZoeDepth](https://github.com/isl-org/ZoeDepth) 深度估计
- [SegFormer](https://huggingface.co/nvidia/segformer-b5-finetuned-cityscapes-1024-1024) 语义分割
- [Qwen2.5-VL](https://github.com/QwenLM/Qwen2.5) 视觉语言模型

 

----

**注意**: 本工具生成的数据集仅用于研究目的。使用时请遵守nuScenes数据集的许可条款。

