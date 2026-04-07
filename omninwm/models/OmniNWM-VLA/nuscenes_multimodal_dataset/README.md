# nuScenes Multimodal Dataset Builder

A complete toolkit for building multimodal dialogue datasets for Vision-Language Model (VLM) training based on the nuScenes dataset, focusing on the ShareGPT dialogue format while maintaining compatibility with legacy JSON formats.

## 🎯 Project Overview

This project converts the raw nuScenes autonomous driving dataset into a multimodal dialogue dataset suitable for VLM fine-tuning. The generated dataset contains:

- **RGB Images**: Raw images from 6 cameras (360° coverage)
- **Depth Maps**: Depth estimation generated using ZoeDepth/MiDaS
- **Semantic Segmentation Maps**: Semantic annotations generated using SegFormer
- **High-Frequency Trajectory**: 3-second future trajectory sampled at 12Hz (36 waypoints)
- **Historical States**: Past 1-second vehicle states containing CAN bus data (12 historical points)
- **Dialogue Format**: Multimodal dialogue data supporting the ShareGPT standard

## ✨ New Features

### 🚀 ShareGPT Dialogue Format Support
- Fully compatible with ShareGPT standard dialogue data format
- Supports multimodal input (RGB + Depth Map + Semantic Segmentation Map)
- User-Assistant dialogue structure, suitable for VLM dialogue training

### 📊 12Hz High-Frequency Trajectory
- Upgraded from nuScenes native 2Hz to 12Hz sampling
- 3-second prediction horizon, 36 precise waypoints
- High-precision trajectory interpolation algorithm

### 🚗 CAN Bus Data Integration
- Historical vehicle state recording (acceleration, speed, steering angle)
- 1-second historical data, 12Hz sampling
- Automatic fallback to kinematic estimation (when CAN data is unavailable)

## 🚀 Quick Start

### Requirements

- Python 3.7+
- CUDA 11.0+ (Recommended for GPU acceleration)
- At least 100GB available disk space
- 16GB+ RAM (32GB recommended)

### Installation

1. **Clone the Project**
```bash
git clone <repository-url>
cd nuscenes_multimodal_dataset
```

2. **Install Dependencies**
```bash
pip install -r requirements.txt
```

3. **Download nuScenes Dataset**

Download and extract the dataset from the [nuScenes website](https://www.nuscenes.org/download):
```
/path/to/nuscenes/
├── maps/
├── samples/
├── sweeps/
└── v1.0-trainval/
```

### Configuration

1. **Edit Configuration File**
```bash
cp configs/sharegpt_dataset_config.yaml configs/my_config.yaml
# Edit my_config.yaml to set the correct data paths
```

2. **Key Configuration Options**
```yaml
dataset:
  nuscenes_dataroot: "/path/to/nuscenes"     # nuScenes data path
  output_directory: "/path/to/output"        # Output directory

processing:
  num_workers: 8                             # Number of parallel workers
  batch_size: 4                              # Batch size
```

### Usage

#### ShareGPT Dialogue Format Dataset (Recommended)
```bash
# Basic usage - Generate ShareGPT format dataset (Recommended)
python scripts/build_sharegpt_dataset.py \
    --config configs/sharegpt_dataset_config.yaml

# Test run (Process only 10 samples)
python scripts/build_sharegpt_dataset.py \
    --config configs/sharegpt_dataset_config.yaml \
    --max-samples 10

# Start processing from a specific scene
python scripts/build_sharegpt_dataset.py \
    --config configs/sharegpt_dataset_config.yaml \
    --start-scene 5
```

#### Server Environment Quick Start
```bash
# Use the provided Shell script
chmod +x run_sharegpt_generation.sh
./run_sharegpt_generation.sh

# Run in background and log output
nohup ./run_sharegpt_generation.sh > generation.log 2>&1 &
```

#### Legacy JSON Format Dataset (Compatibility)
```bash
# Generate legacy JSON format (Backwards compatible)
python scripts/build_dataset.py \
    --config configs/dataset_config.yaml \
    --output /path/to/output \
    --max-samples 100
```

## 📁 Project Structure

```
nuscenes_multimodal_dataset/
├── src/                           # Core source code
│   ├── core/                      # Core data processing modules
│   │   ├── nuscenes_reader.py     # nuScenes data reader (supports CAN bus)
│   │   ├── coordinate_transform.py # Coordinate transformation
│   │   └── trajectory_calculator.py # Trajectory calculator (12Hz interpolation)
│   ├── processors/                # Data processors
│   │   ├── depth_processor.py     # Depth map generation (ZoeDepth/MiDaS)
│   │   ├── semantic_processor.py  # Semantic segmentation (SegFormer)
│   │   ├── multimodal_processor.py # Multimodal integration
│   │   └── modality_fallback.py   # Modality failure fallback processing
│   ├── generators/                # Data generators
│   │   ├── conversation_prompt_generator.py # Conversational prompt generation (Main)
│   │   ├── sharegpt_formatter.py # ShareGPT formatting (Main)
│   │   ├── prompt_generator.py    # Legacy instruction text generation (Compatible)
│   │   └── json_formatter.py     # Legacy JSON formatting (Compatible)
│   ├── utils/                     # Utility modules
│   │   ├── file_utils.py         # File operation utilities
│   │   ├── math_utils.py         # Math calculation utilities
│   │   └── logging_utils.py      # Logging utilities
│   └── config/                    # Configuration management
│       └── config_manager.py     # Configuration manager
├── scripts/                       # Execution scripts
│   ├── build_sharegpt_dataset.py # ShareGPT format build script (Main)
│   ├── build_dataset.py          # Legacy JSON format build script (Compatible)
│   └── run_sharegpt_generation.sh # Server environment startup script
├── configs/                       # Configuration files
│   └── sharegpt_dataset_config.yaml # ShareGPT format configuration (Main)
├── requirements.txt               # Python dependencies
├── setup.py                       # Installation configuration
└── README.md                     # This document
```

## 🔧 Core Features

### 1. Multi-Format Data Generation
- **ShareGPT Dialogue Format**: Main feature, suitable for conversational VLM training, fully compatible with ShareGPT standards
- **Legacy JSON Format**: Backwards compatible, compatible with existing VLM training pipelines

### 2. High-Frequency Trajectory Prediction
- **12Hz High-Frequency Sampling**: Upgraded from nuScenes native 2Hz to 12Hz
- **Smart Interpolation Algorithm**: Cubic spline interpolation ensures trajectory smoothness
- **Angle Handling**: Dedicated algorithm handles -π/π boundary issues for heading angles

### 3. nuScenes Data Processing
- Automatically parses nuScenes relational data structures
- Extracts synchronized image data from 6 cameras (360° panoramic coverage)
- Obtains precise ego-vehicle pose information
- **New**: CAN bus data integration, including vehicle dynamic states

### 2. Coordinate Transformation
- Precise transformation from global coordinate system to ego-vehicle coordinate system
- Quaternion to rotation matrix conversion
- Heading angle calculation and normalization

### 3. Trajectory Calculation
- Calculates future 3-second trajectory based on continuous samples
- 12Hz high-frequency sampling rate, generating 36 waypoints
- Auto-verifies trajectory integrity

### 4. Multimodal Data Generation
- **Depth Maps**: Generated using ZoeDepth or MiDaS
- **Semantic Segmentation**: Generated using SegFormer or Mask2Former
- Supports batch processing and GPU acceleration

### 5. Instruction Data Formatting
- Supports multiple prompt templates (Basic, Chain of Thought, Roleplay)
- Generates JSON data conforming to Qwen2.5-VL format
- Automatic data validation and quality control

## 📊 Output Formats

### ShareGPT Format Dataset Structure (Main Output):
```
output_directory/
├── sharegpt_format/
│   ├── conversation_xxxxx.json    # ShareGPT format dialogue data
│   ├── conversation_yyyyy.json
│   └── sharegpt_manifest.json     # Dataset manifest
├── depth/                         # Depth map files
│   └── samples/
│       └── CAM_FRONT/
│           └── xxxxx_depth.png
├── semantic/                      # Semantic segmentation map files
│   └── samples/
│       └── CAM_FRONT/
│           └── xxxxx_semantic.png
└── logs/                          # Processing logs
    └── sharegpt_generation_*.log
```

### Legacy Format Dataset Structure (Compatibility):
```
output_directory/
├── processed_dataset/
│   ├── scene_xxxx/
│   │   ├── sample_yyyy/
│   │   │   ├── CAM_FRONT.jpg
│   │   │   ├── CAM_FRONT_depth.png
│   │   │   ├── CAM_FRONT_semantic.png
│   │   │   ├── ... (other 5 cameras)
│   │   │   └── prompt.json
│   │   └── ...
│   └── ...
├── dataset_manifest.json
├── statistics.json
└── class_mapping.json
```

### ShareGPT Format Example (Main Output)
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

### Legacy JSON Format Example (Compatibility)
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

## ⚙️ Advanced Configuration

### Depth Estimation Model Configuration
```yaml
depth:
  model_name: "ZoeDepth"    # or "MiDaS"
  device: "auto"            # "cuda", "cpu", or "auto"
  save_format: "png16"      # "png16", "png8", or "npy"
  max_depth: 100.0
```

### Semantic Segmentation Model Configuration
```yaml
semantic:
  model_name: "SegFormer"   # or "Mask2Former"
  device: "auto"
  save_colored: false       # Whether to save colored version
```

### Parallel Processing Configuration
```yaml
processing:
  num_workers: 8            # Number of parallel workers
  batch_size: 4             # Batch size
  enable_validation: true   # Enable data validation
```

## 🔍 Data Validation and Quality Control

### Automated Validation
- Image file integrity check
- Trajectory data format validation
- JSON schema compliance check
- Coordinate boundary validation

### Quality Statistics
```bash
# View the generated dataset manifest
cat output_directory/sharegpt_format/sharegpt_manifest.json

# Check the number of generated samples
find output_directory/sharegpt_format -name "conversation_*.json" | wc -l

# View processing logs
tail -f output_directory/logs/sharegpt_generation_*.log
```

## 🧪 Testing

```bash
# Basic function test (Generate a few samples)
python scripts/build_sharegpt_dataset.py \
    --config configs/sharegpt_dataset_config.yaml \
    --max-samples 5

# Verify component imports
python -c "from src.core.nuscenes_reader import NuScenesReader; print('Import successful')"
python -c "from src.generators.sharegpt_formatter import ShareGPTFormatter; print('Import successful')"

# Check configuration file format
python -c "import yaml; yaml.safe_load(open('configs/sharegpt_dataset_config.yaml')); print('Configuration format correct')"
```

## 📖 Advanced Usage

### Custom Prompt Templates
```python
from src.generators.conversation_prompt_generator import ConversationPromptGenerator

# Create conversation prompt generator
generator = ConversationPromptGenerator()

# Generate custom conversation prompt
user_prompt, assistant_response = generator.generate_conversation_prompt(
    template_name="multimodal_trajectory",
    historical_states=[],  # Historical state data
    future_trajectory=[]   # Future trajectory data
)
```

### Batch Processing Multiple Scenes
```bash
# Start processing from a specific scene
python scripts/build_sharegpt_dataset.py \
    --config configs/sharegpt_dataset_config.yaml \
    --start-scene 10

# Limit the number of processed samples
python scripts/build_sharegpt_dataset.py \
    --config configs/sharegpt_dataset_config.yaml \
    --max-samples 1000
```

### Server Environment Deployment
```bash
# Modify the paths in the configuration file to server paths
# Then use the server script
./run_sharegpt_generation.sh

# Or run in background
nohup ./run_sharegpt_generation.sh > generation.log 2>&1 &
```

## 🚨 Troubleshooting

### Common Issues

1. **Insufficient Memory**
   - Decrease `batch_size` and `num_workers`
   - Use smaller image dimensions

2. **Insufficient GPU Memory**
   - Use mixed precision
   - Set `device: "cpu"` to use CPU processing
   - Decrease batch size

3. **Insufficient Disk Space**
   - Clean up temporary files
   - Use higher image compression rates

4. **Dependency Issues**
   - Check CUDA version compatibility
   - Reinstall PyTorch

### Logs and Debugging
```bash
# ShareGPT generation process automatically creates log files
# View real-time logs
tail -f sharegpt_generation_*.log

# Check generation status
ls -la output_directory/sharegpt_format/

# Check processing progress
find output_directory/sharegpt_format -name "conversation_*.json" | wc -l
```

## 🤝 Contribution Guidelines

1. Fork the project
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [nuScenes](https://www.nuscenes.org/) Dataset
- [ZoeDepth](https://github.com/isl-org/ZoeDepth) Depth Estimation
- [SegFormer](https://huggingface.co/nvidia/segformer-b5-finetuned-cityscapes-1024-1024) Semantic Segmentation
- [Qwen2.5-VL](https://github.com/QwenLM/Qwen2.5) Vision-Language Model

 
----

**Note**: The dataset generated by this tool is for research purposes only. Please comply with the nuScenes dataset license terms when using it.
