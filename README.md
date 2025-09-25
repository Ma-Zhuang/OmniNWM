# OmniNWM: Omniscient Navigation World Models for Autonomous Driving

[![Paper](https://img.shields.io/badge/Paper-Arxiv-red)](https://github.com/Arlo0o/OmniNWM)
[![Project Page](https://img.shields.io/badge/Project-Page-green)](https://github.com/Arlo0o/OmniNWM)
[![License](https://img.shields.io/badge/License-Apache%202.0-orange)](LICENSE)

> **OmniNWM** is a unified panoramic navigation world model that advances autonomous driving simulation by jointly generating multi-modal states (RGB, semantics, depth, 3D occupancy), enabling precise action control via normalized Plücker ray-maps, and facilitating closed-loop evaluation through occupancy-based dense rewards.

---

## 🎯 Overview

OmniNWM addresses three core dimensions of autonomous driving world models:

- **📊 State**: Joint generation of panoramic RGB, semantic, metric depth, and 3D occupancy videos
- **🎮 Action**: Precise panoramic camera control via normalized Plücker ray-maps
- **🏆 Reward**: Integrated occupancy-based dense rewards for driving compliance and safety

![Teaser](assets/teaser.png)  

---

## ✨ Key Features

| Feature | Description |
|-----------|-------------|
|  **Multi-modal Generation** | Jointly generates RGB, semantic, depth, and 3D occupancy in panoramic views |
| **Precise Camera Control** | Uses normalized Plücker ray-maps for pixel-level trajectory interpretation |
| **Long-term Stability** | Flexible forcing strategy enables auto-regressive generation beyond training sequences |
| **Closed-loop Evaluation** | Occupancy-based dense rewards enable realistic driving policy evaluation |
|**Zero-shot Generalization** | Transfers across datasets and camera configurations without fine-tuning |

---

## 🏗️ Architecture

![Architecture](assets/architecture.png)  

---

### 💥 News
- [2025/09]: Demo is released on the [Project Page](https://github.com/Arlo0o/OmniNWM).


## 📚 Citation


```bibtex
@article{
}
```


---

## 📄 License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

---

## ❤️ Acknowledgments

- Built upon great open-source projects like [OpenSora](https://github.com/hpcaitech/Open-Sora) and [Qwen-VL](https://github.com/QwenLM/Qwen-VL)


---

<div align="center">

**🌟 Star us on GitHub if you find this project helpful! 🌟**

</div>

---

*Note: This repository is under active development. Code and models will be released soon.*
