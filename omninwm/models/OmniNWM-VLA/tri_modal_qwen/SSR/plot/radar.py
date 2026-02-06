import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

tasks = [
    "Position", "Existence", "Counting", "Size",
    "Existence", "Attribute", "Action", "Counting", "Position", "Object"
]

data = np.array([
    55.9, 80.0, 76.4, 25.0, 66.4, 58.9, 63.1, 34.1, 60.5, 51.9
    , 64.7, 80.0, 79.6, 30.0, 70.1, 59.5, 63.7, 36.6, 61.1, 53.2
    , 64.7, 80.0, 82.9, 31.7, 83.2, 82.1, 72.6, 51.2, 83.3, 74.7
]).reshape(3, -1)

labels = ["Qwen2.5-VL", "SSR$_{PAP}$", "SSR"]

# 标准化数据
data_scaled = np.zeros_like(data)
for i in range(data.shape[1]):
    col = data[:, i]
    min_val, max_val = col.min(), col.max()
    diff = max_val - min_val
    min_val -= diff
    max_val += diff * 0.1
    if max_val != min_val:
        data_scaled[:, i] = (col - min_val) / (max_val - min_val) * 100
    else:
        data_scaled[:, i] = 100
# data_scaled = data

plt.rc("font", **{"family": "Times New Roman", "size": 18})
matplotlib.rcParams['mathtext.default'] = "regular"
angles = np.linspace(0, 2 * np.pi, len(tasks), endpoint=False).tolist()
angles += angles[:1]

fig = plt.figure(figsize=(6, 6))
ax = fig.add_subplot(111, polar=True)
colors = ["#794292", "#44599B", "#40A93B", "#E97124"]

for i, (label, color) in enumerate(zip(labels, colors)):
    values = data_scaled[i].tolist()
    values += values[:1]
    ax.plot(angles, values, color=color, linewidth=2, label=label)
    ax.fill(angles, values, color=color, alpha=0.1)

ax.set_theta_offset(np.pi / 2)
ax.set_theta_direction(-1)
ax.set_xticks(angles[:-1])
ax.set_xticklabels([])
ax.set_yticklabels([])
ax.spines['polar'].set_visible(False)

radius = 105
# 内环标签
for angle, task in zip(angles[:-1], tasks):
    tangent_angle = angle + np.pi/2
    text_angle_deg = (90 - np.rad2deg(tangent_angle)) % 360
    if 90 < text_angle_deg < 270:
        text_angle_deg += 180
    ax.text(angle, radius, task,
            rotation=text_angle_deg, rotation_mode='anchor',
            horizontalalignment='center', verticalalignment='center',
            fontsize=14, fontfamily='Times New Roman')

leg = plt.legend(loc='lower center', bbox_to_anchor=(0.5, -0.2), ncol=3, prop={"family": 'Times New Roman', "size": 16})
for line in leg.get_lines():
    line.set_linewidth(10.0)

env_labels = ["SpatialBench", "SSRBench$^{G}$", "SSRBench$^{S}$"]
env_counts = [3, 3, 4]

arc_radius = 115
label_radius = 123  # 环境标签半径位置
tasks_count = len(tasks)
unit_angle_deg = 360 / tasks_count

# 计算每个task边界角度
task_angles_deg = np.arange(0, 360, unit_angle_deg)
task_angles_rad = np.deg2rad(task_angles_deg)

# 计算每个env的起始和终止边界的index
env_boundaries_idx = np.cumsum([0] + env_counts)

arc_gap_deg = 10

# 修改圆弧计算的这段代码为如下即可：
for i in range(len(env_labels)):
    start_idx = env_boundaries_idx[i] + (1 if i != 0 else 0)
    end_idx = env_boundaries_idx[i + 1] - (1 if i != len(env_labels) - 1 else 2)

    # 环境的圆弧，必须覆盖任务从左边界到该环境最后一个任务的右边界
    start_angle_deg = task_angles_deg[start_idx] - arc_gap_deg  # 第一个任务左边界
    end_angle_deg = task_angles_deg[end_idx] + unit_angle_deg + arc_gap_deg  # 最后一个任务右边界

    # 将deg转换为rad，绘制圆弧
    arc_theta = np.deg2rad(np.linspace(start_angle_deg, end_angle_deg, 100))
    ax.plot(arc_theta, [arc_radius]*len(arc_theta), lw=2, color=colors[-1], alpha=0.8)

    # 外围环境标签放在圆弧中央位置
    label_angle_deg = (start_angle_deg + end_angle_deg) / 2
    label_angle_rad = np.deg2rad(label_angle_deg % 360)

    tangent_angle = label_angle_rad + np.pi / 2
    text_angle_deg = (90 - np.rad2deg(tangent_angle)) % 360
    if 90 < text_angle_deg < 270:
        text_angle_deg += 180

    ax.text(label_angle_rad, label_radius, env_labels[i], rotation=text_angle_deg, rotation_mode='anchor', ha='center', va='center', fontweight='bold', fontsize=16, fontfamily='Times New Roman')

plt.tight_layout()
# plt.savefig(os.path.join(os.getcwd(), "radar.png"), bbox_inches="tight", dpi=600, pad_inches=0.05)
plt.savefig(os.path.join(os.getcwd(), "radar.pdf"), bbox_inches="tight", pad_inches=0.05)