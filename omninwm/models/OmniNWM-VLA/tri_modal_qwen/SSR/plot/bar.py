import matplotlib
import numpy as np
from itertools import chain
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

# 设置数据
benchmarks = ['SSRBench', 'SpatialBench']
metrics = ['SSR', 'SSR$_{PAP}$', 'Qwen2.5-VL']

# 随机示例数据 (替换为实际数据)
data = {
    "SpatialBench": [64.8, 63.6, 59.3]
    , "SSRBench": [74.5, 57.4, 55.8]
}

plt.rc("font", **{"family": "Times New Roman", "size": 18})
matplotlib.rcParams['mathtext.default'] = "regular"

# 设置图形大小
plt.figure(figsize=(7, 5))

# 定义颜色
colors = list(reversed(["#794292", "#44599B", "#40A93B"]))

# 设置柱状图的宽度和位置
x = np.arange(len(benchmarks))
width = 0.2  # 减小宽度
gap = 0.05   # 设置间隙大小
    
# 绘制柱状图，调整位置以增加间隙
positions = [
    x - width - gap,  # 第一组柱形的位置
    x,                # 第二组柱形的位置
    x + width + gap   # 第三组柱形的位置
]

for i, metric in enumerate(metrics):
    values = [data[benchmark][i] for benchmark in benchmarks]
    plt.barh(positions[i], values, width, label=metric, color=colors[i], zorder=5)

# 添加分隔线
for i in range(1, len(benchmarks)):
    plt.axhline(i - 0.5, color='black', alpha=0.3)

# 添加标签和标题
plt.xlabel('Performance', fontsize=14, fontweight='bold')
# plt.title('Multimodal Reasoning Benchmarks', fontsize=16, fontweight='bold')

# 设置y轴刻度
plt.yticks(x, benchmarks, fontsize=12, fontweight='bold')

def find_special_multiples(numbers):
    if not numbers:
        return None, None
    min_value, max_value = min(numbers), max(numbers)
    smaller_multiple = (min_value // 10) * 10
    if min_value % 10 == 0:
        smaller_multiple -= 10
    larger_multiple = ((max_value // 10) + 1) * 10
    if max_value % 10 == 0:
        larger_multiple = max_value + 10
    return smaller_multiple, larger_multiple

l, r = find_special_multiples(list(chain(*data.values())))
plt.xlim(l, r)
plt.gca().xaxis.set_major_locator(plt.MultipleLocator(10))

# 添加图例
handles, labels = plt.gca().get_legend_handles_labels()
handles = list(reversed(handles))
labels = list(reversed(labels))
plt.legend(handles=handles, labels=labels, fontsize=12, loc="upper right")

# 显示数值
for i, metric in enumerate(metrics):
    for j, benchmark in enumerate(benchmarks):
        score = data[benchmark][i]
        plt.text(score + 0.5, positions[i][j], f'{score:.1f}', 
                 va='center', fontsize=10, fontweight='bold', color='black')

# 设置网格
plt.grid(axis='x', linestyle='--', alpha=0.5, zorder=-2)

for spine in ["top", "right"]:
    plt.gca().spines[spine].set_color("none")
for spine in ["bottom", "left"]:
    plt.gca().spines[spine].set_zorder(10)

# 在绘制完柱状图和文本值后，添加提升标识
arrow_props = dict(arrowstyle='->', color='red', linewidth=1.5)

emoji_font = FontProperties(fname=r"/Users/yliu/Library/Fonts/ChocolateClassicalSans-Regular.ttf")

# 添加提升箭头及差值文本
for j, benchmark in enumerate(benchmarks):
    co_train_score = data[benchmark][0]
    zero_shot_score = data[benchmark][1]
    qwenvl_score = data[benchmark][2]
    plt.annotate("", xy=(zero_shot_score + 0.3, (positions[1][j] + positions[2][j]) / 2), xytext=(qwenvl_score, (positions[1][j] + positions[2][j]) / 2), arrowprops=arrow_props)
    plt.text(max(zero_shot_score, qwenvl_score) + 0.5, (positions[1][j] + positions[2][j]) / 2, f"\u2934{zero_shot_score - qwenvl_score:.1f}", va='center', fontproperties=emoji_font, fontsize=10, color='red')
    plt.annotate("", xy=(co_train_score + 0.3, (positions[0][j] + positions[1][j]) / 2), xytext=(qwenvl_score, (positions[0][j] + positions[1][j]) / 2), arrowprops=arrow_props)
    plt.text(max(co_train_score, qwenvl_score) + 0.5, (positions[0][j] + positions[1][j]) / 2, f"\u2934{co_train_score - qwenvl_score:.1f}", va='center', fontproperties=emoji_font, fontsize=10, color='red')

# 调整x轴限制以确保箭头和文本可见
# plt.xlim(l, r + 10)  # 略微增加x轴的右侧范围确保文本完整显示

# 调整布局并输出PDF
plt.tight_layout()
plt.savefig('bar.pdf', bbox_inches='tight')
