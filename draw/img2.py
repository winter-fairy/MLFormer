from matplotlib import pyplot as plt

# 更新数据和标签
values = [11.2, 82.9, 92.1]
labels = ['ViT', 'ResNet', 'Pretrained ViT']
ylabel = 'mAP'


bar_width = 0.4
colors = ['#a6cee3', '#1f78b4', '#b2df8a']

# 创建条形图
plt.figure(figsize=(8, 4))
bars = plt.bar(labels, values, color=colors, width=bar_width)

# 添加数据标签
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval, round(yval, 1), va='bottom', ha='center')

# 设置标题和标签
plt.xlabel('model name')
plt.ylabel(ylabel)

# 展示图表
plt.show()
