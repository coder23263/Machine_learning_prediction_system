from matplotlib import pyplot as plt
import random
import matplotlib
from matplotlib import font_manager

#windws和linux设置字体的方式
font = {'family' : 'MicroSoft YaHei',
        'weight': 'bold'}

matplotlib.rc("font",**font)
matplotlib.rc("font",family='MicroSoft YaHei',weight="bold")

#另外一种设置字体的方式
# my_font = font_manager.FontProperties(fname="/System/Library/Fonts/PingFang.ttc")
from matplotlib import pyplot as plt
import random
#from matplotlib import font_manage


x = range(1,6)
y = [0.647,0.640,0.816,0.878,0.799]#中等数据集
y_little = [1, 0.987, 1.0, 0.976, 0.903]
y_big = [0.706, 0.706, 0.817, 0.738, 0.886]
plt.figure(figsize=(20,8),dpi=80)

plt.plot(x,y_little, label="小数据集",color="blue",linestyle=":", linewidth=2, alpha=0.5)
plt.plot(x,y, label="中数据集",color="red",linewidth=5, alpha=0.5)
plt.plot(x,y_big, label="大数据集",color="black",linestyle="--", linewidth=10, alpha=0.5)

#调整x轴的刻度
_xtick_labels = ["线性回归", "岭回归", "多项式回归", "决策树回归", "DNN"]
# _xtick_labels += ["11点{}分".format(i) for i in range(60)]
#取步长，数字和字符串一一对应，数据的长度一样
plt.xticks(list(x)[::1],_xtick_labels[::1],rotation=45) #rotaion旋转的度数
plt.yticks([i/10 for i in range(11)])
#添加描述信息
plt.xlabel("算法")
plt.ylabel("评价指标R2：[0-1]")
plt.title("不同算法在不同数据集的表现")

#绘制网格
plt.grid(alpha=0.4,linestyle=':')

#添加图例
plt.legend(prop=font, loc="upper left")

plt.show()