from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt

# 测试集真实标签
y_true = [1, 1, 0, 0, 1, 0]

# 模型对测试集输出的“正类概率”
y_score = [0.9, 0.8, 0.7, 0.3, 0.6, 0.2]

# 自动计算 ROC 曲线上的点
fpr, tpr, thresholds = roc_curve(y_true, y_score)

# 自动计算 AUC
auc = roc_auc_score(y_true, y_score)

# 画 ROC
plt.plot(fpr, tpr, label=f"AUC = {auc:.3f}")

# 随机猜测的基准线
plt.plot([0, 1], [0, 1], "--")

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.show()