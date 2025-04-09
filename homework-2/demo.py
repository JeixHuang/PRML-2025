import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV
import pandas as pd
import seaborn as sns

# 生成3D make-moons数据
def make_moons_3d(n_samples=500, noise=0.1, random_state=None):
    # 设置随机种子以确保可重复性
    if random_state is not None:
        np.random.seed(random_state)
    
    # 生成原始2D make_moons数据
    t = np.linspace(0, 2 * np.pi, n_samples)
    x = 1.5 * np.cos(t)
    y = np.sin(t)
    z = np.sin(2 * t)  
    

    X = np.vstack([np.column_stack([x, y, z]), np.column_stack([-x, y - 1, -z])])
    y = np.hstack([np.zeros(n_samples), np.ones(n_samples)])
    
    # 添加高斯噪声
    X += np.random.normal(scale=noise, size=X.shape)
    
    return X, y

# 可视化3D数据
def plot_3d_data(X, y, title):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    scatter = ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=y, cmap='viridis', marker='o', s=30)
    legend1 = ax.legend(*scatter.legend_elements(), title="Classes")
    ax.add_artist(legend1)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.title(title)
    plt.tight_layout()
    plt.savefig(f"{title.replace(' ', '_')}.png")
    plt.show()

# 生成训练数据（1000个数据点，C0和C1类各500个）
X_train, y_train = make_moons_3d(n_samples=500, noise=0.2, random_state=42)

# 生成测试数据（500个数据点，C0和C1类各250个）
X_test, y_test = make_moons_3d(n_samples=250, noise=0.2, random_state=43)

# 可视化训练集和测试集
plot_3d_data(X_train, y_train, '3D Make Moons - Training Set')
plot_3d_data(X_test, y_test, '3D Make Moons - Test Set')

# 创建评估函数
def evaluate_model(model, X_train, y_train, X_test, y_test, model_name):
    # 训练模型
    model.fit(X_train, y_train)
    
    # 在测试集上预测
    y_pred = model.predict(X_test)
    
    # 计算准确率
    accuracy = accuracy_score(y_test, y_pred)
    
    # 打印分类报告
    print(f"\n===== {model_name} =====")
    print(f"Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # 混淆矩阵
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Class 0', 'Class 1'], 
                yticklabels=['Class 0', 'Class 1'])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(f'Confusion Matrix - {model_name}')
    plt.tight_layout()
    plt.savefig(f"Confusion_Matrix_{model_name.replace(' ', '_')}.png")
    plt.show()
    
    return accuracy

# 1. 决策树
dt_model = DecisionTreeClassifier(random_state=42)
dt_accuracy = evaluate_model(dt_model, X_train, y_train, X_test, y_test, "Decision Tree")

# 2. AdaBoost + 决策树
try:
    ada_model = AdaBoostClassifier(
        estimator=DecisionTreeClassifier(max_depth=3),
        n_estimators=100,
        random_state=42
    )
except TypeError:
    ada_model = AdaBoostClassifier(
        base_estimator=DecisionTreeClassifier(max_depth=3),
        n_estimators=100,
        random_state=42
    )
ada_accuracy = evaluate_model(ada_model, X_train, y_train, X_test, y_test, "AdaBoost + Decision Tree")

# 3. SVM - 线性核
svm_linear = SVC(kernel='linear', random_state=42)
svm_linear_accuracy = evaluate_model(svm_linear, X_train, y_train, X_test, y_test, "SVM - Linear Kernel")

# 4. SVM - 多项式核
svm_poly = SVC(kernel='poly', degree=3, random_state=42)
svm_poly_accuracy = evaluate_model(svm_poly, X_train, y_train, X_test, y_test, "SVM - Polynomial Kernel")

# 5. SVM - RBF核
svm_rbf = SVC(kernel='rbf', random_state=42)
svm_rbf_accuracy = evaluate_model(svm_rbf, X_train, y_train, X_test, y_test, "SVM - RBF Kernel")

# 比较所有模型的准确率
models = ['Decision Tree', 'AdaBoost + DT', 'SVM-Linear', 'SVM-Polynomial', 'SVM-RBF']
accuracies = [dt_accuracy, ada_accuracy, svm_linear_accuracy, svm_poly_accuracy, svm_rbf_accuracy]

plt.figure(figsize=(12, 6))
plt.bar(models, accuracies, color=['blue', 'green', 'red', 'purple', 'orange'])
plt.title('Classification Accuracy Comparison')
plt.xlabel('Models')
plt.ylabel('Accuracy')
plt.ylim(0, 1.0)

# 在条形上显示准确率值
for i, acc in enumerate(accuracies):
    plt.text(i, acc + 0.01, f'{acc:.4f}', ha='center')

plt.tight_layout()
plt.savefig("Classification_Accuracy_Comparison.png")
plt.show()

param_grid = {
    'C': [0.1, 1, 10, 100],
    'gamma': ['scale', 'auto', 0.1, 0.01, 0.001]
}

grid_search = GridSearchCV(SVC(kernel='rbf', random_state=42), param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)

print("\n===== SVM-RBF Grid Search Results =====")
print(f"Best parameters: {grid_search.best_params_}")
print(f"Best cross-validation score: {grid_search.best_score_:.4f}")

# 使用最佳参数的模型
best_svm = grid_search.best_estimator_
best_svm_accuracy = evaluate_model(best_svm, X_train, y_train, X_test, y_test, "SVM-RBF (Optimized)")

# 最终比较
models.append('SVM-RBF (Opt)')
accuracies.append(best_svm_accuracy)

plt.figure(figsize=(14, 6))
bar_colors = ['blue', 'green', 'red', 'purple', 'orange', 'brown']
plt.bar(models, accuracies, color=bar_colors)
plt.title('Classification Accuracy Comparison (Including Optimized Model)')
plt.xlabel('Models')
plt.ylabel('Accuracy')
plt.ylim(0, 1.0)

# 在条形上显示准确率值
for i, acc in enumerate(accuracies):
    plt.text(i, acc + 0.01, f'{acc:.4f}', ha='center')

plt.tight_layout()
plt.savefig("Classification_Accuracy_Comparison_with_Optimized.png")
plt.show()

# 创建分类决策边界可视化函数（只针对前两个维度）
def plot_decision_boundary(X, y, models, model_names):
    # 创建网格点
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                         np.arange(y_min, y_max, 0.1))
    
    # 使用数据集的平均z值
    z_mean = np.mean(X[:, 2])
    
    plt.figure(figsize=(20, 15))
    
    for i, (name, model) in enumerate(zip(model_names, models)):
        plt.subplot(2, 3, i+1)
        
        # 在网格上进行预测
        Z = model.predict(np.c_[xx.ravel(), yy.ravel(), 
                                np.ones(xx.ravel().shape) * z_mean])
        Z = Z.reshape(xx.shape)
        
        # 绘制决策边界
        plt.contourf(xx, yy, Z, alpha=0.3, cmap='viridis')
        
        # 绘制数据点
        scatter = plt.scatter(X[:, 0], X[:, 1], c=y, 
                             edgecolors='k', cmap='viridis')
        
        plt.title(name)
        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
    
    plt.tight_layout()
    plt.savefig("Decision_Boundaries.png")
    plt.show()

# 准备所有模型列表用于决策边界可视化
all_models = [dt_model, ada_model, svm_linear, svm_poly, svm_rbf, best_svm]
all_model_names = ['Decision Tree', 'AdaBoost + DT', 'SVM-Linear', 
                  'SVM-Polynomial', 'SVM-RBF', 'SVM-RBF (Optimized)']

# 绘制决策边界
plot_decision_boundary(X_test, y_test, all_models, all_model_names)

# 计算并展示所有SVM模型的支持向量数量
print("\n===== SVM 支持向量数量比较 =====")
svm_models = [svm_linear, svm_poly, svm_rbf, best_svm]
svm_names = ['SVM-Linear', 'SVM-Polynomial', 'SVM-RBF', 'SVM-RBF (Optimized)']
sv_counts = [model.n_support_.sum() for model in svm_models]

plt.figure(figsize=(10, 6))
plt.bar(svm_names, sv_counts, color=['red', 'purple', 'orange', 'brown'])
plt.title('Number of Support Vectors for Different SVM Models')
plt.xlabel('SVM Models')
plt.ylabel('Number of Support Vectors')

for i, count in enumerate(sv_counts):
    plt.text(i, count + 5, str(count), ha='center')

plt.tight_layout()
plt.savefig("SVM_Support_Vectors_Count.png")
plt.show()
