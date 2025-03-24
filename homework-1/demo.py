import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from scipy.optimize import curve_fit
import argparse
import time
import os

# 确保输出目录存在
os.makedirs("plots", exist_ok=True)

# 读取数据函数
def load_data(train_path, test_path):
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    
    # 提取特征和目标变量
    train_x = train_df['x'].values.reshape(-1, 1)
    train_y = train_df['y_complex'].values
    test_x = test_df['x_new'].values.reshape(-1, 1)
    test_y = test_df['y_new_complex'].values
    
    return train_x, train_y, test_x, test_y

# 1. 最小二乘法（sklearn）
def least_squares_method(train_x, train_y, test_x, test_y, batch_size=None, epochs=10):
    print(f"Running Least Squares Method (batch_size={batch_size}, epochs={epochs})")
    
    # 由于最小二乘法通常是一次性求解的，我们模拟多个epochs来展示训练过程
    train_losses = []
    test_losses = []
    
    # 初始模型
    lr_model = LinearRegression()
    
    # 在每个epoch重新拟合模型（实际上sklearn的LinearRegression是一次性拟合的）
    lr_model.fit(train_x, train_y)
    
    # 计算训练和测试损失
    lr_train_pred = lr_model.predict(train_x)
    lr_test_pred = lr_model.predict(test_x)
    lr_train_mse = mean_squared_error(train_y, lr_train_pred)
    lr_test_mse = mean_squared_error(test_y, lr_test_pred)
    
    # 存储损失值
    train_losses.append(lr_train_mse)
    test_losses.append(lr_test_mse)
    
    # 输出训练进度
    print(f"least_squares_method , Train Loss: {lr_train_mse:.4f}, Test Loss: {lr_test_mse:.4f}")
    
    
    return lr_model, train_losses[-1], test_losses[-1], train_losses, test_losses

# 2. 梯度下降法
def gradient_descent_method(train_x, train_y, test_x, test_y, batch_size=32, epochs=1000, learning_rate=0.01, verbose_interval=50):
    print(f"Running Gradient Descent Method (batch_size={batch_size}, epochs={epochs}, learning_rate={learning_rate})")
    
    m = len(train_y)
    # 添加截距项
    X_b = np.c_[np.ones((m, 1)), train_x]
    X_b_test = np.c_[np.ones((len(test_x), 1)), test_x]
    theta = np.zeros(2)  # [intercept, slope]
    
    train_losses = []
    test_losses = []
    
    # 计算初始损失
    initial_train_pred = X_b.dot(theta)
    initial_test_pred = X_b_test.dot(theta)
    initial_train_mse = mean_squared_error(train_y, initial_train_pred)
    initial_test_mse = mean_squared_error(test_y, initial_test_pred)
    train_losses.append(initial_train_mse)
    test_losses.append(initial_test_mse)
    
    # 使用批量梯度下降
    if batch_size is None or batch_size >= m:
        # 全批量梯度下降
        for epoch in range(1, epochs + 1):
            predictions = X_b.dot(theta)
            errors = predictions - train_y
            gradients = 2/m * X_b.T.dot(errors)
            theta = theta - learning_rate * gradients
            
            # 计算当前训练和测试损失
            train_pred = X_b.dot(theta)
            test_pred = X_b_test.dot(theta)
            train_mse = mean_squared_error(train_y, train_pred)
            test_mse = mean_squared_error(test_y, test_pred)
            
            train_losses.append(train_mse)
            test_losses.append(test_mse)
            
            # 输出训练进度
            if epoch % verbose_interval == 0 or epoch == 1 or epoch == epochs:
                print(f"Epoch {epoch}/{epochs}, Train Loss: {train_mse:.4f}, Test Loss: {test_mse:.4f}")
    else:
        # 小批量梯度下降
        n_batches = int(np.ceil(m / batch_size))
        for epoch in range(1, epochs + 1):
            indices = np.random.permutation(m)
            X_b_shuffled = X_b[indices]
            y_shuffled = train_y[indices]
            
            epoch_loss = 0.0
            
            for i in range(n_batches):
                start_idx = i * batch_size
                end_idx = min(start_idx + batch_size, m)
                
                X_batch = X_b_shuffled[start_idx:end_idx]
                y_batch = y_shuffled[start_idx:end_idx]
                
                predictions = X_batch.dot(theta)
                errors = predictions - y_batch
                gradients = 2/len(X_batch) * X_batch.T.dot(errors)
                theta = theta - learning_rate * gradients
                
                # 计算批次损失
                batch_predictions = X_batch.dot(theta)
                batch_loss = mean_squared_error(y_batch, batch_predictions)
                epoch_loss += batch_loss * len(X_batch) / m
            
            # 计算整个训练集和测试集的损失
            train_pred = X_b.dot(theta)
            test_pred = X_b_test.dot(theta)
            train_mse = mean_squared_error(train_y, train_pred)
            test_mse = mean_squared_error(test_y, test_pred)
            
            train_losses.append(train_mse)
            test_losses.append(test_mse)
            
            # 输出训练进度
            if epoch % verbose_interval == 0 or epoch == 1 or epoch == epochs:
                print(f"Epoch {epoch}/{epochs}, Train Loss: {train_mse:.4f}, Test Loss: {test_mse:.4f}")
    
    # 绘制训练曲线
    plt.figure(figsize=(10, 6))
    plt.plot(range(len(train_losses)), train_losses, 'b-', label='Train Loss')
    plt.plot(range(len(test_losses)), test_losses, 'r-', label='Test Loss')
    plt.title('Gradient Descent Training Curve')
    plt.xlabel('Epoch')
    plt.ylabel('Mean Squared Error')
    plt.legend()
    plt.grid(True)
    plt.savefig('plots/gradient_descent_training_curve.png')
    
    # 如果损失值变化太大，使用对数尺度重新绘制
    if max(train_losses) / min(train_losses) > 100:
        plt.figure(figsize=(10, 6))
        plt.semilogy(range(len(train_losses)), train_losses, 'b-', label='Train Loss')
        plt.semilogy(range(len(test_losses)), test_losses, 'r-', label='Test Loss')
        plt.title('Gradient Descent Training Curve (Log Scale)')
        plt.xlabel('Epoch')
        plt.ylabel('Mean Squared Error (log scale)')
        plt.legend()
        plt.grid(True)
        plt.savefig('plots/gradient_descent_training_curve_log.png')
    
    return theta, train_losses[-1], test_losses[-1], train_losses, test_losses

# 3. 牛顿法
def newton_method(train_x, train_y, test_x, test_y, batch_size=None, epochs=10):
    print(f"Running Newton's Method (batch_size={batch_size}, epochs={epochs})")
    
    m = len(train_y)
    X_b = np.c_[np.ones((m, 1)), train_x]
    X_b_test = np.c_[np.ones((len(test_x), 1)), test_x]
    theta = np.zeros(2)
    
    train_losses = []
    test_losses = []
    
    # 计算初始损失
    initial_train_pred = X_b.dot(theta)
    initial_test_pred = X_b_test.dot(theta)
    initial_train_mse = mean_squared_error(train_y, initial_train_pred)
    initial_test_mse = mean_squared_error(test_y, initial_test_pred)
    train_losses.append(initial_train_mse)
    test_losses.append(initial_test_mse)
    
    predictions = X_b.dot(theta)
    errors = predictions - train_y
    gradients = 2/m * X_b.T.dot(errors)
    hessian = 2/m * X_b.T.dot(X_b)
    
    # 解线性方程组：hessian * delta_theta = -gradients
    delta_theta = np.linalg.solve(hessian, -gradients)
    theta = theta + delta_theta
    
    # 计算当前训练和测试损失
    train_pred = X_b.dot(theta)
    test_pred = X_b_test.dot(theta)
    train_mse = mean_squared_error(train_y, train_pred)
    test_mse = mean_squared_error(test_y, test_pred)
    
    train_losses.append(train_mse)
    test_losses.append(test_mse)
    
    # 输出训练进度
    print(f"newton_method , Train Loss: {train_mse:.4f}, Test Loss: {test_mse:.4f}")
    
    # 绘制训练曲线
    # plt.figure(figsize=(10, 6))
    # plt.plot(range(len(train_losses)), train_losses, 'b-', label='Train Loss')
    # plt.plot(range(len(test_losses)), test_losses, 'r-', label='Test Loss')
    # plt.title('Newton\'s Method Training Curve')
    # plt.xlabel('Epoch')
    # plt.ylabel('Mean Squared Error')
    # plt.legend()
    # plt.grid(True)
    # plt.savefig('plots/newton_method_training_curve.png')
    
    # 如果损失值变化太大，使用对数尺度重新绘制
    # if max(train_losses) / min(train_losses) > 100:
    #     plt.figure(figsize=(10, 6))
    #     plt.semilogy(range(len(train_losses)), train_losses, 'b-', label='Train Loss')
    #     plt.semilogy(range(len(test_losses)), test_losses, 'r-', label='Test Loss')
    #     plt.title('Newton\'s Method Training Curve (Log Scale)')
    #     plt.xlabel('Epoch')
    #     plt.ylabel('Mean Squared Error (log scale)')
    #     plt.legend()
    #     plt.grid(True)
    #     plt.savefig('plots/newton_method_training_curve_log.png')
    
    return theta, train_losses[-1], test_losses[-1], train_losses, test_losses

def plot_polynomial_curve(train_x, train_y, test_x, test_y, model, degree, train_mse, test_mse):
    """
    绘制多项式模型在数据分布上的曲线
    """
    plt.figure(figsize=(10, 6))
    plt.scatter(train_x, train_y, color='blue', label='Training Data')
    plt.scatter(test_x, test_y, color='green', label='Test Data')
    
    # 绘制预测曲线
    x_range = np.linspace(min(train_x.min(), test_x.min()), max(train_x.max(), test_x.max()), 500).reshape(-1, 1)
    y_pred = model.predict(x_range)
    
    plt.plot(x_range, y_pred, color='red', linestyle='-', 
             label=f'Polynomial Degree {degree} (Train MSE: {train_mse:.4f}, Test MSE: {test_mse:.4f})')
    
    plt.title(f'Polynomial Regression (Degree {degree})')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.grid(True)
    
    # 确保plots目录存在
    import os
    if not os.path.exists('plots'):
        os.makedirs('plots')
    
    plt.savefig(f'plots/polynomial/degree_{degree}.png')
    plt.close()
    
# 4. 多项式回归
def polynomial_regression(train_x, train_y, test_x, test_y, batch_size=None, epochs=None, max_degree=100):
    print(f"Running Polynomial Regression (batch_size={batch_size}, epochs={epochs}, max_degree={max_degree})")
    
    degrees = range(1, max_degree + 1)
    train_mse_list = []
    test_mse_list = []
    models = []
    
    for degree in degrees:
        print(f"\nFitting Polynomial Degree {degree}:")
        poly_model = make_pipeline(PolynomialFeatures(degree), LinearRegression())
        poly_model.fit(train_x, train_y)
        models.append(poly_model)
        
        poly_train_pred = poly_model.predict(train_x)
        poly_test_pred = poly_model.predict(test_x)
        
        train_mse = mean_squared_error(train_y, poly_train_pred)
        test_mse = mean_squared_error(test_y, poly_test_pred)
        
        train_mse_list.append(train_mse)
        test_mse_list.append(test_mse)
        
        print(f"Polynomial Degree {degree} - Train Loss: {train_mse:.4f}, Test Loss: {test_mse:.4f}")
        
        # 绘制并保存当前度数的多项式模型在数据分布上的图
        plot_polynomial_curve(train_x, train_y, test_x, test_y, poly_model, degree, train_mse, test_mse)
    
    # 找到最佳度数
    best_degree_idx = np.argmin(test_mse_list)
    best_degree = degrees[best_degree_idx]
    best_poly_model = models[best_degree_idx]
    
    print(f"\nBest Polynomial Model (Degree={best_degree}):")
    print(f"Train Loss: {train_mse_list[best_degree_idx]:.4f}")
    print(f"Test Loss: {test_mse_list[best_degree_idx]:.4f}")
    
    # 绘制不同度数的MSE曲线
    plt.figure(figsize=(10, 6))
    plt.plot(degrees, train_mse_list, 'bo-', label='Train MSE')
    plt.plot(degrees, test_mse_list, 'ro-', label='Test MSE')
    plt.axvline(x=best_degree, color='g', linestyle='--', label=f'Best Degree ({best_degree})')
    plt.title('Polynomial Regression: Degree vs MSE')
    plt.xlabel('Polynomial Degree')
    plt.ylabel('Mean Squared Error')
    plt.legend()
    plt.grid(True)
    plt.savefig('plots/polynomial_degree_comparison.png')
    plt.close()
    
    return best_poly_model, train_mse_list[best_degree_idx], test_mse_list[best_degree_idx], best_degree, train_mse_list, test_mse_list

# 5. 正弦函数模型 (使用随机梯度下降进行拟合来展示训练过程)
def sine_model_fit(train_x, train_y, test_x, test_y, batch_size=32, epochs=1000, learning_rate=0.01, verbose_interval=50):
    print(f"Running Sine Model Fitting (batch_size={batch_size}, epochs={epochs}, learning_rate={learning_rate})")
    
    # 定义正弦模型
    def sine_func(x, params):
        """Sine model: a * sin(b * x + c) + d"""
        a, b, c, d = params
        return a * np.sin(b * x + c) + d
    
    # 定义损失函数
    def loss_func(params, x, y):
        pred = sine_func(x, params)
        return np.mean((pred - y) ** 2)
    
    # 定义梯度
    def gradient(params, x, y):
        a, b, c, d = params
        m = len(y)
        pred = sine_func(x, params)
        error = pred - y
        
        # 各参数的梯度
        da = np.mean(2 * error * np.sin(b * x + c))
        db = np.mean(2 * error * a * np.cos(b * x + c) * x)
        dc = np.mean(2 * error * a * np.cos(b * x + c))
        dd = np.mean(2 * error)
        
        return np.array([da, db, dc, dd])
    
    # 转换为1D数组
    train_x_1d = train_x.flatten()
    test_x_1d = test_x.flatten()
    
    # 初始化参数
    params = np.array([1.0, 10.0, 0.0, 0.0])  # [a, b, c, d]
    
    train_losses = []
    test_losses = []
    
    # 初始损失
    initial_train_loss = loss_func(params, train_x_1d, train_y)
    initial_test_loss = loss_func(params, test_x_1d, test_y)
    train_losses.append(initial_train_loss)
    test_losses.append(initial_test_loss)
    
    m = len(train_y)
    
    # 训练过程
    if batch_size is None or batch_size >= m:
        # 全批量
        for epoch in range(1, epochs + 1):
            grad = gradient(params, train_x_1d, train_y)
            params = params - learning_rate * grad
            
            # 计算损失
            train_loss = loss_func(params, train_x_1d, train_y)
            test_loss = loss_func(params, test_x_1d, test_y)
            
            train_losses.append(train_loss)
            test_losses.append(test_loss)
            
            # 输出训练进度
            if epoch % verbose_interval == 0 or epoch == 1 or epoch == epochs:
                print(f"Epoch {epoch}/{epochs}, Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}")
    else:
        # 小批量
        n_batches = int(np.ceil(m / batch_size))
        for epoch in range(1, epochs + 1):
            indices = np.random.permutation(m)
            x_shuffled = train_x_1d[indices]
            y_shuffled = train_y[indices]
            
            epoch_loss = 0.0
            
            for i in range(n_batches):
                start_idx = i * batch_size
                end_idx = min(start_idx + batch_size, m)
                
                x_batch = x_shuffled[start_idx:end_idx]
                y_batch = y_shuffled[start_idx:end_idx]
                
                grad = gradient(params, x_batch, y_batch)
                params = params - learning_rate * grad
                
                # 计算批次损失
                batch_loss = loss_func(params, x_batch, y_batch)
                epoch_loss += batch_loss * len(x_batch) / m
            
            # 计算整个训练集和测试集的损失
            train_loss = loss_func(params, train_x_1d, train_y)
            test_loss = loss_func(params, test_x_1d, test_y)
            
            train_losses.append(train_loss)
            test_losses.append(test_loss)
            
            # 输出训练进度
            if epoch % verbose_interval == 0 or epoch == 1 or epoch == epochs:
                print(f"Epoch {epoch}/{epochs}, Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}")
    
    # 从自定义拟合切换到scipy的curve_fit进行最终拟合（可选，获取更精确的参数）
    def sine_model_scipy(x, a, b, c, d):
        return a * np.sin(b * x + c) + d
    
    try:
        optimal_params, _ = curve_fit(sine_model_scipy, train_x_1d, train_y, p0=params)
        a, b, c, d = optimal_params
    except:
        a, b, c, d = params
        
    print(f"\nFinal Sine Model Parameters:")
    print(f"a={a:.4f}, b={b:.4f}, c={c:.4f}, d={d:.4f}")
    
    # 绘制训练曲线
    plt.figure(figsize=(10, 6))
    plt.plot(range(len(train_losses)), train_losses, 'b-', label='Train Loss')
    plt.plot(range(len(test_losses)), test_losses, 'r-', label='Test Loss')
    plt.title('Sine Model Training Curve')
    plt.xlabel('Epoch')
    plt.ylabel('Mean Squared Error')
    plt.legend()
    plt.grid(True)
    plt.savefig('plots/sine_model_training_curve.png')
    
    # 如果损失值变化太大，使用对数尺度重新绘制
    if max(train_losses) / min(train_losses) > 100:
        plt.figure(figsize=(10, 6))
        plt.semilogy(range(len(train_losses)), train_losses, 'b-', label='Train Loss')
        plt.semilogy(range(len(test_losses)), test_losses, 'r-', label='Test Loss')
        plt.title('Sine Model Training Curve (Log Scale)')
        plt.xlabel('Epoch')
        plt.ylabel('Mean Squared Error (log scale)')
        plt.legend()
        plt.grid(True)
        plt.savefig('plots/sine_model_training_curve_log.png')
    
    final_params = np.array([a, b, c, d])
    return final_params, train_losses[-1], test_losses[-1], train_losses, test_losses

# 6. 傅里叶级数模型 (使用随机梯度下降进行拟合来展示训练过程)
def fourier_model_fit(train_x, train_y, test_x, test_y, batch_size=32, epochs=1000, learning_rate=0.01, verbose_interval=50):
    print(f"Running Fourier Series Model Fitting (batch_size={batch_size}, epochs={epochs}, learning_rate={learning_rate})")
    
    # 定义傅里叶模型
    def fourier_func(x, params):
        """Fourier series model: a0 + a1*cos(wx) + b1*sin(wx) + a2*cos(2wx) + b2*sin(2wx) + a3*cos(3wx) + b3*sin(3wx)"""
        a0, a1, b1, a2, b2, a3, b3, w = params
        return a0 + a1*np.cos(w*x) + b1*np.sin(w*x) + a2*np.cos(2*w*x) + b2*np.sin(2*w*x) + a3*np.cos(3*w*x) + b3*np.sin(3*w*x)
    
    # 定义损失函数
    def loss_func(params, x, y):
        pred = fourier_func(x, params)
        return np.mean((pred - y) ** 2)
    
    # 定义梯度（近似计算）
    def gradient(params, x, y, eps=1e-6):
        m = len(y)
        grad = np.zeros_like(params)
        for i in range(len(params)):
            params_plus = params.copy()
            params_plus[i] += eps
            params_minus = params.copy()
            params_minus[i] -= eps
            
            loss_plus = loss_func(params_plus, x, y)
            loss_minus = loss_func(params_minus, x, y)
            
            grad[i] = (loss_plus - loss_minus) / (2 * eps)
        
        return grad
    
    # 转换为1D数组
    train_x_1d = train_x.flatten()
    test_x_1d = test_x.flatten()
    
    # 初始化参数
    params = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 10.0])  # [a0, a1, b1, a2, b2, a3, b3, w]
    
    train_losses = []
    test_losses = []
    
    # 初始损失
    initial_train_loss = loss_func(params, train_x_1d, train_y)
    initial_test_loss = loss_func(params, test_x_1d, test_y)
    train_losses.append(initial_train_loss)
    test_losses.append(initial_test_loss)
    
    m = len(train_y)
    
    # 训练过程
    if batch_size is None or batch_size >= m:
        # 全批量
        for epoch in range(1, epochs + 1):
            grad = gradient(params, train_x_1d, train_y)
            params = params - learning_rate * grad
            
            # 计算损失
            train_loss = loss_func(params, train_x_1d, train_y)
            test_loss = loss_func(params, test_x_1d, test_y)
            
            train_losses.append(train_loss)
            test_losses.append(test_loss)
            
            # 输出训练进度
            if epoch % verbose_interval == 0 or epoch == 1 or epoch == epochs:
                print(f"Epoch {epoch}/{epochs}, Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}")
    else:
        # 小批量
        n_batches = int(np.ceil(m / batch_size))
        for epoch in range(1, epochs + 1):
            indices = np.random.permutation(m)
            x_shuffled = train_x_1d[indices]
            y_shuffled = train_y[indices]
            
            epoch_loss = 0.0
            
            for i in range(n_batches):
                start_idx = i * batch_size
                end_idx = min(start_idx + batch_size, m)
                
                x_batch = x_shuffled[start_idx:end_idx]
                y_batch = y_shuffled[start_idx:end_idx]
                
                grad = gradient(params, x_batch, y_batch)
                params = params - learning_rate * grad
                
                # 计算批次损失
                batch_loss = loss_func(params, x_batch, y_batch)
                epoch_loss += batch_loss * len(x_batch) / m
            
            # 计算整个训练集和测试集的损失
            train_loss = loss_func(params, train_x_1d, train_y)
            test_loss = loss_func(params, test_x_1d, test_y)
            
            train_losses.append(train_loss)
            test_losses.append(test_loss)
            
            # 输出训练进度
            if epoch % verbose_interval == 0 or epoch == 1 or epoch == epochs:
                print(f"Epoch {epoch}/{epochs}, Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}")
    
    # 从自定义拟合切换到scipy的curve_fit进行最终拟合（可选，获取更精确的参数）
    def fourier_model_scipy(x, a0, a1, b1, a2, b2, a3, b3, w):
        return a0 + a1*np.cos(w*x) + b1*np.sin(w*x) + a2*np.cos(2*w*x) + b2*np.sin(2*w*x) + a3*np.cos(3*w*x) + b3*np.sin(3*w*x)
    
    try:
        optimal_params, _ = curve_fit(fourier_model_scipy, train_x_1d, train_y, p0=params)
    except:
        optimal_params = params
        
    a0, a1, b1, a2, b2, a3, b3, w = optimal_params
        
    print(f"\nFinal Fourier Model Parameters:")
    print(f"a0={a0:.4f}, a1={a1:.4f}, b1={b1:.4f}, a2={a2:.4f}, b2={b2:.4f}, a3={a3:.4f}, b3={b3:.4f}, w={w:.4f}")
    
    # 绘制训练曲线
    plt.figure(figsize=(10, 6))
    plt.plot(range(len(train_losses)), train_losses, 'b-', label='Train Loss')
    plt.plot(range(len(test_losses)), test_losses, 'r-', label='Test Loss')
    plt.title('Fourier Model Training Curve')
    plt.xlabel('Epoch')
    plt.ylabel('Mean Squared Error')
    plt.legend()
    plt.grid(True)
    plt.savefig('plots/fourier_model_training_curve.png')
    
    # 如果损失值变化太大，使用对数尺度重新绘制
    if max(train_losses) / min(train_losses) > 100:
        plt.figure(figsize=(10, 6))
        plt.semilogy(range(len(train_losses)), train_losses, 'b-', label='Train Loss')
        plt.semilogy(range(len(test_losses)), test_losses, 'r-', label='Test Loss')
        plt.title('Fourier Model Training Curve (Log Scale)')
        plt.xlabel('Epoch')
        plt.ylabel('Mean Squared Error (log scale)')
        plt.legend()
        plt.grid(True)
        plt.savefig('plots/fourier_model_training_curve_log.png')
    
    return optimal_params, train_losses[-1], test_losses[-1], train_losses, test_losses

# 绘制所有模型比较图
def plot_all_models_comparison(train_x, train_y, test_x, test_y, models_data):
    print("Plotting all models comparison...")
    
    # 创建图表
    plt.figure(figsize=(12, 8))
    plt.scatter(train_x, train_y, color='blue', label='Training Data')
    plt.scatter(test_x, test_y, color='green', label='Test Data')
    
    # 绘制预测曲线
    x_range = np.linspace(min(train_x.min(), test_x.min()), max(train_x.max(), test_x.max()), 500).reshape(-1, 1)
    x_range_1d = x_range.flatten()
    
    colors = ['r', 'm', 'g', 'y', 'c', 'k']
    line_styles = ['-', '--', '-.', ':', '-', '--']
    
    for i, (name, model_type, model, train_mse, test_mse) in enumerate(models_data):
        color = colors[i % len(colors)]
        ls = line_styles[i % len(line_styles)]
        
        if model_type == 'linear':
            # 线性模型
            if name == 'Least Squares':
                plt.plot(x_range, model.predict(x_range), color=color, ls=ls, 
                         label=f'{name} (Test MSE: {test_mse:.4f})')
            else:
                # 梯度下降或牛顿法
                X_b_range = np.c_[np.ones((len(x_range), 1)), x_range]
                plt.plot(x_range, X_b_range.dot(model), color=color, ls=ls, 
                         label=f'{name} (Test MSE: {test_mse:.4f})')
        
        elif model_type == 'polynomial':
            # 多项式模型
            plt.plot(x_range, model.predict(x_range), color=color, ls=ls, 
                     label=f'{name} (Test MSE: {test_mse:.4f})')
        
        elif model_type == 'sine':
            # 正弦模型
            a, b, c, d = model
            plt.plot(x_range, a * np.sin(b * x_range_1d + c) + d, color=color, ls=ls, 
                     label=f'{name} (Test MSE: {test_mse:.4f})')
        
        elif model_type == 'fourier':
            # 傅里叶模型
            a0, a1, b1, a2, b2, a3, b3, w = model
            y_pred = a0 + a1*np.cos(w*x_range_1d) + b1*np.sin(w*x_range_1d) + \
                     a2*np.cos(2*w*x_range_1d) + b2*np.sin(2*w*x_range_1d) + \
                     a3*np.cos(3*w*x_range_1d) + b3*np.sin(3*w*x_range_1d)
            plt.plot(x_range, y_pred, color=color, ls=ls, 
                     label=f'{name} (Test MSE: {test_mse:.4f})')
    
    plt.title('Model Comparison')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.grid(True)
    plt.savefig('plots/all_models_comparison.png')
    plt.close()
    
    # 创建MSE比较条形图
    plt.figure(figsize=(12, 6))
    names = [data[0] for data in models_data]
    train_mses = [data[3] for data in models_data]
    test_mses = [data[4] for data in models_data]
    
    x = np.arange(len(names))
    width = 0.35
    plt.bar(x - width/2, train_mses, width, label='Train MSE')
    plt.bar(x + width/2, test_mses, width, label='Test MSE')
    plt.yscale('log')
    plt.xticks(x, names, rotation=45, ha='right')
    plt.ylabel('Mean Squared Error (log scale)')
    plt.title('Model Performance Comparison')
    plt.legend()
    plt.tight_layout()
    plt.grid(True, axis='y')
    plt.savefig('plots/all_models_mse_comparison.png')
    plt.close()

# 主函数
def main():
    parser = argparse.ArgumentParser(description='Run regression models with various methods')
    parser.add_argument('--method', type=str, default='all', 
                        choices=['least_squares', 'gradient_descent', 'newton', 'polynomial', 'sine', 'fourier', 'all'],
                        help='Regression method to use')
    parser.add_argument('--batch_size', type=int, default=None, 
                        help='Batch size for training (mainly for gradient descent)')
    parser.add_argument('--epochs', type=int, default=None, 
                        help='Number of epochs for training')
    parser.add_argument('--train_path', type=str, 
                        default='/mnt/petrelfs/liqingyun/wzk/dev/PRML-2025/homework-1/Data4Regression - Training Data.csv',
                        help='Path to training data CSV')
    parser.add_argument('--test_path', type=str, 
                        default='/mnt/petrelfs/liqingyun/wzk/dev/PRML-2025/homework-1/Data4Regression - Test Data.csv',
                        help='Path to test data CSV')
    parser.add_argument('--plot', action='store_true', default=True, help='Generate and show plots')
    
    args = parser.parse_args()
    
    # 加载数据
    train_x, train_y, test_x, test_y = load_data(args.train_path, args.test_path)
    
    # 存储所有模型数据以便比较
    models_data = []
    
    # 根据选择的方法运行回归
    if args.method == 'least_squares' or args.method == 'all':
        # 如果未指定epochs，使用默认值
        epochs = args.epochs if args.epochs is not None else 10
        model, train_mse, test_mse, train_losses, test_losses = least_squares_method(
            train_x, train_y, test_x, test_y, args.batch_size, epochs)
        models_data.append(('Least Squares', 'linear', model, train_mse, test_mse))
        
    if args.method == 'gradient_descent' or args.method == 'all':
        # 如果未指定epochs和batch_size，使用默认值
        epochs = args.epochs if args.epochs is not None else 1000
        batch_size = args.batch_size if args.batch_size is not None else 32
        model, train_mse, test_mse, train_losses, test_losses = gradient_descent_method(
            train_x, train_y, test_x, test_y, batch_size, epochs)
        models_data.append(('Gradient Descent', 'linear', model, train_mse, test_mse))
        
    if args.method == 'newton' or args.method == 'all':
        # 如果未指定epochs，使用默认值
        epochs = args.epochs if args.epochs is not None else 10
        model, train_mse, test_mse, train_losses, test_losses = newton_method(
            train_x, train_y, test_x, test_y, args.batch_size, epochs)
        models_data.append(('Newton\'s Method', 'linear', model, train_mse, test_mse))
        
    if args.method == 'polynomial' or args.method == 'all':
        model, train_mse, test_mse, best_degree, train_mse_list, test_mse_list = polynomial_regression(
            train_x, train_y, test_x, test_y, args.batch_size, args.epochs)
        models_data.append((f'Polynomial (Degree={best_degree})', 'polynomial', model, train_mse, test_mse))
        
    if args.method == 'sine' or args.method == 'all':
        # 如果未指定epochs和batch_size，使用默认值
        epochs = args.epochs if args.epochs is not None else 1000
        batch_size = args.batch_size if args.batch_size is not None else 32
        model, train_mse, test_mse, train_losses, test_losses = sine_model_fit(
            train_x, train_y, test_x, test_y, batch_size, epochs)
        models_data.append(('Sine Model', 'sine', model, train_mse, test_mse))
        
    if args.method == 'fourier' or args.method == 'all':
        # 如果未指定epochs和batch_size，使用默认值
        epochs = args.epochs if args.epochs is not None else 1000
        batch_size = args.batch_size if args.batch_size is not None else 32
        model, train_mse, test_mse, train_losses, test_losses = fourier_model_fit(
            train_x, train_y, test_x, test_y, batch_size, epochs)
        models_data.append(('Fourier Series', 'fourier', model, train_mse, test_mse))
    
    # 如果指定了绘图，绘制所有模型的比较图
    if args.plot and len(models_data) > 1:
        plot_all_models_comparison(train_x, train_y, test_x, test_y, models_data)
    
    # 找出最佳模型
    if len(models_data) > 0:
        best_model_idx = np.argmin([data[4] for data in models_data])  # 基于测试MSE
        print(f"\nBest Model: {models_data[best_model_idx][0]} with Test MSE: {models_data[best_model_idx][4]:.6f}")

if __name__ == "__main__":
    main()