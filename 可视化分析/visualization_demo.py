#!/usr/bin/env python
# -*- coding: utf-8 -*-
# 说明：本文件演示优化算法的可视化，包括线性规划可行域、梯度下降过程动画、网络流图形展示、敏感性分析与3D优化表面。
# 语法与规则：使用Matplotlib/Seaborn绘制图表；中文标签需加载字体；统一PNG输出规范。
"""
运筹学优化可视化演示
Operations Research Optimization Visualization Demo

本文件专门用于展示各种优化算法的可视化效果，包括：
1. 线性规划可行域可视化
2. 优化过程动态演示
3. 网络流可视化
4. 敏感性分析图表
5. 3D优化表面
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
import networkx as nx
import pandas as pd
from matplotlib.animation import FuncAnimation
import warnings
# 抑制非关键警告，保证教学输出清爽
warnings.filterwarnings('ignore')

# 路径与中文字体：移动到子目录后也能导入根目录的配置
import os, sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)
from font_config import setup_chinese_font

# 先设置样式，再设置中文字体（避免样式覆盖字体设置）
plt.style.use('seaborn-v0_8')
setup_chinese_font()

class OptimizationVisualization:
    """优化可视化演示类
    作用：封装五类可视化——线性规划可行域、梯度下降过程动画、网络流展示、敏感性分析与三维优化表面。
    设计：面向对象组织流程；统一输出样式与PNG文件保存规则；入口集中在 main()。
    """
    
    def __init__(self):
        print("🎨 运筹学优化可视化演示系统")
        print("=" * 50)
        # 确保字体配置正确应用
        setup_chinese_font()
    
    def linear_programming_feasible_region(self):
        """线性规划可行域可视化
        作用：绘制二维线性约束的可行域与目标等值线，直观展示最优解位置与极点特性。
        语法要点：
        - 使用 `contourf`/`fill_between` 表示不等式约束的有效区域
        - 使用等利润线（目标函数等值线）帮助理解优化方向
        规则：中文标签、统一样式、PNG输出（dpi=300）。
        """
        print("\n📐 1. 线性规划可行域可视化")
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # 示例1: 简单的二维线性规划
        x = np.linspace(0, 10, 400)
        y = np.linspace(0, 10, 400)
        X, Y = np.meshgrid(x, y)
        
        # 约束条件
        # 2x + 3y <= 12
        # x + 2y <= 8
        # x >= 0, y >= 0
        
        constraint1 = (2*X + 3*Y <= 12)
        constraint2 = (X + 2*Y <= 8)
        constraint3 = (X >= 0)
        constraint4 = (Y >= 0)
        
        # 可行域
        feasible = constraint1 & constraint2 & constraint3 & constraint4
        
        # 绘制约束线
        y1 = (12 - 2*x) / 3
        y2 = (8 - x) / 2
        
        ax1.plot(x, y1, 'r-', linewidth=2, label='2x + 3y ≤ 12')
        ax1.plot(x, y2, 'b-', linewidth=2, label='x + 2y ≤ 8')
        ax1.axhline(y=0, color='k', linewidth=1)
        ax1.axvline(x=0, color='k', linewidth=1)
        
        # 填充可行域
        ax1.contourf(X, Y, feasible.astype(int), levels=[0.5, 1.5], 
                    colors=['lightgreen'], alpha=0.5)
        
        # 目标函数等高线 (max 3x + 2y)
        for c in [6, 9, 12, 15]:
            y_obj = (c - 3*x) / 2
            ax1.plot(x, y_obj, '--', alpha=0.7, label=f'3x + 2y = {c}')
        
        # 最优解
        ax1.plot(2, 3, 'ro', markersize=10, label='最优解 (2, 3)')
        
        ax1.set_xlim(0, 8)
        ax1.set_ylim(0, 6)
        ax1.set_xlabel('$x_1$')
        ax1.set_ylabel('$x_2$')
        ax1.set_title('线性规划可行域\nmax $3x_1 + 2x_2$', fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 示例2: 三维线性规划投影
        # 生成随机约束
        np.random.seed(42)
        n_constraints = 5
        
        # 约束系数
        A = np.random.uniform(0.5, 2.0, (n_constraints, 2))
        b = np.random.uniform(5, 15, n_constraints)
        
        # 绘制多个约束
        colors = plt.cm.Set3(np.linspace(0, 1, n_constraints))
        
        for i in range(n_constraints):
            if A[i, 1] != 0:
                y_constraint = (b[i] - A[i, 0] * x) / A[i, 1]
                ax2.plot(x, y_constraint, color=colors[i], linewidth=2,
                        label=f'{A[i,0]:.1f}$x_1$ + {A[i,1]:.1f}$x_2$ ≤ {b[i]:.1f}')
        
        # 计算可行域（简化）
        feasible_complex = np.ones_like(X, dtype=bool)
        for i in range(n_constraints):
            feasible_complex &= (A[i, 0] * X + A[i, 1] * Y <= b[i])
        feasible_complex &= (X >= 0) & (Y >= 0)
        
        ax2.contourf(X, Y, feasible_complex.astype(int), levels=[0.5, 1.5], 
                    colors=['lightblue'], alpha=0.5)
        
        ax2.set_xlim(0, 10)
        ax2.set_ylim(0, 10)
        ax2.set_xlabel('$x_1$')
        ax2.set_ylabel('$x_2$')
        ax2.set_title('复杂约束可行域', fontweight='bold')
        ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        save_path = os.path.join(BASE_DIR, 'feasible_region.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def optimization_process_animation(self):
        """优化过程动态演示
        作用：以动画或轨迹图展示梯度下降在二维函数上的迭代路径，帮助理解学习率与收敛过程。
        语法要点：
        - 使用 `matplotlib.animation.FuncAnimation` 或逐帧绘制更新点位
        - 函数等高线与迭代路径同时呈现
        规则：中文标签、统一样式、PNG输出（dpi=300），教学友好。
        """
        print("\n🎬 2. 梯度下降优化过程动画")
        
        # 定义目标函数 f(x,y) = (x-3)² + (y-2)²
        def objective_function(x, y):
            return (x - 3)**2 + (y - 2)**2
        
        # 梯度函数
        def gradient(x, y):
            return np.array([2*(x-3), 2*(y-2)])
        
        # 创建网格
        x = np.linspace(-1, 7, 100)
        y = np.linspace(-1, 5, 100)
        X, Y = np.meshgrid(x, y)
        Z = objective_function(X, Y)
        
        # 梯度下降路径
        learning_rate = 0.1
        max_iterations = 50
        
        # 起始点
        path_x = [0.5]
        path_y = [0.5]
        
        current_x, current_y = 0.5, 0.5
        
        for i in range(max_iterations):
            grad = gradient(current_x, current_y)
            current_x -= learning_rate * grad[0]
            current_y -= learning_rate * grad[1]
            path_x.append(current_x)
            path_y.append(current_y)
            
            # 如果收敛则停止
            if np.linalg.norm(grad) < 0.01:
                break
        
        # 静态图显示优化路径
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # 等高线图
        contour = ax.contour(X, Y, Z, levels=20, alpha=0.6)
        ax.clabel(contour, inline=True, fontsize=8)
        
        # 优化路径
        ax.plot(path_x, path_y, 'ro-', linewidth=2, markersize=6, 
               label='梯度下降路径')
        ax.plot(path_x[0], path_y[0], 'go', markersize=10, label='起始点')
        ax.plot(3, 2, 'r*', markersize=15, label='最优解')
        
        # 添加箭头显示方向
        for i in range(0, len(path_x)-1, 3):
            ax.annotate('', xy=(path_x[i+1], path_y[i+1]), 
                       xytext=(path_x[i], path_y[i]),
                       arrowprops=dict(arrowstyle='->', color='red', lw=1.5))
        
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_title('梯度下降优化过程\nf(x,y) = (x-3)² + (y-2)²', fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        save_path = os.path.join(BASE_DIR, 'optimization_process.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"优化完成，共迭代 {len(path_x)-1} 次")
        print(f"   最终解: ({current_x:.3f}, {current_y:.3f})")
        print(f"   目标函数值: {objective_function(current_x, current_y):.6f}")
    
    def network_flow_visualization(self):
        """网络流可视化
        作用：绘制供应-中转-需求三层网络结构与最大流结果，线宽表示流量，标签显示流量/容量。
        语法要点：
        - 使用 NetworkX 构建有向图并设置 `capacity`/`flow` 属性
        - 自定义布局保证教学可读性，边标签显示 `flow/capacity`
        规则：中文标签、统一样式、PNG输出（dpi=300）。
        """
        print("\n3. 网络流优化可视化")
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        
        # 创建网络图
        G = nx.DiGraph()
        
        # 添加节点
        supply_nodes = ['S1', 'S2', 'S3']  # 供应节点
        demand_nodes = ['D1', 'D2', 'D3', 'D4']  # 需求节点
        intermediate_nodes = ['T1', 'T2']  # 中转节点
        
        all_nodes = supply_nodes + intermediate_nodes + demand_nodes
        G.add_nodes_from(all_nodes)
        
        # 添加边和容量
        edges_with_capacity = [
            ('S1', 'T1', 50), ('S1', 'T2', 40),
            ('S2', 'T1', 60), ('S2', 'T2', 30),
            ('S3', 'T1', 30), ('S3', 'T2', 50),
            ('T1', 'D1', 35), ('T1', 'D2', 25),
            ('T1', 'D3', 30), ('T2', 'D1', 20),
            ('T2', 'D2', 40), ('T2', 'D3', 25),
            ('T2', 'D4', 35)
        ]
        
        for source, target, capacity in edges_with_capacity:
            G.add_edge(source, target, capacity=capacity, flow=0)
        
        # 节点位置
        pos = {
            'S1': (0, 2), 'S2': (0, 1), 'S3': (0, 0),
            'T1': (2, 1.5), 'T2': (2, 0.5),
            'D1': (4, 2), 'D2': (4, 1.5), 'D3': (4, 0.5), 'D4': (4, 0)
        }
        
        # 绘制原始网络
        node_colors = ['lightcoral' if node in supply_nodes 
                      else 'lightblue' if node in demand_nodes 
                      else 'lightgreen' for node in G.nodes()]
        
        nx.draw(G, pos, ax=ax1, with_labels=True, node_color=node_colors,
               node_size=1500, font_size=10, font_weight='bold',
               arrows=True, arrowsize=20, edge_color='gray')
        
        # 添加容量标签
        edge_labels = {(u, v): f"{d['capacity']}" for u, v, d in G.edges(data=True)}
        nx.draw_networkx_edge_labels(G, pos, edge_labels, ax=ax1, font_size=8)
        
        ax1.set_title('网络流结构图\n(数字表示容量)', fontweight='bold')
        
        # 模拟最大流结果
        np.random.seed(42)
        for u, v, d in G.edges(data=True):
            # 随机分配流量（不超过容量）
            d['flow'] = min(d['capacity'], np.random.randint(0, d['capacity'] + 1))
        
        # 绘制流量结果
        nx.draw(G, pos, ax=ax2, with_labels=True, node_color=node_colors,
               node_size=1500, font_size=10, font_weight='bold',
               arrows=True, arrowsize=20)
        
        # 根据流量调整边的粗细和颜色
        for u, v, d in G.edges(data=True):
            flow_ratio = d['flow'] / d['capacity'] if d['capacity'] > 0 else 0
            width = 1 + 4 * flow_ratio
            color = plt.cm.Reds(0.3 + 0.7 * flow_ratio)
            
            nx.draw_networkx_edges(G, pos, [(u, v)], ax=ax2,
                                 width=width, edge_color=[color],
                                 arrows=True, arrowsize=20)
        
        # 添加流量标签
        flow_labels = {(u, v): f"{d['flow']}/{d['capacity']}" 
                      for u, v, d in G.edges(data=True)}
        nx.draw_networkx_edge_labels(G, pos, flow_labels, ax=ax2, font_size=8)
        
        ax2.set_title('最大流结果\n(流量/容量，线条粗细表示流量)', fontweight='bold')
        
        # 计算总流量
        total_flow = sum(d['flow'] for u, v, d in G.edges(data=True) 
                        if u in supply_nodes)
        
        plt.figtext(0.5, 0.02, f'总流量: {total_flow} 单位', 
                   ha='center', fontsize=12, fontweight='bold')
        
        plt.tight_layout()
        save_path = os.path.join(BASE_DIR, 'network_flow.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def sensitivity_analysis(self):
        """敏感性分析可视化"""
        print("\n4. 敏感性分析可视化")
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. 参数敏感性分析
        # 假设线性规划问题：max cx subject to Ax <= b
        c_values = np.linspace(1, 10, 50)
        optimal_values = []
        
        for c in c_values:
            # 模拟最优值随目标函数系数变化
            optimal_value = c * 5 - 0.1 * c**2  # 二次函数模拟
            optimal_values.append(optimal_value)
        
        ax1.plot(c_values, optimal_values, 'b-', linewidth=2)
        ax1.fill_between(c_values, optimal_values, alpha=0.3)
        ax1.set_xlabel('目标函数系数 c')
        ax1.set_ylabel('最优目标值')
        ax1.set_title('目标函数系数敏感性', fontweight='bold')
        ax1.grid(True, alpha=0.3)
        
        # 2. 约束右端项敏感性
        b_values = np.linspace(5, 25, 50)
        shadow_prices = []
        
        for b in b_values:
            # 模拟影子价格
            if b < 10:
                shadow_price = 2.0
            elif b < 20:
                shadow_price = 2.0 - 0.1 * (b - 10)
            else:
                shadow_price = 0
            shadow_prices.append(shadow_price)
        
        ax2.plot(b_values, shadow_prices, 'r-', linewidth=2)
        ax2.fill_between(b_values, shadow_prices, alpha=0.3, color='red')
        ax2.set_xlabel('约束右端项 b')
        ax2.set_ylabel('影子价格')
        ax2.set_title('约束右端项敏感性', fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        # 3. 多参数敏感性热力图
        param1_range = np.linspace(0.5, 2.0, 20)
        param2_range = np.linspace(1.0, 3.0, 20)
        P1, P2 = np.meshgrid(param1_range, param2_range)
        
        # 模拟目标函数值随两个参数变化
        Z_sensitivity = P1 * P2 * 10 - 0.5 * P1**2 - 0.3 * P2**2
        
        im = ax3.contourf(P1, P2, Z_sensitivity, levels=20, cmap='viridis')
        contour = ax3.contour(P1, P2, Z_sensitivity, levels=10, colors='white', alpha=0.5)
        ax3.clabel(contour, inline=True, fontsize=8)
        
        ax3.set_xlabel('参数 1')
        ax3.set_ylabel('参数 2')
        ax3.set_title('双参数敏感性分析', fontweight='bold')
        plt.colorbar(im, ax=ax3, label='目标函数值')
        
        # 4. 稳定性区间
        scenarios = ['悲观', '基准', '乐观']
        parameters = ['需求', '成本', '容量', '价格']
        
        # 模拟数据
        np.random.seed(42)
        stability_data = np.random.uniform(0.7, 1.3, (len(scenarios), len(parameters)))
        stability_data[1, :] = 1.0  # 基准情况
        
        # 热力图
        sns.heatmap(stability_data, annot=True, fmt='.2f', 
                   xticklabels=parameters, yticklabels=scenarios,
                   cmap='RdYlGn', center=1.0, ax=ax4)
        ax4.set_title('参数稳定性区间\n(相对于基准值)', fontweight='bold')
        
        plt.tight_layout()
        save_path = os.path.join(BASE_DIR, 'sensitivity_analysis.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def three_dimensional_optimization(self):
        """三维优化表面可视化"""
        print("\n🏔️ 5. 三维优化表面可视化")
        
        fig = plt.figure(figsize=(16, 12))
        
        # 1. 单目标优化表面
        ax1 = fig.add_subplot(2, 2, 1, projection='3d')
        
        x = np.linspace(-5, 5, 50)
        y = np.linspace(-5, 5, 50)
        X, Y = np.meshgrid(x, y)
        
        # Rosenbrock函数
        Z1 = (1 - X)**2 + 100 * (Y - X**2)**2
        
        surf1 = ax1.plot_surface(X, Y, Z1, cmap='viridis', alpha=0.8)
        ax1.contour(X, Y, Z1, zdir='z', offset=0, cmap='viridis', alpha=0.5)
        
        # 标记全局最优解
        ax1.scatter([1], [1], [0], color='red', s=100, label='全局最优')
        
        ax1.set_xlabel('$x_1$')
        ax1.set_ylabel('$x_2$')
        ax1.set_zlabel('$f(x_1, x_2)$')
        ax1.set_title('Rosenbrock函数\n(经典优化测试函数)', fontweight='bold')
        
        # 2. 多峰函数
        ax2 = fig.add_subplot(2, 2, 2, projection='3d')
        
        # Ackley函数
        Z2 = (-20 * np.exp(-0.2 * np.sqrt(0.5 * (X**2 + Y**2))) - 
              np.exp(0.5 * (np.cos(2*np.pi*X) + np.cos(2*np.pi*Y))) + 
              np.e + 20)
        
        surf2 = ax2.plot_surface(X, Y, Z2, cmap='plasma', alpha=0.8)
        ax2.contour(X, Y, Z2, zdir='z', offset=0, cmap='plasma', alpha=0.5)
        
        ax2.set_xlabel('$x_1$')
        ax2.set_ylabel('$x_2$')
        ax2.set_zlabel('$f(x_1, x_2)$')
        ax2.set_title('Ackley函数\n(多峰优化问题)', fontweight='bold')
        
        # 3. 约束优化问题
        ax3 = fig.add_subplot(2, 2, 3, projection='3d')
        
        # 目标函数
        Z3 = X**2 + Y**2
        
        # 约束区域
        constraint_mask = (X**2 + Y**2 <= 9) & (X + Y >= 1)
        Z3_constrained = np.where(constraint_mask, Z3, np.nan)
        
        surf3 = ax3.plot_surface(X, Y, Z3_constrained, cmap='coolwarm', alpha=0.8)
        
        # 绘制约束边界
        theta = np.linspace(0, 2*np.pi, 100)
        x_circle = 3 * np.cos(theta)
        y_circle = 3 * np.sin(theta)
        z_circle = x_circle**2 + y_circle**2
        ax3.plot(x_circle, y_circle, z_circle, 'r-', linewidth=3, label='约束边界')
        
        ax3.set_xlabel('$x_1$')
        ax3.set_ylabel('$x_2$')
        ax3.set_zlabel('$f(x_1, x_2)$')
        ax3.set_title('约束优化问题\nmin $x_1^2 + x_2^2$', fontweight='bold')
        
        # 4. 帕累托前沿（多目标优化）
        ax4 = fig.add_subplot(2, 2, 4, projection='3d')
        
        # 生成帕累托前沿数据
        n_points = 100
        t = np.linspace(0, 1, n_points)
        
        # 两个冲突目标
        obj1 = t**2
        obj2 = (1 - t)**2
        obj3 = t * (1 - t)  # 第三个目标
        
        # 绘制帕累托前沿
        ax4.plot(obj1, obj2, obj3, 'b-', linewidth=3, label='帕累托前沿')
        ax4.scatter(obj1[::10], obj2[::10], obj3[::10], c='red', s=50)
        
        # 添加一些非帕累托解
        np.random.seed(42)
        n_dominated = 50
        dom_obj1 = np.random.uniform(0, 1, n_dominated)
        dom_obj2 = np.random.uniform(0, 1, n_dominated)
        dom_obj3 = np.random.uniform(0, 0.5, n_dominated)
        
        ax4.scatter(dom_obj1, dom_obj2, dom_obj3, c='gray', alpha=0.5, s=20, 
                   label='被支配解')
        
        ax4.set_xlabel('目标1')
        ax4.set_ylabel('目标2')
        ax4.set_zlabel('目标3')
        ax4.set_title('多目标优化\n帕累托前沿', fontweight='bold')
        ax4.legend()
        
        plt.tight_layout()
        save_path = os.path.join(BASE_DIR, '3d_optimization.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def algorithm_comparison_dashboard(self):
        """算法对比仪表板"""
        print("\n📈 6. 算法性能对比仪表板")
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. 收敛速度对比
        iterations = np.arange(1, 51)
        
        # 模拟不同算法的收敛曲线
        gradient_descent = 100 * np.exp(-0.1 * iterations) + np.random.normal(0, 1, 50)
        newton_method = 100 * np.exp(-0.3 * iterations) + np.random.normal(0, 0.5, 50)
        genetic_algorithm = 100 - 2 * iterations + np.random.normal(0, 5, 50)
        simulated_annealing = 100 * np.exp(-0.05 * iterations) + 10 * np.sin(iterations/5) + np.random.normal(0, 2, 50)
        
        ax1.plot(iterations, gradient_descent, 'b-', label='梯度下降', linewidth=2)
        ax1.plot(iterations, newton_method, 'r-', label='牛顿法', linewidth=2)
        ax1.plot(iterations, genetic_algorithm, 'g-', label='遗传算法', linewidth=2)
        ax1.plot(iterations, simulated_annealing, 'm-', label='模拟退火', linewidth=2)
        
        ax1.set_xlabel('迭代次数')
        ax1.set_ylabel('目标函数值')
        ax1.set_title('算法收敛速度对比', fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_yscale('log')
        
        # 2. 算法性能雷达图
        algorithms = ['梯度下降', '牛顿法', '遗传算法', '模拟退火', '粒子群']
        metrics = ['收敛速度', '全局搜索', '内存使用', '实现难度', '稳定性']
        
        # 性能评分 (1-10)
        scores = np.array([
            [8, 3, 9, 8, 7],  # 梯度下降
            [9, 4, 8, 6, 8],  # 牛顿法
            [5, 9, 6, 7, 6],  # 遗传算法
            [6, 8, 7, 8, 7],  # 模拟退火
            [7, 8, 7, 6, 8]   # 粒子群
        ])
        
        # 转换为雷达图坐标
        angles = np.linspace(0, 2*np.pi, len(metrics), endpoint=False).tolist()
        angles += angles[:1]  # 闭合
        
        ax2 = plt.subplot(2, 2, 2, projection='polar')
        
        colors = ['blue', 'red', 'green', 'magenta', 'orange']
        for i, (algorithm, score) in enumerate(zip(algorithms, scores)):
            score_closed = score.tolist() + [score[0]]  # 闭合
            ax2.plot(angles, score_closed, 'o-', linewidth=2, 
                    label=algorithm, color=colors[i])
            ax2.fill(angles, score_closed, alpha=0.1, color=colors[i])
        
        ax2.set_xticks(angles[:-1])
        ax2.set_xticklabels(metrics)
        ax2.set_ylim(0, 10)
        ax2.set_title('算法性能雷达图', fontweight='bold', pad=20)
        ax2.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
        
        # 3. 问题规模 vs 求解时间
        problem_sizes = [10, 50, 100, 500, 1000, 5000]
        
        # 模拟不同算法的时间复杂度
        linear_time = np.array(problem_sizes) * 0.001
        quadratic_time = np.array(problem_sizes)**2 * 0.000001
        exponential_time = 2**(np.array(problem_sizes)/1000) * 0.01
        
        ax3.loglog(problem_sizes, linear_time, 'b-o', label='O(n) - 线性规划', linewidth=2)
        ax3.loglog(problem_sizes, quadratic_time, 'r-s', label='O(n²) - 二次规划', linewidth=2)
        ax3.loglog(problem_sizes[:4], exponential_time[:4], 'g-^', label='O(2ⁿ) - 整数规划', linewidth=2)
        
        ax3.set_xlabel('问题规模')
        ax3.set_ylabel('求解时间 (秒)')
        ax3.set_title('算法时间复杂度对比', fontweight='bold')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. 成功率 vs 问题难度
        difficulty_levels = ['简单', '中等', '困难', '极难']
        success_rates = {
            '精确算法': [100, 95, 70, 30],
            '启发式算法': [95, 90, 85, 60],
            '元启发式算法': [90, 88, 82, 75],
            '近似算法': [85, 80, 75, 70]
        }
        
        x_pos = np.arange(len(difficulty_levels))
        width = 0.2
        
        for i, (algorithm, rates) in enumerate(success_rates.items()):
            ax4.bar(x_pos + i*width, rates, width, label=algorithm, alpha=0.8)
        
        ax4.set_xlabel('问题难度')
        ax4.set_ylabel('成功率 (%)')
        ax4.set_title('算法成功率对比', fontweight='bold')
        ax4.set_xticks(x_pos + width * 1.5)
        ax4.set_xticklabels(difficulty_levels)
        ax4.legend()
        ax4.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        save_path = os.path.join(BASE_DIR, 'algorithm_comparison.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

def main():
    """主函数
    作用：顺序运行所有可视化方法并保存PNG输出，作为教学演示的一键入口。
    使用规则：脚本运行时触发；导入为模块时不自动执行。
    """
    viz = OptimizationVisualization()
    
    print("开始生成运筹学优化可视化演示...")
    
    # 运行所有可视化演示
    viz.linear_programming_feasible_region()
    viz.optimization_process_animation()
    viz.network_flow_visualization()
    viz.sensitivity_analysis()
    viz.three_dimensional_optimization()
    viz.algorithm_comparison_dashboard()
    
    print("\n" + "="*50)
    print("🎉 所有可视化演示完成！")
    print("图表已保存到 or 文件夹中：")
    print("  • feasible_region.png - 可行域可视化")
    print("  • optimization_process.png - 优化过程")
    print("  • network_flow.png - 网络流")
    print("  • sensitivity_analysis.png - 敏感性分析")
    print("  • 3d_optimization.png - 三维优化")
    print("  • algorithm_comparison.png - 算法对比")
    print("="*50)

if __name__ == "__main__":
    main()