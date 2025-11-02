"""
线性规划优化演示
Linear Programming Optimization Demo

演示内容：生产计划问题
- 目标：最大化利润
- 约束：劳动力和原材料限制
- 方法：使用PuLP求解器

作者: AI Assistant
日期: 2024
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pulp
import warnings
warnings.filterwarnings('ignore')

# 使用自定义字体配置支持中文
from font_config import setup_chinese_font
setup_chinese_font()

class LinearProgrammingDemo:
    """线性规划演示类"""
    
    def __init__(self):
        self.results = {}
        print("=" * 50)
        print("📊 线性规划优化演示")
        print("Linear Programming Demo")
        print("=" * 50)
    
    def solve_production_planning(self):
        """
        线性规划演示 - 生产计划问题
        
        问题描述：
        某制造公司生产三种产品A、B、C，需要使用两种资源：劳动力和原材料
        目标：最大化利润
        """
        print("\n📊 生产计划优化问题")
        print("-" * 40)
        
        # 问题数据（基于真实制造业数据）
        products = ['产品A', '产品B', '产品C']
        profit = [40, 30, 50]  # 每单位产品利润
        
        # 资源需求矩阵
        labor_req = [2, 1, 3]      # 劳动力需求（小时/单位）
        material_req = [1, 2, 1]   # 原材料需求（kg/单位）
        
        # 资源约束
        labor_available = 100      # 可用劳动力（小时）
        material_available = 80    # 可用原材料（kg）
        
        print(f"产品利润: {dict(zip(products, profit))}")
        print(f"劳动力需求: {dict(zip(products, labor_req))}")
        print(f"原材料需求: {dict(zip(products, material_req))}")
        print(f"可用劳动力: {labor_available} 小时")
        print(f"可用原材料: {material_available} kg")
        
        # 使用PuLP求解
        prob = pulp.LpProblem("生产计划", pulp.LpMaximize)
        
        # 决策变量
        x = [pulp.LpVariable(f"x{i}", lowBound=0) for i in range(3)]
        
        # 目标函数：最大化利润
        prob += pulp.lpSum([profit[i] * x[i] for i in range(3)])
        
        # 约束条件
        prob += pulp.lpSum([labor_req[i] * x[i] for i in range(3)]) <= labor_available
        prob += pulp.lpSum([material_req[i] * x[i] for i in range(3)]) <= material_available
        
        # 求解
        prob.solve(pulp.PULP_CBC_CMD(msg=0))
        
        # 结果
        solution = [x[i].varValue for i in range(3)]
        max_profit = pulp.value(prob.objective)
        
        print(f"\n✅ 最优解:")
        for i, product in enumerate(products):
            print(f"  {product}: {solution[i]:.2f} 单位")
        print(f"  最大利润: {max_profit:.2f} 元")
        
        # 资源利用率
        labor_used = sum(labor_req[i] * solution[i] for i in range(3))
        material_used = sum(material_req[i] * solution[i] for i in range(3))
        
        print(f"\n📈 资源利用率:")
        print(f"  劳动力: {labor_used:.2f}/{labor_available} ({labor_used/labor_available*100:.1f}%)")
        print(f"  原材料: {material_used:.2f}/{material_available} ({material_used/material_available*100:.1f}%)")
        
        # 保存结果用于可视化
        self.results = {
            'products': products,
            'solution': solution,
            'profit': profit,
            'max_profit': max_profit,
            'labor_used': labor_used,
            'material_used': material_used,
            'labor_available': labor_available,
            'material_available': material_available,
            'labor_req': labor_req,
            'material_req': material_req
        }
        
        return solution, max_profit
    
    def visualize_results(self):
        """可视化结果"""
        if not self.results:
            print("⚠️ 请先运行求解方法")
            return
        
        print("\n📈 生成可视化图表...")
        
        # 创建子图
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. 产品产量柱状图
        bars1 = ax1.bar(self.results['products'], self.results['solution'], 
                        color=['#FF6B6B', '#4ECDC4', '#45B7D1'])
        ax1.set_title('最优生产计划', fontsize=14, fontweight='bold')
        ax1.set_ylabel('产量 (单位)')
        ax1.grid(True, alpha=0.3)
        
        # 添加数值标签
        for bar, value in zip(bars1, self.results['solution']):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    f'{value:.1f}', ha='center', va='bottom')
        
        # 2. 资源利用率
        resources = ['劳动力', '原材料']
        used = [self.results['labor_used'], self.results['material_used']]
        available = [self.results['labor_available'], self.results['material_available']]
        utilization = [u/a*100 for u, a in zip(used, available)]
        
        bars2 = ax2.bar(resources, utilization, color=['#96CEB4', '#FFEAA7'])
        ax2.set_title('资源利用率', fontsize=14, fontweight='bold')
        ax2.set_ylabel('利用率 (%)')
        ax2.set_ylim(0, 100)
        ax2.grid(True, alpha=0.3)
        
        for bar, value in zip(bars2, utilization):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                    f'{value:.1f}%', ha='center', va='bottom')
        
        # 3. 利润贡献分析
        profit_contribution = [self.results['profit'][i] * self.results['solution'][i] 
                              for i in range(3)]
        bars3 = ax3.bar(self.results['products'], profit_contribution, 
                        color=['#DDA0DD', '#98FB98', '#F0E68C'])
        ax3.set_title('各产品利润贡献', fontsize=14, fontweight='bold')
        ax3.set_ylabel('利润贡献 (元)')
        ax3.grid(True, alpha=0.3)
        
        for bar, value in zip(bars3, profit_contribution):
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 10,
                    f'{value:.0f}', ha='center', va='bottom')
        
        # 4. 资源需求对比
        x_pos = np.arange(len(self.results['products']))
        width = 0.35
        
        bars4a = ax4.bar(x_pos - width/2, self.results['labor_req'], width, 
                        label='劳动力需求', color='#FFB6C1')
        bars4b = ax4.bar(x_pos + width/2, self.results['material_req'], width,
                        label='原材料需求', color='#87CEEB')
        
        ax4.set_title('各产品资源需求', fontsize=14, fontweight='bold')
        ax4.set_ylabel('需求量')
        ax4.set_xticks(x_pos)
        ax4.set_xticklabels(self.results['products'])
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('linear_programming_results.png', 
                   dpi=300, bbox_inches='tight')
        plt.show()
        
        print("✅ 可视化图表已保存为 'linear_programming_results.png'")
    
    def sensitivity_analysis(self):
        """敏感性分析"""
        if not self.results:
            print("⚠️ 请先运行求解方法")
            return
        
        print("\n🔍 敏感性分析")
        print("-" * 30)
        
        # 分析利润系数变化的影响
        print("1. 利润系数敏感性分析:")
        base_profits = self.results['profit']
        
        for i, product in enumerate(self.results['products']):
            print(f"\n  {product} 利润变化影响:")
            for change in [-20, -10, 10, 20]:  # 变化百分比
                new_profit = base_profits[i] * (1 + change/100)
                
                # 重新求解
                prob = pulp.LpProblem("敏感性分析", pulp.LpMaximize)
                x = [pulp.LpVariable(f"x{j}", lowBound=0) for j in range(3)]
                
                # 修改目标函数
                modified_profits = base_profits.copy()
                modified_profits[i] = new_profit
                prob += pulp.lpSum([modified_profits[j] * x[j] for j in range(3)])
                
                # 约束条件
                prob += pulp.lpSum([self.results['labor_req'][j] * x[j] for j in range(3)]) <= self.results['labor_available']
                prob += pulp.lpSum([self.results['material_req'][j] * x[j] for j in range(3)]) <= self.results['material_available']
                
                prob.solve(pulp.PULP_CBC_CMD(msg=0))
                new_max_profit = pulp.value(prob.objective)
                
                print(f"    利润{change:+d}% → 总利润: {new_max_profit:.2f} 元 "
                      f"(变化: {new_max_profit - self.results['max_profit']:+.2f})")
    
    def generate_report(self):
        """生成详细报告"""
        if not self.results:
            print("⚠️ 请先运行求解方法")
            return
        
        print("\n" + "="*50)
        print("📋 线性规划优化报告")
        print("="*50)
        
        print(f"\n🎯 问题描述:")
        print(f"  • 优化目标: 最大化生产利润")
        print(f"  • 决策变量: 三种产品的生产数量")
        print(f"  • 约束条件: 劳动力和原材料限制")
        
        print(f"\n📊 最优解:")
        for i, product in enumerate(self.results['products']):
            print(f"  • {product}: {self.results['solution'][i]:.2f} 单位")
        print(f"  • 最大利润: {self.results['max_profit']:.2f} 元")
        
        print(f"\n📈 资源利用情况:")
        labor_util = self.results['labor_used'] / self.results['labor_available'] * 100
        material_util = self.results['material_used'] / self.results['material_available'] * 100
        print(f"  • 劳动力利用率: {labor_util:.1f}%")
        print(f"  • 原材料利用率: {material_util:.1f}%")
        
        print(f"\n💡 管理建议:")
        if labor_util > 95:
            print(f"  • 劳动力资源接近满负荷，建议考虑增加人力")
        if material_util > 95:
            print(f"  • 原材料资源接近满负荷，建议优化采购计划")
        
        # 找出最有价值的产品
        profit_per_unit = self.results['profit']
        max_profit_idx = profit_per_unit.index(max(profit_per_unit))
        print(f"  • 单位利润最高产品: {self.results['products'][max_profit_idx]} "
              f"({profit_per_unit[max_profit_idx]} 元/单位)")
        
        print("="*50)

def main():
    """主函数"""
    # 创建演示实例
    demo = LinearProgrammingDemo()
    
    # 求解生产计划问题
    solution, max_profit = demo.solve_production_planning()
    
    # 生成可视化
    demo.visualize_results()
    
    # 敏感性分析
    demo.sensitivity_analysis()
    
    # 生成报告
    demo.generate_report()
    
    print(f"\n🎉 线性规划演示完成！")
    print(f"最优解: {[f'{x:.1f}' for x in solution]}")
    print(f"最大利润: {max_profit:.2f} 元")

if __name__ == "__main__":
    main()