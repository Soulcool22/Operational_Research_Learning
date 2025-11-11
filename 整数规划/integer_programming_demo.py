#!/usr/bin/env python
# -*- coding: utf-8 -*-
# 说明：本文件演示整数规划（设施选址、背包问题），统一教学风格中文注释与可视化规范。
# 语法与规则：PuLP二进制变量与线性约束；中文字体配置；PNG输出（dpi=300）。
"""
整数规划优化演示
Integer Programming Optimization Demo

演示内容：设施选址问题
- 目标：最小化总成本（建设成本+运营成本）
- 约束：选择固定数量的地点，满足需求
- 方法：使用PuLP求解器的二进制变量

作者: AI Assistant
日期: 2024
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pulp
import warnings
warnings.filterwarnings('ignore')

# 路径与中文字体：移动到子目录后也能导入根目录的配置
import os, sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)
from font_config import setup_chinese_font
setup_chinese_font()

class IntegerProgrammingDemo:
    """整数规划演示类
    作用：封装设施选址与背包问题的建模、求解、可视化、情景分析与报告生成。
    设计：面向对象组织流程；共享结果通过 self.results 以便各方法复用。
    规则：中文输出、统一图表样式、PNG高分辨率保存。
    """
    
    def __init__(self):
        self.results = {}
        print("=" * 50)
        print("🏭 整数规划优化演示")
        print("Integer Programming Demo")
        print("=" * 50)
    
    def solve_facility_location(self):
        """设施选址问题
        作用：在候选地点中选择固定数量的设施以最小化总成本（建设+运营），并确保服务能力满足需求。
        语法要点：
        - LpProblem(name, LpMinimize)
        - 二进制变量 y_i ∈ {0,1} 表示是否建设
        - 目标函数：Σ (建设成本 + 年运营成本×5) · y_i
        - 约束：选址个数=3；Σ capacity_i · y_i ≥ total_demand
        原理：整数规划的0/1选址模型；极点最优性与组合选择。
        规则：中文输出、教学友好、图表统一样式与PNG保存。
        """
        print("\n🏭 设施选址优化问题")
        print("-" * 40)
        
        # 候选地点
        locations = ['北京', '上海', '广州', '成都', '西安']
        
        # 建设成本（万元）
        construction_cost = [500, 600, 450, 350, 300]
        
        # 年运营成本（万元）
        operating_cost = [200, 250, 180, 150, 120]
        
        # 服务能力（万件/年）
        capacity = [1000, 1200, 800, 600, 500]
        
        # 需求量
        total_demand = 2000  # 万件/年
        
        print("候选地点信息:")
        df_locations = pd.DataFrame({
            '地点': locations,
            '建设成本(万元)': construction_cost,
            '运营成本(万元/年)': operating_cost,
            '服务能力(万件/年)': capacity
        })
        print(df_locations.to_string(index=False))
        print(f"\n总需求量: {total_demand} 万件/年")
        
        # 使用PuLP求解
        prob = pulp.LpProblem("设施选址", pulp.LpMinimize)
        
        # 决策变量：是否在地点i建设设施（0或1）
        y = [pulp.LpVariable(f"y{i}", cat='Binary') for i in range(5)]
        
        # 目标函数：最小化总成本（建设成本+5年运营成本）
        total_cost = pulp.lpSum([(construction_cost[i] + 5 * operating_cost[i]) * y[i] 
                                for i in range(5)])
        prob += total_cost
        
        # 约束条件
        # 1. 选择恰好3个地点
        prob += pulp.lpSum(y) == 3
        
        # 2. 满足需求量
        prob += pulp.lpSum([capacity[i] * y[i] for i in range(5)]) >= total_demand
        
        # 求解
        prob.solve(pulp.PULP_CBC_CMD(msg=0))
        
        # 结果
        selected = [int(y[i].varValue) for i in range(5)]
        min_cost = pulp.value(prob.objective)
        
print(f"\n最优选址方案：")
        selected_locations = []
        total_capacity = 0
        selected_details = []
        
        for i in range(5):
            if selected[i]:
                selected_locations.append(locations[i])
                total_capacity += capacity[i]
                selected_details.append({
                    'location': locations[i],
                    'construction_cost': construction_cost[i],
                    'operating_cost': operating_cost[i],
                    'capacity': capacity[i]
                })
                print(f"  ✓ {locations[i]} - 建设成本: {construction_cost[i]}万元, "
                      f"年运营成本: {operating_cost[i]}万元, 服务能力: {capacity[i]}万件/年")
        
print(f"\n方案总结：")
        print(f"  选中地点: {', '.join(selected_locations)}")
        print(f"  总服务能力: {total_capacity} 万件/年")
        print(f"  需求满足率: {total_capacity/total_demand*100:.1f}%")
        print(f"  总成本(5年): {min_cost:.2f} 万元")
        
        # 保存结果
        self.results = {
            'locations': locations,
            'selected': selected,
            'selected_locations': selected_locations,
            'selected_details': selected_details,
            'construction_cost': construction_cost,
            'operating_cost': operating_cost,
            'capacity': capacity,
            'total_capacity': total_capacity,
            'total_demand': total_demand,
            'min_cost': min_cost
        }
        
        return selected, min_cost
    
    def solve_knapsack_problem(self):
        """背包问题
        作用：在容量约束下选择价值最大的物品组合，演示0/1整数规划。
        语法要点：
        - LpProblem(name, LpMaximize)
        - 二进制变量 x_i ∈ {0,1}
        - 目标函数：Σ v_i x_i；约束：Σ w_i x_i ≤ C
        原理：组合优化的典型问题；价值密度可提供启发式直觉。
        规则：中文输出、教学友好、图表统一样式与PNG保存。
        """
        print("\n🎒 背包问题演示")
        print("-" * 30)
        
        # 物品数据
        items = ['笔记本电脑', '平板电脑', '智能手机', '相机', '充电宝']
        values = [3000, 1500, 2000, 1200, 300]  # 价值（元）
        weights = [2.5, 1.2, 0.5, 0.8, 0.6]    # 重量（kg）
        
        # 背包容量
        capacity = 4.0  # kg
        
        print("物品信息:")
        df_items = pd.DataFrame({
            '物品': items,
            '价值(元)': values,
            '重量(kg)': weights,
            '价值密度(元/kg)': [v/w for v, w in zip(values, weights)]
        })
        print(df_items.to_string(index=False))
        print(f"\n背包容量: {capacity} kg")
        
        # 使用PuLP求解
        prob = pulp.LpProblem("背包问题", pulp.LpMaximize)
        
        # 决策变量：是否选择物品i（0或1）
        x = [pulp.LpVariable(f"x{i}", cat='Binary') for i in range(len(items))]
        
        # 目标函数：最大化总价值
        prob += pulp.lpSum([values[i] * x[i] for i in range(len(items))])
        
        # 约束条件：重量不超过背包容量
        prob += pulp.lpSum([weights[i] * x[i] for i in range(len(items))]) <= capacity
        
        # 求解
        prob.solve(pulp.PULP_CBC_CMD(msg=0))
        
        # 结果
        selected_items = [int(x[i].varValue) for i in range(len(items))]
        max_value = pulp.value(prob.objective)
        total_weight = sum(weights[i] * selected_items[i] for i in range(len(items)))
        
print(f"\n最优选择方案：")
        selected_item_names = []
        for i in range(len(items)):
            if selected_items[i]:
                selected_item_names.append(items[i])
                print(f"  ✓ {items[i]} - 价值: {values[i]}元, 重量: {weights[i]}kg")
        
print(f"\n方案总结：")
        print(f"  选中物品: {', '.join(selected_item_names)}")
        print(f"  总价值: {max_value:.0f} 元")
        print(f"  总重量: {total_weight:.1f} kg")
        print(f"  容量利用率: {total_weight/capacity*100:.1f}%")
        
        # 保存背包问题结果
        self.results['knapsack'] = {
            'items': items,
            'selected_items': selected_items,
            'selected_item_names': selected_item_names,
            'values': values,
            'weights': weights,
            'max_value': max_value,
            'total_weight': total_weight,
            'capacity': capacity
        }
        
        return selected_items, max_value
    
    def visualize_results(self):
        """可视化结果
        作用：多维度展示选址、成本、背包选择与价值密度分析，统一中文标签和样式。
        规则：figsize统一；网格 alpha=0.3；PNG输出（dpi=300）。
        """
        if not self.results:
            print("⚠️ 请先运行求解方法")
            return
        
        print("\n📈 生成可视化图表...")
        
        # 设置统一图表样式
        plt.style.use('seaborn-v0_8')
        
        # 创建2x3子图布局，展示更全面的分析
        if 'knapsack' in self.results:
            fig, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2, 3, figsize=(18, 12))
        else:
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

        # 1. 设施选址结果
        colors = ['#FF6B6B' if selected else '#DDD' 
                 for selected in self.results['selected']]
        bars1 = ax1.bar(self.results['locations'], self.results['capacity'], color=colors)
        ax1.set_title('设施选址结果', fontsize=14, fontweight='bold')
        ax1.set_ylabel('服务能力 (万件/年)')
        ax1.tick_params(axis='x', rotation=45)
        ax1.grid(True, alpha=0.3)
        
        # 添加选中标记
        for i, (bar, selected) in enumerate(zip(bars1, self.results['selected'])):
            if selected:
                ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 20,
                        '✓', ha='center', va='bottom', fontsize=16, color='red')
        
        # 2. 成本结构分析
        selected_indices = [i for i, selected in enumerate(self.results['selected']) if selected]
        selected_locations = [self.results['locations'][i] for i in selected_indices]
        construction_costs = [self.results['construction_cost'][i] for i in selected_indices]
        operating_costs = [self.results['operating_cost'][i] * 5 for i in selected_indices]  # 5年运营成本
        
        x_pos = np.arange(len(selected_locations))
        width = 0.35
        
        bars2_1 = ax2.bar(x_pos - width/2, construction_costs, width, 
                         label='建设成本', color='#FF9999', alpha=0.8)
        bars2_2 = ax2.bar(x_pos + width/2, operating_costs, width, 
                         label='5年运营成本', color='#99CCFF', alpha=0.8)
        
        ax2.set_title('选中设施成本分析', fontsize=14, fontweight='bold')
        ax2.set_ylabel('成本 (万元)')
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels(selected_locations, rotation=45)
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        # 添加成本标签
        for bars in [bars2_1, bars2_2]:
            for bar in bars:
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width()/2, height + 10,
                        f'{height:.0f}', ha='center', va='bottom')
        
        # 3. 成本效益分析
        cost_efficiency = []
        for i in selected_indices:
            total_cost = self.results['construction_cost'][i] + 5 * self.results['operating_cost'][i]
            efficiency = self.results['capacity'][i] / total_cost  # 万件/万元
            cost_efficiency.append(efficiency)
        
        bars3 = ax3.bar(selected_locations, cost_efficiency, 
                       color=['#32CD32', '#FFD700', '#FF6347'])
        ax3.set_title('成本效益分析', fontsize=14, fontweight='bold')
        ax3.set_ylabel('效益 (万件/万元)')
        ax3.tick_params(axis='x', rotation=45)
        ax3.grid(True, alpha=0.3)
        
        # 添加效益标签
        for bar, value in zip(bars3, cost_efficiency):
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{value:.2f}', ha='center', va='bottom')
        
        if 'knapsack' in self.results:
            knapsack = self.results['knapsack']
            
            # 4. 背包问题 - 物品选择
            colors = ['#32CD32' if selected else '#DDD' 
                     for selected in knapsack['selected_items']]
            bars4 = ax4.bar(knapsack['items'], knapsack['values'], color=colors)
            ax4.set_title('背包问题 - 物品选择', fontsize=14, fontweight='bold')
            ax4.set_ylabel('价值 (元)')
            ax4.tick_params(axis='x', rotation=45)
            ax4.grid(True, alpha=0.3)
            
            # 添加选中标记
            for i, (bar, selected) in enumerate(zip(bars4, knapsack['selected_items'])):
                if selected:
                    ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 50,
                            '✓', ha='center', va='bottom', fontsize=16, color='red')
            
            # 5. 价值密度对比
            value_density = [v/w for v, w in zip(knapsack['values'], knapsack['weights'])]
            colors5 = ['#32CD32' if selected else '#DDD' 
                      for selected in knapsack['selected_items']]
            
            bars5 = ax5.bar(knapsack['items'], value_density, color=colors5)
            ax5.set_title('物品价值密度对比', fontsize=14, fontweight='bold')
            ax5.set_ylabel('价值密度 (元/kg)')
            ax5.tick_params(axis='x', rotation=45)
            ax5.grid(True, alpha=0.3)
            
            # 添加密度标签
            for bar, value in zip(bars5, value_density):
                ax5.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 50,
                        f'{value:.0f}', ha='center', va='bottom')
            
            # 6. 背包容量利用分析
            selected_weights = [knapsack['weights'][i] for i in range(len(knapsack['items'])) 
                               if knapsack['selected_items'][i]]
            selected_values = [knapsack['values'][i] for i in range(len(knapsack['items'])) 
                              if knapsack['selected_items'][i]]
            
            # 饼图显示重量分布
            if selected_weights:
                selected_names = [knapsack['items'][i] for i in range(len(knapsack['items'])) 
                                 if knapsack['selected_items'][i]]
                colors6 = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFD93D', '#6BCF7F'][:len(selected_weights)]
                
                wedges, texts, autotexts = ax6.pie(selected_weights, labels=selected_names, 
                                                  colors=colors6, autopct='%1.1f%%', startangle=90)
                ax6.set_title('选中物品重量分布', fontsize=14, fontweight='bold')
                
                # 添加总重量信息
                total_weight = sum(selected_weights)
                ax6.text(0, -1.3, f'总重量: {total_weight:.1f}kg / {knapsack["capacity"]:.1f}kg', 
                        ha='center', va='center', fontsize=12, 
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))
        else:
            # 如果没有背包问题，显示设施地理分布（简化版）
            ax4.axis('off')
            ax4.text(0.5, 0.5, '背包问题未运行', ha='center', va='center', 
                    transform=ax4.transAxes, fontsize=16)
        
        plt.tight_layout()
        save_path = os.path.join(BASE_DIR, 'integer_programming_results.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        
print("可视化图表已保存为 'integer_programming_results.png'")
    
    def scenario_analysis(self):
        """情景分析
        作用：考察需求量变化对选址与成本的影响，输出不同情景下的最优方案与成本。
        语法要点：重新构建并求解选址模型，参数为不同需求倍数。
        规则：仅教学用途，保持中文输出与结构化展示。
        """
        if not self.results:
            print("⚠️ 请先运行求解方法")
            return
        
        print("\n🔍 情景分析")
        print("-" * 30)
        
        # 分析不同需求量下的最优方案
        print("1. 需求量变化影响分析:")
        base_demand = self.results['total_demand']
        
        for demand_change in [0.8, 0.9, 1.1, 1.2]:  # 需求量变化倍数
            new_demand = base_demand * demand_change
            
            # 重新求解
            prob = pulp.LpProblem("情景分析", pulp.LpMinimize)
            y = [pulp.LpVariable(f"y{i}", cat='Binary') for i in range(5)]
            
            # 目标函数
            total_cost = pulp.lpSum([(self.results['construction_cost'][i] + 
                                    5 * self.results['operating_cost'][i]) * y[i] 
                                   for i in range(5)])
            prob += total_cost
            
            # 约束条件
            prob += pulp.lpSum(y) == 3
            prob += pulp.lpSum([self.results['capacity'][i] * y[i] 
                               for i in range(5)]) >= new_demand
            
            try:
                prob.solve(pulp.PULP_CBC_CMD(msg=0))
                if prob.status == 1:  # 最优解
                    new_cost = pulp.value(prob.objective)
                    selected_new = [int(y[i].varValue) for i in range(5)]
                    selected_locations_new = [self.results['locations'][i] 
                                            for i in range(5) if selected_new[i]]
                    
                    print(f"  需求量 {new_demand:.0f} 万件/年:")
                    print(f"    选中地点: {', '.join(selected_locations_new)}")
                    print(f"    总成本: {new_cost:.2f} 万元")
                else:
                    print(f"  需求量 {new_demand:.0f} 万件/年: 无可行解")
            except:
                print(f"  需求量 {new_demand:.0f} 万件/年: 求解失败")
    
    def generate_report(self):
        """生成详细报告
        作用：结构化总结优化目标、关键结果、成本分析与管理建议，便于教学与决策。
        规则：条理清晰、中文描述、数值格式统一。
        """
        if not self.results:
            print("⚠️ 请先运行求解方法")
            return
        
        print("\n" + "="*50)
print("整数规划优化报告")
        print("="*50)
        
        print(f"\n🎯 设施选址问题:")
        print(f"  • 优化目标: 最小化总成本")
        print(f"  • 决策变量: 是否在候选地点建设设施")
        print(f"  • 约束条件: 选择3个地点，满足需求")
        
print(f"\n最优方案：")
        for detail in self.results['selected_details']:
            print(f"  • {detail['location']}: 建设成本 {detail['construction_cost']}万元, "
                  f"年运营成本 {detail['operating_cost']}万元, "
                  f"服务能力 {detail['capacity']}万件/年")
        
print(f"\n成本分析：")
        total_construction = sum(detail['construction_cost'] 
                               for detail in self.results['selected_details'])
        total_operating = sum(detail['operating_cost'] * 5 
                            for detail in self.results['selected_details'])
        print(f"  • 总建设成本: {total_construction:.2f} 万元")
        print(f"  • 5年运营成本: {total_operating:.2f} 万元")
        print(f"  • 总成本: {self.results['min_cost']:.2f} 万元")
        
        print(f"\n📈 服务能力:")
        print(f"  • 总服务能力: {self.results['total_capacity']} 万件/年")
        print(f"  • 需求满足率: {self.results['total_capacity']/self.results['total_demand']*100:.1f}%")
        
        if 'knapsack' in self.results:
            print(f"\n🎒 背包问题结果:")
            knapsack = self.results['knapsack']
            print(f"  • 选中物品: {', '.join(knapsack['selected_item_names'])}")
            print(f"  • 总价值: {knapsack['max_value']:.0f} 元")
            print(f"  • 总重量: {knapsack['total_weight']:.1f} kg")
            print(f"  • 容量利用率: {knapsack['total_weight']/knapsack['capacity']*100:.1f}%")
        
print(f"\n管理建议：")
        if self.results['total_capacity'] / self.results['total_demand'] < 1.1:
            print(f"  • 服务能力余量较小，建议考虑增加备用方案")
        
        # 找出成本效益最好的地点
        cost_efficiency = []
        for detail in self.results['selected_details']:
            total_cost_per_location = detail['construction_cost'] + 5 * detail['operating_cost']
            efficiency = detail['capacity'] / total_cost_per_location
            cost_efficiency.append((detail['location'], efficiency))
        
        best_location = max(cost_efficiency, key=lambda x: x[1])
        print(f"  • 成本效益最佳地点: {best_location[0]} "
              f"({best_location[1]:.2f} 万件/万元)")
        
        print("="*50)

def main():
    """主函数
    作用：顺序运行选址、背包、可视化、情景分析与报告。
    使用规则：脚本运行时触发；导入为模块时不自动执行。
    """
    # 创建演示实例
    demo = IntegerProgrammingDemo()
    
    # 求解设施选址问题
    selected, min_cost = demo.solve_facility_location()
    
    # 求解背包问题
    knapsack_solution, max_value = demo.solve_knapsack_problem()
    
    # 生成可视化
    demo.visualize_results()
    
    # 情景分析
    demo.scenario_analysis()
    
    # 生成报告
    demo.generate_report()
    
    print(f"\n🎉 整数规划演示完成！")
    print(f"设施选址最优解: {[i for i, s in enumerate(selected) if s]}")
    print(f"最小成本: {min_cost:.2f} 万元")
    print(f"背包问题最大价值: {max_value:.0f} 元")

if __name__ == "__main__":
    main()