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

# 使用自定义字体配置支持中文
from font_config import setup_chinese_font
setup_chinese_font()

class IntegerProgrammingDemo:
    """整数规划演示类"""
    
    def __init__(self):
        self.results = {}
        print("=" * 50)
        print("🏭 整数规划优化演示")
        print("Integer Programming Demo")
        print("=" * 50)
    
    def solve_facility_location(self):
        """
        整数规划演示 - 设施选址问题
        
        问题描述：
        公司需要在5个候选地点中选择3个建设配送中心，
        以最小化总成本（建设成本+运营成本）
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
        
        print(f"\n✅ 最优选址方案:")
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
        
        print(f"\n📊 方案总结:")
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
        """
        背包问题演示 - 另一个经典整数规划问题
        
        问题描述：
        在有限的背包容量下，选择价值最大的物品组合
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
        
        print(f"\n✅ 最优选择方案:")
        selected_item_names = []
        for i in range(len(items)):
            if selected_items[i]:
                selected_item_names.append(items[i])
                print(f"  ✓ {items[i]} - 价值: {values[i]}元, 重量: {weights[i]}kg")
        
        print(f"\n📊 方案总结:")
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
        """可视化结果"""
        if not self.results:
            print("⚠️ 请先运行求解方法")
            return
        
        print("\n📈 生成可视化图表...")
        
        # 创建子图
        fig = plt.figure(figsize=(16, 12))
        
        # 1. 设施选址结果
        ax1 = plt.subplot(2, 3, 1)
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
        
        # 2. 成本分析
        ax2 = plt.subplot(2, 3, 2)
        selected_locations = [self.results['locations'][i] for i in range(5) 
                             if self.results['selected'][i]]
        construction_costs = [self.results['construction_cost'][i] for i in range(5) 
                             if self.results['selected'][i]]
        operating_costs = [self.results['operating_cost'][i] * 5 for i in range(5) 
                          if self.results['selected'][i]]  # 5年运营成本
        
        x_pos = np.arange(len(selected_locations))
        width = 0.35
        
        bars2a = ax2.bar(x_pos - width/2, construction_costs, width, 
                        label='建设成本', color='#FFB6C1')
        bars2b = ax2.bar(x_pos + width/2, operating_costs, width,
                        label='5年运营成本', color='#87CEEB')
        
        ax2.set_title('选中地点成本分析', fontsize=14, fontweight='bold')
        ax2.set_ylabel('成本 (万元)')
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels(selected_locations)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. 容量需求对比
        ax3 = plt.subplot(2, 3, 3)
        categories = ['总需求', '总供给']
        values = [self.results['total_demand'], self.results['total_capacity']]
        colors = ['#FF9999', '#66B2FF']
        
        bars3 = ax3.bar(categories, values, color=colors)
        ax3.set_title('供需平衡分析', fontsize=14, fontweight='bold')
        ax3.set_ylabel('数量 (万件/年)')
        ax3.grid(True, alpha=0.3)
        
        for bar, value in zip(bars3, values):
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 50,
                    f'{value}', ha='center', va='bottom')
        
        # 如果有背包问题结果，显示相关图表
        if 'knapsack' in self.results:
            # 4. 背包问题 - 物品选择
            ax4 = plt.subplot(2, 3, 4)
            knapsack = self.results['knapsack']
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
            
            # 5. 价值密度分析
            ax5 = plt.subplot(2, 3, 5)
            value_density = [v/w for v, w in zip(knapsack['values'], knapsack['weights'])]
            bars5 = ax5.bar(knapsack['items'], value_density, 
                           color=['#FFA500' if selected else '#DDD' 
                                 for selected in knapsack['selected_items']])
            ax5.set_title('价值密度分析', fontsize=14, fontweight='bold')
            ax5.set_ylabel('价值密度 (元/kg)')
            ax5.tick_params(axis='x', rotation=45)
            ax5.grid(True, alpha=0.3)
            
            # 6. 重量利用率
            ax6 = plt.subplot(2, 3, 6)
            weight_data = ['已用重量', '剩余容量']
            weight_values = [knapsack['total_weight'], 
                           knapsack['capacity'] - knapsack['total_weight']]
            colors = ['#FF6347', '#F0F0F0']
            
            wedges, texts, autotexts = ax6.pie(weight_values, labels=weight_data, 
                                              colors=colors, autopct='%1.1f%%',
                                              startangle=90)
            ax6.set_title('背包容量利用率', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig('integer_programming_results.png', 
                   dpi=300, bbox_inches='tight')
        plt.show()
        
        print("✅ 可视化图表已保存为 'integer_programming_results.png'")
    
    def scenario_analysis(self):
        """情景分析"""
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
        """生成详细报告"""
        if not self.results:
            print("⚠️ 请先运行求解方法")
            return
        
        print("\n" + "="*50)
        print("📋 整数规划优化报告")
        print("="*50)
        
        print(f"\n🎯 设施选址问题:")
        print(f"  • 优化目标: 最小化总成本")
        print(f"  • 决策变量: 是否在候选地点建设设施")
        print(f"  • 约束条件: 选择3个地点，满足需求")
        
        print(f"\n📊 最优方案:")
        for detail in self.results['selected_details']:
            print(f"  • {detail['location']}: 建设成本 {detail['construction_cost']}万元, "
                  f"年运营成本 {detail['operating_cost']}万元, "
                  f"服务能力 {detail['capacity']}万件/年")
        
        print(f"\n💰 成本分析:")
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
        
        print(f"\n💡 管理建议:")
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
    """主函数"""
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