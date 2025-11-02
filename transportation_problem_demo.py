"""
运输问题优化演示
Transportation Problem Optimization Demo

演示内容：供应链优化问题
- 目标：最小化运输成本
- 约束：供应量和需求量平衡
- 方法：使用PuLP求解器和运输单纯形法

作者: AI Assistant
日期: 2024
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pulp
import warnings
warnings.filterwarnings('ignore')

# 使用自定义字体配置支持中文
from font_config import setup_chinese_font
setup_chinese_font()

class TransportationProblemDemo:
    """运输问题演示类"""
    
    def __init__(self):
        self.results = {}
        print("=" * 50)
        print("🚛 运输问题优化演示")
        print("Transportation Problem Demo")
        print("=" * 50)
    
    def solve_basic_transportation(self):
        """
        基础运输问题演示 - 供应链优化
        
        问题描述：
        3个工厂向4个仓库运输产品，最小化运输成本
        """
        print("\n🚛 基础运输问题 - 供应链优化")
        print("-" * 40)
        
        # 工厂和仓库
        factories = ['工厂A', '工厂B', '工厂C']
        warehouses = ['仓库1', '仓库2', '仓库3', '仓库4']
        
        # 供应量（吨）
        supply = [300, 400, 500]
        
        # 需求量（吨）
        demand = [250, 350, 400, 200]
        
        # 运输成本矩阵（元/吨）
        cost_matrix = np.array([
            [8, 6, 10, 9],   # 工厂A到各仓库
            [9, 12, 13, 7],  # 工厂B到各仓库
            [14, 9, 16, 5]   # 工厂C到各仓库
        ])
        
        print("供需信息:")
        print(f"工厂供应量: {dict(zip(factories, supply))}")
        print(f"仓库需求量: {dict(zip(warehouses, demand))}")
        print(f"总供应量: {sum(supply)} 吨")
        print(f"总需求量: {sum(demand)} 吨")
        
        print(f"\n运输成本矩阵 (元/吨):")
        cost_df = pd.DataFrame(cost_matrix, index=factories, columns=warehouses)
        print(cost_df)
        
        # 检查平衡性
        original_warehouses = warehouses.copy()
        original_demand = demand.copy()
        
        if sum(supply) != sum(demand):
            print(f"⚠️  非平衡运输问题：供应量 ≠ 需求量")
            if sum(supply) > sum(demand):
                # 添加虚拟仓库
                demand.append(sum(supply) - sum(demand))
                warehouses.append('虚拟仓库')
                cost_matrix = np.column_stack([cost_matrix, np.zeros(3)])
                print(f"添加虚拟仓库，需求量: {demand[-1]} 吨")
            else:
                # 添加虚拟工厂
                supply.append(sum(demand) - sum(supply))
                factories.append('虚拟工厂')
                cost_matrix = np.vstack([cost_matrix, np.zeros(len(warehouses))])
                print(f"添加虚拟工厂，供应量: {supply[-1]} 吨")
        
        # 使用PuLP求解
        prob = pulp.LpProblem("运输问题", pulp.LpMinimize)
        
        # 决策变量：从工厂i到仓库j的运输量
        x = {}
        for i in range(len(factories)):
            for j in range(len(warehouses)):
                x[i,j] = pulp.LpVariable(f"x_{i}_{j}", lowBound=0)
        
        # 目标函数：最小化运输成本
        prob += pulp.lpSum([cost_matrix[i][j] * x[i,j] 
                           for i in range(len(factories)) 
                           for j in range(len(warehouses))])
        
        # 约束条件
        # 1. 供应约束
        for i in range(len(factories)):
            prob += pulp.lpSum([x[i,j] for j in range(len(warehouses))]) == supply[i]
        
        # 2. 需求约束
        for j in range(len(warehouses)):
            prob += pulp.lpSum([x[i,j] for i in range(len(factories))]) == demand[j]
        
        # 求解
        prob.solve(pulp.PULP_CBC_CMD(msg=0))
        
        # 结果
        solution_matrix = np.zeros((len(factories), len(warehouses)))
        for i in range(len(factories)):
            for j in range(len(warehouses)):
                solution_matrix[i][j] = x[i,j].varValue
        
        min_transport_cost = pulp.value(prob.objective)
        
        print(f"\n✅ 最优运输方案:")
        solution_df = pd.DataFrame(solution_matrix, 
                                 index=factories, 
                                 columns=warehouses)
        print(solution_df.round(1))
        
        print(f"\n📊 运输成本分析:")
        print(f"  最小运输成本: {min_transport_cost:.2f} 元")
        
        # 计算各路线成本
        print(f"\n🛣️  主要运输路线:")
        route_details = []
        for i in range(len(factories)):
            for j in range(len(warehouses)):
                if solution_matrix[i][j] > 0:
                    route_cost = solution_matrix[i][j] * cost_matrix[i][j]
                    route_details.append({
                        'from': factories[i],
                        'to': warehouses[j],
                        'quantity': solution_matrix[i][j],
                        'unit_cost': cost_matrix[i][j],
                        'total_cost': route_cost
                    })
                    print(f"  {factories[i]} → {warehouses[j]}: "
                          f"{solution_matrix[i][j]:.1f}吨, 成本: {route_cost:.2f}元")
        
        # 保存结果
        self.results['basic'] = {
            'factories': factories,
            'warehouses': warehouses,
            'original_warehouses': original_warehouses,
            'supply': supply,
            'demand': demand,
            'original_demand': original_demand,
            'cost_matrix': cost_matrix,
            'solution_matrix': solution_matrix,
            'min_cost': min_transport_cost,
            'route_details': route_details
        }
        
        return solution_matrix, min_transport_cost
    
    def solve_multi_product_transportation(self):
        """
        多产品运输问题演示
        
        问题描述：
        2个工厂生产2种产品，向3个市场供应
        """
        print("\n📦 多产品运输问题")
        print("-" * 30)
        
        # 工厂、产品、市场
        factories = ['工厂X', '工厂Y']
        products = ['产品P1', '产品P2']
        markets = ['市场M1', '市场M2', '市场M3']
        
        # 各工厂各产品的供应量
        supply_matrix = np.array([
            [200, 150],  # 工厂X的P1, P2供应量
            [180, 220]   # 工厂Y的P1, P2供应量
        ])
        
        # 各市场各产品的需求量
        demand_matrix = np.array([
            [120, 100],  # 市场M1的P1, P2需求量
            [140, 130],  # 市场M2的P1, P2需求量
            [120, 140]   # 市场M3的P1, P2需求量
        ])
        
        # 运输成本矩阵 [工厂][产品][市场]
        cost_tensor = np.array([
            [[5, 7, 6],   # 工厂X的P1到各市场
             [6, 8, 7]],  # 工厂X的P2到各市场
            [[8, 6, 9],   # 工厂Y的P1到各市场
             [7, 5, 8]]   # 工厂Y的P2到各市场
        ])
        
        print("供应信息:")
        supply_df = pd.DataFrame(supply_matrix, index=factories, columns=products)
        print(supply_df)
        
        print("\n需求信息:")
        demand_df = pd.DataFrame(demand_matrix, index=markets, columns=products)
        print(demand_df)
        
        print(f"\n各产品总供应量: P1={supply_matrix[:, 0].sum()}, P2={supply_matrix[:, 1].sum()}")
        print(f"各产品总需求量: P1={demand_matrix[:, 0].sum()}, P2={demand_matrix[:, 1].sum()}")
        
        # 使用PuLP求解
        prob = pulp.LpProblem("多产品运输问题", pulp.LpMinimize)
        
        # 决策变量：从工厂i的产品p到市场j的运输量
        x = {}
        for i in range(len(factories)):
            for p in range(len(products)):
                for j in range(len(markets)):
                    x[i,p,j] = pulp.LpVariable(f"x_{i}_{p}_{j}", lowBound=0)
        
        # 目标函数：最小化总运输成本
        prob += pulp.lpSum([cost_tensor[i][p][j] * x[i,p,j] 
                           for i in range(len(factories))
                           for p in range(len(products))
                           for j in range(len(markets))])
        
        # 约束条件
        # 1. 供应约束：每个工厂每种产品的供应量限制
        for i in range(len(factories)):
            for p in range(len(products)):
                prob += pulp.lpSum([x[i,p,j] for j in range(len(markets))]) <= supply_matrix[i][p]
        
        # 2. 需求约束：每个市场每种产品的需求量满足
        for j in range(len(markets)):
            for p in range(len(products)):
                prob += pulp.lpSum([x[i,p,j] for i in range(len(factories))]) >= demand_matrix[j][p]
        
        # 求解
        prob.solve(pulp.PULP_CBC_CMD(msg=0))
        
        # 结果
        min_cost = pulp.value(prob.objective)
        
        print(f"\n✅ 最优运输方案:")
        print(f"  最小运输成本: {min_cost:.2f} 元")
        
        print(f"\n🛣️  运输路线详情:")
        multi_route_details = []
        for i in range(len(factories)):
            for p in range(len(products)):
                for j in range(len(markets)):
                    quantity = x[i,p,j].varValue
                    if quantity > 0:
                        cost = quantity * cost_tensor[i][p][j]
                        multi_route_details.append({
                            'factory': factories[i],
                            'product': products[p],
                            'market': markets[j],
                            'quantity': quantity,
                            'unit_cost': cost_tensor[i][p][j],
                            'total_cost': cost
                        })
                        print(f"  {factories[i]} {products[p]} → {markets[j]}: "
                              f"{quantity:.1f}单位, 成本: {cost:.2f}元")
        
        # 保存多产品运输结果
        self.results['multi_product'] = {
            'factories': factories,
            'products': products,
            'markets': markets,
            'supply_matrix': supply_matrix,
            'demand_matrix': demand_matrix,
            'cost_tensor': cost_tensor,
            'min_cost': min_cost,
            'route_details': multi_route_details
        }
        
        return min_cost
    
    def visualize_results(self):
        """可视化结果"""
        if not self.results:
            print("⚠️ 请先运行求解方法")
            return
        
        print("\n📈 生成可视化图表...")
        
        # 创建子图
        fig = plt.figure(figsize=(18, 12))
        
        if 'basic' in self.results:
            basic = self.results['basic']
            
            # 1. 运输成本热力图
            ax1 = plt.subplot(2, 3, 1)
            # 只显示原始仓库的成本
            original_cost_matrix = basic['cost_matrix'][:, :len(basic['original_warehouses'])]
            sns.heatmap(original_cost_matrix, 
                       xticklabels=basic['original_warehouses'],
                       yticklabels=basic['factories'][:len(original_cost_matrix)],
                       annot=True, fmt='d', cmap='YlOrRd', ax=ax1)
            ax1.set_title('运输成本矩阵 (元/吨)', fontsize=14, fontweight='bold')
            
            # 2. 运输方案热力图
            ax2 = plt.subplot(2, 3, 2)
            # 只显示原始仓库的运输方案
            original_solution = basic['solution_matrix'][:len(original_cost_matrix), :len(basic['original_warehouses'])]
            sns.heatmap(original_solution, 
                       xticklabels=basic['original_warehouses'],
                       yticklabels=basic['factories'][:len(original_cost_matrix)],
                       annot=True, fmt='.1f', cmap='Blues', ax=ax2)
            ax2.set_title('最优运输方案 (吨)', fontsize=14, fontweight='bold')
            
            # 3. 供需平衡分析
            ax3 = plt.subplot(2, 3, 3)
            categories = ['总供应', '总需求']
            values = [sum(basic['supply'][:len(original_cost_matrix)]), sum(basic['original_demand'])]
            colors = ['#66B2FF', '#FF9999']
            
            bars = ax3.bar(categories, values, color=colors)
            ax3.set_title('供需平衡分析', fontsize=14, fontweight='bold')
            ax3.set_ylabel('数量 (吨)')
            ax3.grid(True, alpha=0.3)
            
            for bar, value in zip(bars, values):
                ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 10,
                        f'{value}', ha='center', va='bottom')
            
            # 4. 运输路线成本分析
            ax4 = plt.subplot(2, 3, 4)
            if basic['route_details']:
                route_costs = [detail['total_cost'] for detail in basic['route_details']]
                route_labels = [f"{detail['from'][:2]}-{detail['to'][:2]}" 
                               for detail in basic['route_details']]
                
                bars = ax4.bar(range(len(route_costs)), route_costs, 
                              color=plt.cm.Set3(np.linspace(0, 1, len(route_costs))))
                ax4.set_title('各路线运输成本', fontsize=14, fontweight='bold')
                ax4.set_ylabel('成本 (元)')
                ax4.set_xticks(range(len(route_labels)))
                ax4.set_xticklabels(route_labels, rotation=45)
                ax4.grid(True, alpha=0.3)
        
        if 'multi_product' in self.results:
            multi = self.results['multi_product']
            
            # 5. 多产品供需对比
            ax5 = plt.subplot(2, 3, 5)
            products = multi['products']
            supply_totals = [multi['supply_matrix'][:, i].sum() for i in range(len(products))]
            demand_totals = [multi['demand_matrix'][:, i].sum() for i in range(len(products))]
            
            x_pos = np.arange(len(products))
            width = 0.35
            
            bars1 = ax5.bar(x_pos - width/2, supply_totals, width, 
                           label='总供应', color='#87CEEB')
            bars2 = ax5.bar(x_pos + width/2, demand_totals, width,
                           label='总需求', color='#FFB6C1')
            
            ax5.set_title('多产品供需对比', fontsize=14, fontweight='bold')
            ax5.set_ylabel('数量')
            ax5.set_xticks(x_pos)
            ax5.set_xticklabels(products)
            ax5.legend()
            ax5.grid(True, alpha=0.3)
            
            # 6. 多产品运输成本分布
            ax6 = plt.subplot(2, 3, 6)
            if multi['route_details']:
                product_costs = {}
                for detail in multi['route_details']:
                    product = detail['product']
                    if product not in product_costs:
                        product_costs[product] = 0
                    product_costs[product] += detail['total_cost']
                
                products_list = list(product_costs.keys())
                costs_list = list(product_costs.values())
                
                wedges, texts, autotexts = ax6.pie(costs_list, labels=products_list, 
                                                  autopct='%1.1f%%', startangle=90,
                                                  colors=['#FF9999', '#66B2FF'])
                ax6.set_title('各产品运输成本占比', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig('transportation_results.png', 
                   dpi=300, bbox_inches='tight')
        plt.show()
        
        print("✅ 可视化图表已保存为 'transportation_results.png'")
    
    def cost_sensitivity_analysis(self):
        """运输成本敏感性分析"""
        if 'basic' not in self.results:
            print("⚠️ 请先运行基础运输问题求解")
            return
        
        print("\n🔍 运输成本敏感性分析")
        print("-" * 30)
        
        basic = self.results['basic']
        base_cost = basic['min_cost']
        
        # 分析关键路线成本变化的影响
        print("关键路线成本变化影响:")
        
        for route in basic['route_details'][:3]:  # 分析前3条主要路线
            print(f"\n  {route['from']} → {route['to']} 路线:")
            
            for cost_change in [-20, -10, 10, 20]:  # 成本变化百分比
                # 这里简化处理，实际应该重新求解整个问题
                estimated_cost_change = route['total_cost'] * cost_change / 100
                new_total_cost = base_cost + estimated_cost_change
                
                print(f"    成本{cost_change:+d}% → 预估总成本: {new_total_cost:.2f} 元 "
                      f"(变化: {estimated_cost_change:+.2f})")
    
    def generate_report(self):
        """生成详细报告"""
        if not self.results:
            print("⚠️ 请先运行求解方法")
            return
        
        print("\n" + "="*50)
        print("📋 运输问题优化报告")
        print("="*50)
        
        if 'basic' in self.results:
            basic = self.results['basic']
            print(f"\n🚛 基础运输问题:")
            print(f"  • 优化目标: 最小化运输成本")
            print(f"  • 工厂数量: {len(basic['factories'])}")
            print(f"  • 仓库数量: {len(basic['original_warehouses'])}")
            print(f"  • 最小运输成本: {basic['min_cost']:.2f} 元")
            
            print(f"\n📊 运输方案统计:")
            total_quantity = sum(detail['quantity'] for detail in basic['route_details'])
            print(f"  • 总运输量: {total_quantity:.1f} 吨")
            print(f"  • 平均运输成本: {basic['min_cost']/total_quantity:.2f} 元/吨")
            print(f"  • 活跃路线数: {len(basic['route_details'])}")
            
            # 找出成本最高和最低的路线
            if basic['route_details']:
                max_cost_route = max(basic['route_details'], key=lambda x: x['unit_cost'])
                min_cost_route = min(basic['route_details'], key=lambda x: x['unit_cost'])
                
                print(f"\n💰 路线成本分析:")
                print(f"  • 最高成本路线: {max_cost_route['from']} → {max_cost_route['to']} "
                      f"({max_cost_route['unit_cost']} 元/吨)")
                print(f"  • 最低成本路线: {min_cost_route['from']} → {min_cost_route['to']} "
                      f"({min_cost_route['unit_cost']} 元/吨)")
        
        if 'multi_product' in self.results:
            multi = self.results['multi_product']
            print(f"\n📦 多产品运输问题:")
            print(f"  • 工厂数量: {len(multi['factories'])}")
            print(f"  • 产品种类: {len(multi['products'])}")
            print(f"  • 市场数量: {len(multi['markets'])}")
            print(f"  • 最小运输成本: {multi['min_cost']:.2f} 元")
            
            # 各产品的运输成本分析
            product_costs = {}
            for detail in multi['route_details']:
                product = detail['product']
                if product not in product_costs:
                    product_costs[product] = 0
                product_costs[product] += detail['total_cost']
            
            print(f"\n📈 各产品运输成本:")
            for product, cost in product_costs.items():
                percentage = cost / multi['min_cost'] * 100
                print(f"  • {product}: {cost:.2f} 元 ({percentage:.1f}%)")
        
        print(f"\n💡 优化建议:")
        if 'basic' in self.results:
            basic = self.results['basic']
            if basic['route_details']:
                # 建议优化高成本路线
                high_cost_routes = [r for r in basic['route_details'] if r['unit_cost'] > 10]
                if high_cost_routes:
                    print(f"  • 考虑优化高成本路线，寻找替代运输方案")
                
                # 建议增加低成本路线的利用
                low_cost_routes = [r for r in basic['route_details'] if r['unit_cost'] < 8]
                if low_cost_routes:
                    print(f"  • 充分利用低成本路线，提高运输效率")
        
        print("="*50)

def main():
    """主函数"""
    # 创建演示实例
    demo = TransportationProblemDemo()
    
    # 求解基础运输问题
    solution_matrix, min_cost = demo.solve_basic_transportation()
    
    # 求解多产品运输问题
    multi_min_cost = demo.solve_multi_product_transportation()
    
    # 生成可视化
    demo.visualize_results()
    
    # 敏感性分析
    demo.cost_sensitivity_analysis()
    
    # 生成报告
    demo.generate_report()
    
    print(f"\n🎉 运输问题演示完成！")
    print(f"基础运输最小成本: {min_cost:.2f} 元")
    print(f"多产品运输最小成本: {multi_min_cost:.2f} 元")

if __name__ == "__main__":
    main()