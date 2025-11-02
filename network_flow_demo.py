"""
网络流优化演示
Network Flow Optimization Demo

演示内容：网络流问题
- 最大流问题：网络容量限制下的最大流量
- 最小费用流问题：在满足需求的前提下最小化成本
- 最短路径问题：寻找两点间的最短路径

作者: AI Assistant
日期: 2024
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
import pulp
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

# 使用自定义字体配置支持中文
from font_config import setup_chinese_font
setup_chinese_font()

class NetworkFlowDemo:
    """网络流优化演示类"""
    
    def __init__(self):
        self.results = {}
        self.graphs = {}
        print("=" * 50)
        print("🌐 网络流优化演示")
        print("Network Flow Optimization Demo")
        print("=" * 50)
    
    def solve_max_flow_problem(self):
        """
        最大流问题演示 - 供水网络优化
        
        问题描述：
        从水源到各个用户的供水网络，求最大供水量
        """
        print("\n💧 最大流问题 - 供水网络优化")
        print("-" * 40)
        
        # 创建网络图
        G = nx.DiGraph()
        
        # 节点：源点S，中间节点A,B,C,D，汇点T
        nodes = ['S', 'A', 'B', 'C', 'D', 'T']
        G.add_nodes_from(nodes)
        
        # 边和容量 (起点, 终点, 容量)
        edges_capacity = [
            ('S', 'A', 16), ('S', 'B', 13),
            ('A', 'B', 4), ('A', 'C', 12),
            ('B', 'D', 14), ('C', 'B', 9),
            ('C', 'T', 20), ('D', 'C', 7),
            ('D', 'T', 4)
        ]
        
        # 添加边
        for start, end, capacity in edges_capacity:
            G.add_edge(start, end, capacity=capacity, flow=0)
        
        print("网络结构:")
        print("节点: 水源S → 中间节点A,B,C,D → 用户T")
        print("边容量 (管道最大流量):")
        for start, end, capacity in edges_capacity:
            print(f"  {start} → {end}: {capacity} 单位/小时")
        
        # 使用NetworkX求解最大流
        max_flow_value, max_flow_dict = nx.maximum_flow(G, 'S', 'T')
        
        print(f"\n✅ 最大流结果:")
        print(f"  最大流量: {max_flow_value} 单位/小时")
        
        print(f"\n🌊 最优流量分配:")
        total_flow_used = 0
        flow_details = []
        for start in max_flow_dict:
            for end in max_flow_dict[start]:
                flow = max_flow_dict[start][end]
                if flow > 0:
                    capacity = G[start][end]['capacity']
                    utilization = flow / capacity * 100
                    flow_details.append({
                        'from': start,
                        'to': end,
                        'flow': flow,
                        'capacity': capacity,
                        'utilization': utilization
                    })
                    print(f"  {start} → {end}: {flow}/{capacity} "
                          f"(利用率: {utilization:.1f}%)")
                    total_flow_used += flow
        
        # 找出瓶颈边
        bottleneck_edges = [detail for detail in flow_details 
                           if detail['utilization'] >= 99.9]
        
        if bottleneck_edges:
            print(f"\n🚧 网络瓶颈:")
            for edge in bottleneck_edges:
                print(f"  {edge['from']} → {edge['to']}: 满负荷运行")
        
        # 保存结果
        self.results['max_flow'] = {
            'graph': G,
            'max_flow_value': max_flow_value,
            'flow_dict': max_flow_dict,
            'flow_details': flow_details,
            'bottleneck_edges': bottleneck_edges
        }
        self.graphs['max_flow'] = G
        
        return max_flow_value, max_flow_dict
    
    def solve_min_cost_flow_problem(self):
        """
        最小费用流问题演示 - 物流配送优化
        
        问题描述：
        从多个仓库向多个客户配送货物，最小化配送成本
        """
        print("\n🚚 最小费用流问题 - 物流配送优化")
        print("-" * 40)
        
        # 网络节点
        warehouses = ['仓库1', '仓库2']
        customers = ['客户A', '客户B', '客户C']
        
        # 供应量和需求量
        supply = {'仓库1': 100, '仓库2': 150}
        demand = {'客户A': 80, '客户B': 90, '客户C': 80}
        
        print("供需信息:")
        print(f"仓库供应量: {supply}")
        print(f"客户需求量: {demand}")
        print(f"总供应量: {sum(supply.values())}")
        print(f"总需求量: {sum(demand.values())}")
        
        # 运输成本和容量
        # (起点, 终点, 单位成本, 容量)
        transport_data = [
            ('仓库1', '客户A', 4, 60),
            ('仓库1', '客户B', 6, 70),
            ('仓库1', '客户C', 8, 50),
            ('仓库2', '客户A', 5, 50),
            ('仓库2', '客户B', 3, 80),
            ('仓库2', '客户C', 7, 60)
        ]
        
        print(f"\n运输成本和容量限制:")
        for start, end, cost, capacity in transport_data:
            print(f"  {start} → {end}: 成本{cost}元/单位, 容量{capacity}单位")
        
        # 使用PuLP求解最小费用流
        prob = pulp.LpProblem("最小费用流问题", pulp.LpMinimize)
        
        # 决策变量：从仓库i到客户j的运输量
        x = {}
        for start, end, cost, capacity in transport_data:
            x[start, end] = pulp.LpVariable(f"x_{start}_{end}", 
                                          lowBound=0, upBound=capacity)
        
        # 目标函数：最小化总运输成本
        prob += pulp.lpSum([cost * x[start, end] 
                           for start, end, cost, capacity in transport_data])
        
        # 约束条件
        # 1. 供应约束
        for warehouse in warehouses:
            prob += pulp.lpSum([x[warehouse, customer] 
                               for customer in customers 
                               if (warehouse, customer) in x]) <= supply[warehouse]
        
        # 2. 需求约束
        for customer in customers:
            prob += pulp.lpSum([x[warehouse, customer] 
                               for warehouse in warehouses 
                               if (warehouse, customer) in x]) >= demand[customer]
        
        # 求解
        prob.solve(pulp.PULP_CBC_CMD(msg=0))
        
        # 结果
        min_cost = pulp.value(prob.objective)
        
        print(f"\n✅ 最小费用流结果:")
        print(f"  最小运输成本: {min_cost:.2f} 元")
        
        print(f"\n🛣️  最优配送方案:")
        flow_solution = {}
        cost_details = []
        for start, end, cost, capacity in transport_data:
            flow = x[start, end].varValue
            if flow > 0:
                total_cost = flow * cost
                cost_details.append({
                    'from': start,
                    'to': end,
                    'flow': flow,
                    'unit_cost': cost,
                    'total_cost': total_cost,
                    'capacity': capacity,
                    'utilization': flow / capacity * 100
                })
                flow_solution[start, end] = flow
                print(f"  {start} → {end}: {flow:.1f}单位, "
                      f"成本: {total_cost:.2f}元")
        
        # 保存结果
        self.results['min_cost_flow'] = {
            'warehouses': warehouses,
            'customers': customers,
            'supply': supply,
            'demand': demand,
            'transport_data': transport_data,
            'min_cost': min_cost,
            'flow_solution': flow_solution,
            'cost_details': cost_details
        }
        
        return min_cost, flow_solution
    
    def solve_shortest_path_problem(self):
        """
        最短路径问题演示 - 城市交通网络
        
        问题描述：
        在城市交通网络中寻找从起点到终点的最短路径
        """
        print("\n🗺️  最短路径问题 - 城市交通网络")
        print("-" * 40)
        
        # 创建城市交通网络
        G = nx.Graph()
        
        # 城市节点
        cities = ['起点', '城市A', '城市B', '城市C', '城市D', '终点']
        G.add_nodes_from(cities)
        
        # 道路和距离 (城市1, 城市2, 距离km)
        roads = [
            ('起点', '城市A', 10), ('起点', '城市B', 15),
            ('城市A', '城市C', 12), ('城市A', '城市D', 15),
            ('城市B', '城市C', 8), ('城市B', '城市D', 7),
            ('城市C', '终点', 10), ('城市D', '终点', 12),
            ('城市A', '城市B', 6), ('城市C', '城市D', 5)
        ]
        
        # 添加边
        for city1, city2, distance in roads:
            G.add_edge(city1, city2, weight=distance)
        
        print("交通网络:")
        print("城市节点:", cities)
        print("道路距离:")
        for city1, city2, distance in roads:
            print(f"  {city1} ↔ {city2}: {distance} km")
        
        # 使用Dijkstra算法求最短路径
        shortest_path = nx.shortest_path(G, '起点', '终点', weight='weight')
        shortest_distance = nx.shortest_path_length(G, '起点', '终点', weight='weight')
        
        print(f"\n✅ 最短路径结果:")
        print(f"  最短距离: {shortest_distance} km")
        print(f"  最短路径: {' → '.join(shortest_path)}")
        
        # 计算路径详情
        path_details = []
        total_distance = 0
        for i in range(len(shortest_path) - 1):
            start = shortest_path[i]
            end = shortest_path[i + 1]
            distance = G[start][end]['weight']
            total_distance += distance
            path_details.append({
                'from': start,
                'to': end,
                'distance': distance,
                'cumulative': total_distance
            })
            print(f"  第{i+1}段: {start} → {end}, {distance} km "
                  f"(累计: {total_distance} km)")
        
        # 计算所有节点间的最短路径（用于分析网络连通性）
        all_shortest_paths = dict(nx.all_pairs_shortest_path_length(G))
        
        print(f"\n🌐 网络连通性分析:")
        print(f"  网络直径: {nx.diameter(G, weight='weight'):.1f} km")
        print(f"  平均路径长度: {nx.average_shortest_path_length(G, weight='weight'):.1f} km")
        
        # 保存结果
        self.results['shortest_path'] = {
            'graph': G,
            'cities': cities,
            'roads': roads,
            'shortest_path': shortest_path,
            'shortest_distance': shortest_distance,
            'path_details': path_details,
            'all_shortest_paths': all_shortest_paths
        }
        self.graphs['shortest_path'] = G
        
        return shortest_path, shortest_distance
    
    def visualize_results(self):
        """可视化网络流结果"""
        if not self.results:
            print("⚠️ 请先运行求解方法")
            return
        
        print("\n📈 生成网络流可视化图表...")
        
        # 创建子图
        fig = plt.figure(figsize=(20, 15))
        
        # 1. 最大流网络图
        if 'max_flow' in self.results:
            ax1 = plt.subplot(2, 3, 1)
            max_flow_data = self.results['max_flow']
            G = max_flow_data['graph']
            
            # 设置节点位置
            pos = {
                'S': (0, 1),
                'A': (1, 1.5),
                'B': (1, 0.5),
                'C': (2, 1.5),
                'D': (2, 0.5),
                'T': (3, 1)
            }
            
            # 绘制节点
            nx.draw_networkx_nodes(G, pos, node_color='lightblue', 
                                 node_size=1000, ax=ax1)
            nx.draw_networkx_labels(G, pos, font_size=12, ax=ax1)
            
            # 绘制边，根据流量设置粗细
            for start, end in G.edges():
                flow = max_flow_data['flow_dict'][start][end]
                capacity = G[start][end]['capacity']
                if flow > 0:
                    width = max(1, flow / 5)  # 根据流量调整线宽
                    nx.draw_networkx_edges(G, pos, [(start, end)], 
                                         width=width, edge_color='red', ax=ax1)
                    # 添加流量标签
                    x1, y1 = pos[start]
                    x2, y2 = pos[end]
                    ax1.text((x1+x2)/2, (y1+y2)/2, f'{flow}/{capacity}', 
                            fontsize=8, ha='center', 
                            bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8))
                else:
                    nx.draw_networkx_edges(G, pos, [(start, end)], 
                                         width=1, edge_color='gray', 
                                         style='dashed', ax=ax1)
            
            ax1.set_title(f'最大流网络 (最大流量: {max_flow_data["max_flow_value"]})', 
                         fontsize=14, fontweight='bold')
            ax1.axis('off')
        
        # 2. 最大流边利用率
        if 'max_flow' in self.results:
            ax2 = plt.subplot(2, 3, 2)
            flow_details = max_flow_data['flow_details']
            
            edges = [f"{detail['from']}-{detail['to']}" for detail in flow_details]
            utilizations = [detail['utilization'] for detail in flow_details]
            
            colors = ['red' if u >= 99.9 else 'orange' if u >= 80 else 'green' 
                     for u in utilizations]
            
            bars = ax2.bar(range(len(edges)), utilizations, color=colors)
            ax2.set_title('边容量利用率', fontsize=14, fontweight='bold')
            ax2.set_ylabel('利用率 (%)')
            ax2.set_xticks(range(len(edges)))
            ax2.set_xticklabels(edges, rotation=45)
            ax2.grid(True, alpha=0.3)
            ax2.axhline(y=100, color='red', linestyle='--', alpha=0.7)
        
        # 3. 最小费用流成本分析
        if 'min_cost_flow' in self.results:
            ax3 = plt.subplot(2, 3, 3)
            mcf_data = self.results['min_cost_flow']
            
            if mcf_data['cost_details']:
                routes = [f"{detail['from'][:2]}-{detail['to'][:2]}" 
                         for detail in mcf_data['cost_details']]
                costs = [detail['total_cost'] for detail in mcf_data['cost_details']]
                
                bars = ax3.bar(range(len(routes)), costs, 
                              color=plt.cm.Set3(np.linspace(0, 1, len(routes))))
                ax3.set_title('各路线运输成本', fontsize=14, fontweight='bold')
                ax3.set_ylabel('成本 (元)')
                ax3.set_xticks(range(len(routes)))
                ax3.set_xticklabels(routes, rotation=45)
                ax3.grid(True, alpha=0.3)
        
        # 4. 最短路径网络图
        if 'shortest_path' in self.results:
            ax4 = plt.subplot(2, 3, 4)
            sp_data = self.results['shortest_path']
            G = sp_data['graph']
            
            # 使用spring布局
            pos = nx.spring_layout(G, seed=42)
            
            # 绘制所有边
            nx.draw_networkx_edges(G, pos, edge_color='lightgray', ax=ax4)
            
            # 高亮最短路径
            shortest_path = sp_data['shortest_path']
            path_edges = [(shortest_path[i], shortest_path[i+1]) 
                         for i in range(len(shortest_path)-1)]
            nx.draw_networkx_edges(G, pos, path_edges, 
                                 edge_color='red', width=3, ax=ax4)
            
            # 绘制节点
            node_colors = ['red' if node in shortest_path else 'lightblue' 
                          for node in G.nodes()]
            nx.draw_networkx_nodes(G, pos, node_color=node_colors, 
                                 node_size=800, ax=ax4)
            nx.draw_networkx_labels(G, pos, font_size=10, ax=ax4)
            
            # 添加边权重标签
            edge_labels = nx.get_edge_attributes(G, 'weight')
            nx.draw_networkx_edge_labels(G, pos, edge_labels, font_size=8, ax=ax4)
            
            ax4.set_title(f'最短路径 (距离: {sp_data["shortest_distance"]} km)', 
                         fontsize=14, fontweight='bold')
            ax4.axis('off')
        
        # 5. 供需平衡分析（最小费用流）
        if 'min_cost_flow' in self.results:
            ax5 = plt.subplot(2, 3, 5)
            mcf_data = self.results['min_cost_flow']
            
            # 仓库供应量
            warehouses = list(mcf_data['supply'].keys())
            supply_values = list(mcf_data['supply'].values())
            
            # 客户需求量
            customers = list(mcf_data['demand'].keys())
            demand_values = list(mcf_data['demand'].values())
            
            x_pos = np.arange(max(len(warehouses), len(customers)))
            width = 0.35
            
            # 供应量柱状图
            ax5.bar(x_pos[:len(warehouses)] - width/2, supply_values, width, 
                   label='供应量', color='#87CEEB')
            
            # 需求量柱状图
            ax5.bar(x_pos[:len(customers)] + width/2, demand_values, width,
                   label='需求量', color='#FFB6C1')
            
            ax5.set_title('供需平衡分析', fontsize=14, fontweight='bold')
            ax5.set_ylabel('数量')
            ax5.set_xticks(x_pos)
            ax5.set_xticklabels([f'节点{i+1}' for i in range(len(x_pos))])
            ax5.legend()
            ax5.grid(True, alpha=0.3)
        
        # 6. 网络性能对比
        ax6 = plt.subplot(2, 3, 6)
        
        # 收集各种网络问题的关键指标
        metrics = []
        values = []
        
        if 'max_flow' in self.results:
            metrics.append('最大流量')
            values.append(self.results['max_flow']['max_flow_value'])
        
        if 'min_cost_flow' in self.results:
            metrics.append('最小成本')
            values.append(self.results['min_cost_flow']['min_cost'])
        
        if 'shortest_path' in self.results:
            metrics.append('最短距离')
            values.append(self.results['shortest_path']['shortest_distance'])
        
        if metrics:
            # 标准化数值以便比较
            normalized_values = [v/max(values) * 100 for v in values]
            
            bars = ax6.bar(range(len(metrics)), normalized_values, 
                          color=['#FF9999', '#66B2FF', '#99FF99'][:len(metrics)])
            ax6.set_title('网络优化指标对比', fontsize=14, fontweight='bold')
            ax6.set_ylabel('标准化值 (%)')
            ax6.set_xticks(range(len(metrics)))
            ax6.set_xticklabels(metrics, rotation=45)
            ax6.grid(True, alpha=0.3)
            
            # 添加实际数值标签
            for bar, value in zip(bars, values):
                ax6.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                        f'{value:.1f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig('network_flow_results.png', 
                   dpi=300, bbox_inches='tight')
        plt.show()
        
        print("✅ 网络流可视化图表已保存为 'network_flow_results.png'")
    
    def network_analysis(self):
        """网络结构分析"""
        if not self.graphs:
            print("⚠️ 请先运行求解方法")
            return
        
        print("\n🔍 网络结构分析")
        print("-" * 30)
        
        for problem_type, G in self.graphs.items():
            print(f"\n{problem_type.upper()} 网络:")
            print(f"  • 节点数: {G.number_of_nodes()}")
            print(f"  • 边数: {G.number_of_edges()}")
            print(f"  • 网络密度: {nx.density(G):.3f}")
            
            if nx.is_connected(G.to_undirected()):
                print(f"  • 网络连通性: 连通")
                if problem_type == 'shortest_path':
                    print(f"  • 网络直径: {nx.diameter(G):.1f}")
                    print(f"  • 平均路径长度: {nx.average_shortest_path_length(G):.1f}")
            else:
                print(f"  • 网络连通性: 非连通")
    
    def generate_report(self):
        """生成详细报告"""
        if not self.results:
            print("⚠️ 请先运行求解方法")
            return
        
        print("\n" + "="*50)
        print("📋 网络流优化报告")
        print("="*50)
        
        if 'max_flow' in self.results:
            max_flow_data = self.results['max_flow']
            print(f"\n💧 最大流问题:")
            print(f"  • 优化目标: 最大化网络流量")
            print(f"  • 最大流量: {max_flow_data['max_flow_value']} 单位/小时")
            print(f"  • 活跃边数: {len(max_flow_data['flow_details'])}")
            
            if max_flow_data['bottleneck_edges']:
                print(f"  • 瓶颈边数: {len(max_flow_data['bottleneck_edges'])}")
                print(f"  • 瓶颈位置: {', '.join([f"{e['from']}-{e['to']}" for e in max_flow_data['bottleneck_edges']])}")
        
        if 'min_cost_flow' in self.results:
            mcf_data = self.results['min_cost_flow']
            print(f"\n🚚 最小费用流问题:")
            print(f"  • 优化目标: 最小化运输成本")
            print(f"  • 最小成本: {mcf_data['min_cost']:.2f} 元")
            print(f"  • 总供应量: {sum(mcf_data['supply'].values())} 单位")
            print(f"  • 总需求量: {sum(mcf_data['demand'].values())} 单位")
            
            if mcf_data['cost_details']:
                avg_cost = mcf_data['min_cost'] / sum(detail['flow'] for detail in mcf_data['cost_details'])
                print(f"  • 平均运输成本: {avg_cost:.2f} 元/单位")
        
        if 'shortest_path' in self.results:
            sp_data = self.results['shortest_path']
            print(f"\n🗺️  最短路径问题:")
            print(f"  • 优化目标: 最小化路径距离")
            print(f"  • 最短距离: {sp_data['shortest_distance']} km")
            print(f"  • 路径长度: {len(sp_data['shortest_path'])} 个节点")
            print(f"  • 路径: {' → '.join(sp_data['shortest_path'])}")
        
        print(f"\n💡 优化建议:")
        
        if 'max_flow' in self.results and max_flow_data['bottleneck_edges']:
            print(f"  • 最大流: 考虑扩容瓶颈边以提高网络流量")
        
        if 'min_cost_flow' in self.results:
            print(f"  • 最小费用流: 优化高成本路线，寻找替代方案")
        
        if 'shortest_path' in self.results:
            print(f"  • 最短路径: 考虑建设新道路缩短关键路径")
        
        print("="*50)

def main():
    """主函数"""
    # 创建演示实例
    demo = NetworkFlowDemo()
    
    # 求解最大流问题
    max_flow_value, max_flow_dict = demo.solve_max_flow_problem()
    
    # 求解最小费用流问题
    min_cost, flow_solution = demo.solve_min_cost_flow_problem()
    
    # 求解最短路径问题
    shortest_path, shortest_distance = demo.solve_shortest_path_problem()
    
    # 生成可视化
    demo.visualize_results()
    
    # 网络分析
    demo.network_analysis()
    
    # 生成报告
    demo.generate_report()
    
    print(f"\n🎉 网络流优化演示完成！")
    print(f"最大流量: {max_flow_value} 单位/小时")
    print(f"最小运输成本: {min_cost:.2f} 元")
    print(f"最短路径距离: {shortest_distance} km")

if __name__ == "__main__":
    main()