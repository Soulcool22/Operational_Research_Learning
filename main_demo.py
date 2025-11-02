"""
运筹学优化演示主程序
Operations Research Optimization Main Demo

整合所有优化问题演示：
- 线性规划 (Linear Programming)
- 整数规划 (Integer Programming) 
- 运输问题 (Transportation Problem)
- 网络流优化 (Network Flow Optimization)

作者: AI Assistant
日期: 2024
"""

import sys
import time
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# 导入各个演示模块
try:
    from linear_programming_demo import LinearProgrammingDemo
    from integer_programming_demo import IntegerProgrammingDemo
    from transportation_problem_demo import TransportationProblemDemo
    from network_flow_demo import NetworkFlowDemo
except ImportError as e:
    print(f"❌ 导入模块失败: {e}")
    print("请确保所有演示文件都在同一目录下")
    sys.exit(1)

# 使用自定义字体配置支持中文
from font_config import setup_chinese_font
setup_chinese_font()

class OperationsResearchMainDemo:
    """运筹学优化主演示类"""
    
    def __init__(self):
        self.demos = {}
        self.results_summary = {}
        print("=" * 60)
        print("🎯 运筹学优化演示系统")
        print("Operations Research Optimization Demo System")
        print("=" * 60)
        print(f"启动时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print()
    
    def show_menu(self):
        """显示主菜单"""
        print("\n" + "="*50)
        print("📋 请选择要运行的演示:")
        print("="*50)
        print("1. 🔢 线性规划演示 (Linear Programming)")
        print("2. 🎲 整数规划演示 (Integer Programming)")
        print("3. 🚛 运输问题演示 (Transportation Problem)")
        print("4. 🌐 网络流优化演示 (Network Flow)")
        print("5. 🚀 运行所有演示 (Run All Demos)")
        print("6. 📊 查看结果汇总 (View Summary)")
        print("7. 🔄 重新运行特定演示 (Re-run Specific Demo)")
        print("0. 🚪 退出程序 (Exit)")
        print("="*50)
    
    def run_linear_programming_demo(self):
        """运行线性规划演示"""
        print("\n🔢 启动线性规划演示...")
        start_time = time.time()
        
        try:
            demo = LinearProgrammingDemo()
            
            # 运行生产计划问题
            max_profit, solution = demo.solve_production_planning()
            
            # 运行投资组合问题
            portfolio_return, portfolio_solution = demo.solve_portfolio_optimization()
            
            # 生成可视化
            demo.visualize_results()
            
            # 敏感性分析
            demo.sensitivity_analysis()
            
            # 生成报告
            demo.generate_report()
            
            execution_time = time.time() - start_time
            
            # 保存结果
            self.demos['linear_programming'] = demo
            self.results_summary['linear_programming'] = {
                'max_profit': max_profit,
                'portfolio_return': portfolio_return,
                'execution_time': execution_time,
                'status': 'completed'
            }
            
            print(f"✅ 线性规划演示完成 (耗时: {execution_time:.2f}秒)")
            return True
            
        except Exception as e:
            print(f"❌ 线性规划演示失败: {e}")
            self.results_summary['linear_programming'] = {
                'status': 'failed',
                'error': str(e)
            }
            return False
    
    def run_integer_programming_demo(self):
        """运行整数规划演示"""
        print("\n🎲 启动整数规划演示...")
        start_time = time.time()
        
        try:
            demo = IntegerProgrammingDemo()
            
            # 运行设施选址问题
            min_cost, facility_solution = demo.solve_facility_location()
            
            # 运行背包问题
            max_value, knapsack_solution = demo.solve_knapsack_problem()
            
            # 生成可视化
            demo.visualize_results()
            
            # 场景分析
            demo.scenario_analysis()
            
            # 生成报告
            demo.generate_report()
            
            execution_time = time.time() - start_time
            
            # 保存结果
            self.demos['integer_programming'] = demo
            self.results_summary['integer_programming'] = {
                'min_facility_cost': min_cost,
                'max_knapsack_value': max_value,
                'execution_time': execution_time,
                'status': 'completed'
            }
            
            print(f"✅ 整数规划演示完成 (耗时: {execution_time:.2f}秒)")
            return True
            
        except Exception as e:
            print(f"❌ 整数规划演示失败: {e}")
            self.results_summary['integer_programming'] = {
                'status': 'failed',
                'error': str(e)
            }
            return False
    
    def run_transportation_demo(self):
        """运行运输问题演示"""
        print("\n🚛 启动运输问题演示...")
        start_time = time.time()
        
        try:
            demo = TransportationProblemDemo()
            
            # 运行基础运输问题
            solution_matrix, min_cost = demo.solve_basic_transportation()
            
            # 运行多产品运输问题
            multi_min_cost = demo.solve_multi_product_transportation()
            
            # 生成可视化
            demo.visualize_results()
            
            # 敏感性分析
            demo.cost_sensitivity_analysis()
            
            # 生成报告
            demo.generate_report()
            
            execution_time = time.time() - start_time
            
            # 保存结果
            self.demos['transportation'] = demo
            self.results_summary['transportation'] = {
                'basic_min_cost': min_cost,
                'multi_min_cost': multi_min_cost,
                'execution_time': execution_time,
                'status': 'completed'
            }
            
            print(f"✅ 运输问题演示完成 (耗时: {execution_time:.2f}秒)")
            return True
            
        except Exception as e:
            print(f"❌ 运输问题演示失败: {e}")
            self.results_summary['transportation'] = {
                'status': 'failed',
                'error': str(e)
            }
            return False
    
    def run_network_flow_demo(self):
        """运行网络流优化演示"""
        print("\n🌐 启动网络流优化演示...")
        start_time = time.time()
        
        try:
            demo = NetworkFlowDemo()
            
            # 运行最大流问题
            max_flow_value, max_flow_dict = demo.solve_max_flow_problem()
            
            # 运行最小费用流问题
            min_cost, flow_solution = demo.solve_min_cost_flow_problem()
            
            # 运行最短路径问题
            shortest_path, shortest_distance = demo.solve_shortest_path_problem()
            
            # 生成可视化
            demo.visualize_results()
            
            # 网络分析
            demo.network_analysis()
            
            # 生成报告
            demo.generate_report()
            
            execution_time = time.time() - start_time
            
            # 保存结果
            self.demos['network_flow'] = demo
            self.results_summary['network_flow'] = {
                'max_flow_value': max_flow_value,
                'min_cost_flow': min_cost,
                'shortest_distance': shortest_distance,
                'execution_time': execution_time,
                'status': 'completed'
            }
            
            print(f"✅ 网络流优化演示完成 (耗时: {execution_time:.2f}秒)")
            return True
            
        except Exception as e:
            print(f"❌ 网络流优化演示失败: {e}")
            self.results_summary['network_flow'] = {
                'status': 'failed',
                'error': str(e)
            }
            return False
    
    def run_all_demos(self):
        """运行所有演示"""
        print("\n🚀 启动所有演示...")
        print("这可能需要几分钟时间，请耐心等待...")
        
        total_start_time = time.time()
        success_count = 0
        total_demos = 4
        
        # 运行所有演示
        demos_to_run = [
            ("线性规划", self.run_linear_programming_demo),
            ("整数规划", self.run_integer_programming_demo),
            ("运输问题", self.run_transportation_demo),
            ("网络流优化", self.run_network_flow_demo)
        ]
        
        for i, (name, demo_func) in enumerate(demos_to_run, 1):
            print(f"\n📍 进度: {i}/{total_demos} - 运行{name}演示")
            if demo_func():
                success_count += 1
            
            # 显示进度
            progress = i / total_demos * 100
            print(f"⏳ 总体进度: {progress:.1f}% ({i}/{total_demos})")
        
        total_execution_time = time.time() - total_start_time
        
        print(f"\n🎉 所有演示完成!")
        print(f"✅ 成功: {success_count}/{total_demos}")
        print(f"⏱️  总耗时: {total_execution_time:.2f}秒")
        
        if success_count < total_demos:
            print(f"⚠️  {total_demos - success_count} 个演示失败，请检查错误信息")
    
    def view_results_summary(self):
        """查看结果汇总"""
        if not self.results_summary:
            print("⚠️ 还没有运行任何演示，请先选择运行演示")
            return
        
        print("\n" + "="*60)
        print("📊 运筹学优化结果汇总")
        print("="*60)
        
        total_execution_time = 0
        successful_demos = 0
        
        for demo_name, results in self.results_summary.items():
            print(f"\n🔹 {demo_name.upper().replace('_', ' ')}:")
            
            if results['status'] == 'completed':
                successful_demos += 1
                execution_time = results.get('execution_time', 0)
                total_execution_time += execution_time
                
                print(f"  ✅ 状态: 成功完成")
                print(f"  ⏱️  执行时间: {execution_time:.2f}秒")
                
                # 显示关键结果
                if demo_name == 'linear_programming':
                    print(f"  💰 最大利润: {results.get('max_profit', 'N/A')}")
                    print(f"  📈 投资组合收益: {results.get('portfolio_return', 'N/A'):.4f}")
                
                elif demo_name == 'integer_programming':
                    print(f"  🏭 最小设施成本: {results.get('min_facility_cost', 'N/A')}")
                    print(f"  🎒 最大背包价值: {results.get('max_knapsack_value', 'N/A')}")
                
                elif demo_name == 'transportation':
                    print(f"  🚛 基础运输最小成本: {results.get('basic_min_cost', 'N/A'):.2f}")
                    print(f"  📦 多产品运输最小成本: {results.get('multi_min_cost', 'N/A'):.2f}")
                
                elif demo_name == 'network_flow':
                    print(f"  💧 最大流量: {results.get('max_flow_value', 'N/A')}")
                    print(f"  💰 最小费用流成本: {results.get('min_cost_flow', 'N/A'):.2f}")
                    print(f"  🗺️  最短路径距离: {results.get('shortest_distance', 'N/A')}")
            
            else:
                print(f"  ❌ 状态: 失败")
                print(f"  🐛 错误: {results.get('error', '未知错误')}")
        
        print(f"\n📈 总体统计:")
        print(f"  • 成功演示: {successful_demos}/{len(self.results_summary)}")
        print(f"  • 总执行时间: {total_execution_time:.2f}秒")
        print(f"  • 平均执行时间: {total_execution_time/len(self.results_summary):.2f}秒")
        
        success_rate = successful_demos / len(self.results_summary) * 100
        print(f"  • 成功率: {success_rate:.1f}%")
        
        print("="*60)
    
    def rerun_specific_demo(self):
        """重新运行特定演示"""
        if not self.results_summary:
            print("⚠️ 还没有运行任何演示")
            return
        
        print("\n🔄 选择要重新运行的演示:")
        demo_options = {
            '1': ('linear_programming', '线性规划', self.run_linear_programming_demo),
            '2': ('integer_programming', '整数规划', self.run_integer_programming_demo),
            '3': ('transportation', '运输问题', self.run_transportation_demo),
            '4': ('network_flow', '网络流优化', self.run_network_flow_demo)
        }
        
        for key, (_, name, _) in demo_options.items():
            status = self.results_summary.get(demo_options[key][0], {}).get('status', '未运行')
            print(f"{key}. {name} (状态: {status})")
        
        choice = input("\n请输入选择 (1-4): ").strip()
        
        if choice in demo_options:
            demo_key, demo_name, demo_func = demo_options[choice]
            print(f"\n🔄 重新运行{demo_name}演示...")
            demo_func()
        else:
            print("❌ 无效选择")
    
    def run(self):
        """运行主程序"""
        while True:
            self.show_menu()
            
            try:
                choice = input("\n请输入您的选择 (0-7): ").strip()
                
                if choice == '0':
                    print("\n👋 感谢使用运筹学优化演示系统!")
                    print(f"结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                    break
                
                elif choice == '1':
                    self.run_linear_programming_demo()
                
                elif choice == '2':
                    self.run_integer_programming_demo()
                
                elif choice == '3':
                    self.run_transportation_demo()
                
                elif choice == '4':
                    self.run_network_flow_demo()
                
                elif choice == '5':
                    self.run_all_demos()
                
                elif choice == '6':
                    self.view_results_summary()
                
                elif choice == '7':
                    self.rerun_specific_demo()
                
                else:
                    print("❌ 无效选择，请输入 0-7 之间的数字")
                
                # 等待用户确认继续
                if choice != '0':
                    input("\n按回车键继续...")
                
            except KeyboardInterrupt:
                print("\n\n👋 程序被用户中断，再见!")
                break
            except Exception as e:
                print(f"\n❌ 程序出现错误: {e}")
                input("按回车键继续...")

def main():
    """主函数"""
    try:
        # 创建主演示实例
        main_demo = OperationsResearchMainDemo()
        
        # 运行主程序
        main_demo.run()
        
    except Exception as e:
        print(f"❌ 程序启动失败: {e}")
        print("请检查所有依赖模块是否正确安装")

if __name__ == "__main__":
    main()