"""
中文字体配置模块
解决matplotlib中文显示问题
"""
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import warnings

def setup_chinese_font():
    """
    设置matplotlib中文字体支持
    按优先级尝试不同的中文字体
    """
    # 中文字体优先级列表
    chinese_fonts = [
        'Microsoft YaHei',  # 微软雅黑 - Windows常用
        'SimHei',           # 黑体 - Windows系统字体
        'Noto Sans SC',     # Google Noto字体
        'STHeiti',          # 华文黑体 - macOS
        'WenQuanYi Micro Hei',  # 文泉驿微米黑 - Linux
        'DejaVu Sans'       # 备用字体
    ]
    
    # 获取系统可用字体
    available_fonts = [f.name for f in fm.fontManager.ttflist]
    
    # 找到第一个可用的中文字体
    selected_font = None
    for font in chinese_fonts:
        if font in available_fonts:
            selected_font = font
            break
    
    if selected_font:
        # 强制设置字体配置
        plt.rcParams['font.sans-serif'] = [selected_font]
        plt.rcParams['font.family'] = 'sans-serif'
        
        # 额外设置确保所有文本元素都使用中文字体
        plt.rcParams['font.serif'] = [selected_font]
        plt.rcParams['font.monospace'] = [selected_font]
        
        print(f"已设置中文字体: {selected_font}")
    else:
        warnings.warn("未找到合适的中文字体，可能无法正确显示中文")
    
    # 解决负号显示问题
    plt.rcParams['axes.unicode_minus'] = False
    
    # 清除字体缓存（新版本matplotlib的方法）
    try:
        fm._rebuild()
    except AttributeError:
        # 新版本matplotlib使用不同的方法
        pass
    
    # 返回选中的字体名称
    return selected_font