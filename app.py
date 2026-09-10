import os

import matplotlib.font_manager as fm
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import streamlit as st
from PIL import Image

from src.classifier import WasteSteelClassifier
from src.features import extract_heuristic_features
from src.visualization import map_pc_to_pixel


def load_demo_font():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    font_path = os.path.join(base_dir, "fonts", "SourceHanSansSC-Regular.otf.ttf")
    if os.path.exists(font_path):
        fm.fontManager.addfont(font_path)
        plt.rcParams['font.family'] = fm.FontProperties(fname=font_path).get_name()
        plt.rcParams['axes.unicode_minus'] = False
        return fm.FontProperties(fname=font_path)
    return None

my_font = load_demo_font()

# 初始化分类器
classifier = WasteSteelClassifier()

# 设置页面配置
st.set_page_config(
    page_title="影簇智检 - 启发式演示终端",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed"  # 默认折叠侧边栏，适合移动设备
)
# --- 替换 1：侧边栏全中文逻辑 ---
with st.sidebar:
    st.header("⚙️ 终端控制台")
    # 专家模式开关完全中文化
    expert_mode = st.toggle("开启专家模式", value=False, help="开启后展示启发式代理值与合成 PCA 投影坐标")
    st.divider()
    st.info("💡 当前为启发式可视化原型，不是经过工业数据验证的自动判级系统。")
# 顶部大标题
st.markdown("""
    <style>
        .main-title {
            font-size: 2.8rem;
            font-weight: bold;
            color: #1e3a5f;
            text-align: center;
            margin-top: 30px;
            margin-bottom: 40px;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
        }
        .btn-large {
            font-size: 1.1rem;
            padding: 10px 20px;
        }
        /* 工业风格主色调 */
        .stApp {
            background-color: #f5f7fa;
            color: #333;
        }
        /* 卡片样式 */
        .result-card {
            background-color: white;
            border-radius: 12px;
            padding: 20px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            margin-bottom: 20px;
            text-align: center;
        }
        .result-card h3 {
            margin-top: 0;
            color: #1e3a5f;
        }
        .result-value {
            font-size: 1.8rem;
            font-weight: bold;
            margin: 10px 0;
        }
        .pass {
            border-left: 6px solid #28a745;
        }
        .review {
            border-left: 6px solid #ffc107;
        }
        .error {
            border-left: 6px solid #dc3545;
        }
        /* 时间线样式 */
        .timeline {
            position: relative;
            max-width: 1200px;
            margin: 0 auto;
        }
        .timeline::after {
            content: '';
            position: absolute;
            width: 6px;
            background-color: #1e3a5f;
            top: 0;
            bottom: 0;
            left: 50%;
            margin-left: -3px;
        }
        .timeline-item {
            padding: 10px 40px;
            position: relative;
            background-color: inherit;
            width: 50%;
        }
        .timeline-item::after {
            content: '';
            position: absolute;
            width: 25px;
            height: 25px;
            right: -12px;
            background-color: white;
            border: 4px solid #1e3a5f;
            top: 15px;
            border-radius: 50%;
            z-index: 1;
        }
        .left {
            left: 0;
        }
        .right {
            left: 50%;
        }
        .left::after {
            right: -12px;
        }
        .right::after {
            left: -12px;
        }
        .timeline-content {
            padding: 20px 30px;
            background-color: white;
            position: relative;
            border-radius: 6px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        @media screen and (max-width: 768px) {
            .timeline::after {
                left: 31px;
            }
            .timeline-item {
                width: 100%;
                padding-left: 70px;
                padding-right: 25px;
            }
            .timeline-item::after {
                left: 18px;
            }
            .left::after, .right::after {
                left: 18px;
            }
            .right {
                left: 0%;
            }
        }
    </style>
    <h1 class="main-title">🚀 影簇智检 - 废钢数字化判级终端</h1>
""", unsafe_allow_html=True)

# 中间文件上传区域
st.markdown("<h3 style='text-align: center; margin-bottom: 20px;'>上传图片体验启发式匹配</h3>", unsafe_allow_html=True)

col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    # --- 替换 2：上传器中文显示 ---
    uploaded_file = st.file_uploader("请上传或拖拽废钢现场照片", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        # 显示上传的图片
        image = Image.open(uploaded_file)
        st.image(image, caption="上传的废钢照片", use_container_width=True)
        
        if st.button("运行演示匹配", key="classify_btn", help="计算启发式代理特征并匹配固定演示中心"):
            # 提取特征向量
            feature_vector = extract_heuristic_features(image)
            
            # 分类
            predicted_class, matching_score, pc_coords = classifier.classify(feature_vector)
            
            # 保存当前特征向量和主成分坐标用于后续显示
            st.session_state['current_features'] = feature_vector
            st.session_state['predicted_class'] = predicted_class
            st.session_state['pc_coords'] = pc_coords
            st.session_state['matching_score'] = matching_score
            st.session_state['class_name'] = classifier.class_names[predicted_class]

            # --- 优化后的结果展示区 ---
            st.markdown("---")
            st.subheader("🧭 演示匹配结果")

            # 改用HTML+CSS卡片布局展示结果
            if matching_score > 75:
                # 绿色通过面板
                st.markdown(f"""
                    <div class="result-card pass">
                        <h3>最近中心匹配结果</h3>
                        <div class="result-value">{classifier.class_names[predicted_class]}</div>
                        <div>启发式相对匹配分数：{matching_score}%</div>
                        <p style="color: green; margin-top: 10px;">该分数不是校准概率，也不代表样本符合工业质量标准。</p>
                    </div>
                """, unsafe_allow_html=True)
            elif matching_score > 65:
                # 黄色警告面板
                st.markdown(f"""
                    <div class="result-card review">
                        <h3>⚠️ 边界匹配结果</h3>
                        <div class="result-value">{classifier.class_names[predicted_class]}</div>
                        <div>启发式相对匹配分数：{matching_score}%</div>
                        <p style="color: orange; margin-top: 10px;">该演示结果只适合流程展示，不可代替人工检验。</p>
                    </div>
                """, unsafe_allow_html=True)
                
                # 反馈机制：人工复核最终等级
                st.markdown("""
                    <div style="margin-top: 20px; padding: 15px; background-color: #fff3cd; border-radius: 8px;">
                        <h4 style="margin-top: 0; color: #856404;">人工复核反馈</h4>
                    </div>
                """, unsafe_allow_html=True)
                
                final_level = st.radio(
                    "人工复核最终等级为何？",
                    options=["I类（演示中心）", "II类（演示中心）", "III类（演示中心）"],
                    key="final_level"
                )
                
                if st.button("提交复核结果", key="submit_feedback"):
                    # 保存反馈数据到本地
                    import csv
                    import datetime
                    
                    feedback_data = {
                        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "original_class": classifier.class_names[predicted_class],
                        "corrected_class": final_level,
                        "matching_score": matching_score,
                        "thickness_proxy": feature_vector[0],
                        "corrosion_proxy": feature_vector[1],
                        "purity_proxy": feature_vector[2]
                    }
                    
                    # 写入CSV文件
                    with open('feedback.csv', 'a', newline='', encoding='utf-8') as f:
                        writer = csv.DictWriter(f, fieldnames=feedback_data.keys())
                        # 如果文件为空，写入表头
                        if f.tell() == 0:
                            writer.writeheader()
                        writer.writerow(feedback_data)
                    
                    st.success("✅ 复核结果已保存到本地 feedback.csv；当前原型不会自动训练或更新模型。")
            else:
                # 红色错误面板
                st.markdown(f"""
                    <div class="result-card error">
                        <h3>低匹配度</h3>
                        <div class="result-value">{classifier.class_names[predicted_class]}</div>
                        <div>启发式相对匹配分数：{matching_score}%</div>
                        <p style="color: red; margin-top: 10px;">样本与固定演示中心的距离较远；不能据此作质量判断。</p>
                    </div>
                """, unsafe_allow_html=True)

            # 保留详细数据，放在折叠栏里，显得专业又不乱
            with st.expander("🔍 专家视图：查看启发式代理特征向量", expanded=expert_mode):
                st.write(
                    f"厚度代理值: {feature_vector[0]:.2f} | 锈蚀代理值: {feature_vector[1]:.2f} | 纯度代理值: {feature_vector[2]:.2f}")
                st.write(f"合成 PCA 投影坐标: PC1={pc_coords[0]:.2f}, PC2={pc_coords[1]:.2f}")
                
                # 增加判定过程的时间线展示
                st.markdown("""
                    <h4 style="margin-top: 20px; color: #1e3a5f;">判定过程时间线</h4>
                    <div class="timeline">
                        <div class="timeline-item left">
                            <div class="timeline-content">
                                <h5>图像采集完成</h5>
                                <p>成功获取废钢现场照片</p>
                            </div>
                        </div>
                        <div class="timeline-item right">
                            <div class="timeline-content">
                                <h5>代理特征计算</h5>
                                <p>将亮度和像素标准差映射为演示值</p>
                            </div>
                        </div>
                        <div class="timeline-item left">
                            <div class="timeline-content">
                                <h5>加权欧氏距离核算</h5>
                                <p>计算代理向量与各固定演示中心的距离</p>
                            </div>
                        </div>
                        <div class="timeline-item right">
                            <div class="timeline-content">
                                <h5>结果输出</h5>
                                <p>生成最终判定结论</p>
                            </div>
                        </div>
                    </div>
                """, unsafe_allow_html=True)

# 视觉增强：渲染聚类分布图并添加红色十字光标
st.markdown("<h3 style='text-align: center; margin-top: 40px; margin-bottom: 20px;'>影簇矩界_最终聚类分布图</h3>", unsafe_allow_html=True)

col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    try:
        # 加载并显示聚类分布图
        img = mpimg.imread('影簇矩界_最终聚类分布图.png')
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.imshow(img)
        
        # 检查是否有当前特征向量和主成分坐标
        if 'current_features' in st.session_state and 'pc_coords' in st.session_state:
            # 获取主成分坐标
            pc_coords = st.session_state['pc_coords']
            # 假设图像大小为 (width, height)
            img_height, img_width, _ = img.shape
            
            # 使用 classifier 的方法将主成分坐标映射到像素位置
            x, y = map_pc_to_pixel(pc_coords, img_width, img_height)
            
            # 添加红色十字光标
            cross_size = 20
            ax.plot(x, y, 'r+', markersize=cross_size, markeredgewidth=2)
            
            # 添加标签
            ax.text(x + 25, y - 25, f'当前样本: {st.session_state["predicted_class"]}类', 
                    bbox=dict(facecolor='white', alpha=0.7), fontsize=10)
            # 添加主成分坐标信息
            ax.text(x + 25, y + 10, f'PC1: {pc_coords[0]:.2f}, PC2: {pc_coords[1]:.2f}', 
                    bbox=dict(facecolor='white', alpha=0.7), fontsize=8)
        
        # 隐藏坐标轴
        ax.axis('off')
        
        # 显示图像
        st.pyplot(fig)
        
    except FileNotFoundError:
        st.error("未找到 '影簇矩界_最终聚类分布图.png' 文件，请确保该文件存在于当前目录。")

# 移动适配优化
st.markdown("""
    <style>
        /* 隐藏右上角的 Deploy 按钮和三点菜单 */
        .stDeployButton {
            display: none !important;
        }
        
        .stApp > header {
            display: none !important;
        }
        
        /* 确保按钮足够大，适合移动设备 */
        .stButton > button {
            font-size: 1.1rem;
            padding: 10px 20px;
            width: 100%;
        }
        
        /* 确保文件上传器在移动设备上显示正常 */
        .stFileUploader > label {
            font-size: 1rem;
        }
        
        /* 确保侧边栏可以隐藏 */
        @media (max-width: 768px) {
            .main-title {
                font-size: 2rem !important;
            }
            
            .stImage {
                margin-bottom: 15px;
            }
        }
    </style>
""", unsafe_allow_html=True)

# 底部信息
st.markdown("""
    <div style='text-align: center; margin-top: 40px; color: #666; font-size: 0.9rem;'>
        <p>影簇智检 - 废钢图像启发式演示原型</p>
        <p>基于固定演示中心、加权欧氏距离与合成 PCA 投影</p>
    </div>
""", unsafe_allow_html=True)
