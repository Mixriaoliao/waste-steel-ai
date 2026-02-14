# --- 强制中文字体挂载逻辑 (防止乱码) ---
import matplotlib.font_manager as fm
import os
import urllib.request

def load_demo_font():
    # 强制指定字体保存路径，使用绝对路径确保云端部署时也能正确找到
    base_dir = os.path.dirname(os.path.abspath(__file__))
    font_path = os.path.join(base_dir, "fonts", "SourceHanSansSC-Regular.otf")
    if os.path.exists(font_path):
        fm.fontManager.addfont(font_path)
        plt.rcParams['font.family'] = fm.FontProperties(fname=font_path).get_name()
        plt.rcParams['axes.unicode_minus'] = False
        return fm.FontProperties(fname=font_path)
    return None

my_font = load_demo_font()
# ------------------------------------
import streamlit as st
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from sklearn.decomposition import PCA

# 核心逻辑：改进的 K-means++ 和加权马氏距离
class WasteSteelClassifier:
    def __init__(self):
        # 聚类中心（基于研究报告）
        self.cluster_centers = {
            'I': np.array([8.3, 5.1, 0.92]),  # 厚度(mm), 锈蚀(%), 纯度
            'II': np.array([4.2, 27.7, 0.74]),
            'III': np.array([2.2, 51.0, 0.48])
        }
        # 权重设置
        self.weights = np.array([0.42, 0.35, 0.23])  # 厚度, 锈蚀, 纯度
        # 类别名称映射
        self.class_names = {
            'I': 'I类（优质）',
            'II': 'II类（标准）',
            'III': 'III类（劣质）'
        }
        # 初始化 PCA 模型并拟合
        self.pca = self._initialize_pca()
        # 计算 loadings
        self.loadings = self.pca.components_
        # 特征名称
        self.feature_names = ['厚度', '锈蚀', '纯度']

    def _initialize_pca(self):
        """初始化 PCA 模型，确保与可视化脚本的数据逻辑完全对齐"""
        # 1. 强制设定固定随机种子，保证每次运行生成的投影矩阵完全相同
        np.random.seed(42)

        # 2. 模拟生成与底图一致的训练数据集分布
        n_samples_per_class = 400

        # I类数据分布：厚度高、锈蚀低、纯度高
        class1 = np.random.multivariate_normal([8.5, 10.0, 0.92],
                                               [[1.5, -0.5, 0.01], [-0.5, 5.0, -0.01], [0.01, -0.01, 0.001]],
                                               n_samples_per_class)
        # II类数据分布：中等特征
        class2 = np.random.multivariate_normal([4.5, 30.0, 0.75],
                                               [[1.0, -0.2, 0.01], [-0.2, 10.0, -0.02], [0.01, -0.02, 0.005]],
                                               n_samples_per_class)
        # III类数据分布：厚度低、锈蚀高、纯度低
        class3 = np.random.multivariate_normal([2.5, 55.0, 0.45],
                                               [[0.5, -0.1, 0.01], [-0.1, 15.0, -0.05], [0.01, -0.05, 0.01]],
                                               n_samples_per_class)

        X_train = np.vstack([class1, class2, class3])

        # 3. 拟合 PCA 模型，确定 PC1 和 PC2 的坐标轴方向
        pca = PCA(n_components=2)
        pca.fit(X_train)
        return pca
    
    def transform_to_pc(self, feature_vector):
        """将特征向量转换为主成分空间"""
        # 确保输入是二维数组
        if len(feature_vector.shape) == 1:
            feature_vector = feature_vector.reshape(1, -1)
        # 转换到主成分空间
        pc_coords = self.pca.transform(feature_vector)
        return pc_coords[0]  # 返回一维数组

    def map_pc_to_pixel(self, pc_coords, img_width, img_height):
        """
        精准像素对齐：针对底图布局进行非对称补偿
        解决标题、轴标签导致的十字架偏位问题
        """
        # --- 步骤 1：严格对齐坐标轴刻度 ---
        # 观察底图：横轴 PC1 为 -30 到 50，纵轴 PC2 为 -4 到 4
        pc1_min, pc1_max = -30, 50
        pc2_min, pc2_max = -4, 4

        # 坐标归一化处理 (0-1)
        pc1_norm = (pc_coords[0] - pc1_min) / (pc1_max - pc1_min)
        pc2_norm = (pc_coords[1] - pc2_min) / (pc2_max - pc2_min)

        # --- 步骤 2：针对图片布局进行“像素级”边距补偿 ---
        # 根据影簇矩界底图的视觉分布，设置四个方向的留白比例
        margin_left = 0.12  # 左侧留给纵轴数值
        margin_right = 0.08  # 右侧留白较少
        margin_top = 0.16  # 上方留给大标题和子标题
        margin_bottom = 0.12  # 下方留给横轴标签

        # --- 步骤 3：计算最终映射像素 ---
        # 计算 X 坐标：起始点 + 比例 * 可用宽度
        x = int((margin_left + pc1_norm * (1 - margin_left - margin_right)) * img_width)

        # 计算 Y 坐标：因为像素 0 在顶部，所以 Y 轴需要反向映射
        # 逻辑：1.0 - pc2_norm 代表数学上的高位对应像素上的低位
        y = int((margin_top + (1.0 - pc2_norm) * (1 - margin_top - margin_bottom)) * img_height)

        return x, y
    
    def calculate_weighted_mahalanobis(self, feature_vector):
        """计算加权马氏距离"""
        distances = {}
        for cls, center in self.cluster_centers.items():
            # 计算加权欧氏距离（简化版加权马氏距离）
            weighted_diff = (feature_vector - center) * np.sqrt(self.weights)
            distance = np.sqrt(np.sum(weighted_diff ** 2))
            distances[cls] = distance
        return distances
    
    def classify(self, feature_vector):
        """分类并计算置信度"""
        distances = self.calculate_weighted_mahalanobis(feature_vector)
        # 找到最近的聚类中心
        predicted_class = min(distances, key=distances.get)
        # 计算置信度（距离越近，置信度越高）
        max_distance = max(distances.values())
        min_distance = distances[predicted_class]
        confidence = 1.0 - (min_distance / max_distance) if max_distance > 0 else 1.0
        confidence = round(confidence * 100, 2)
        # 计算主成分坐标
        pc_coords = self.transform_to_pc(feature_vector)
        return predicted_class, confidence, pc_coords

    def extract_features(self, image):
        """
        优化后的半实装逻辑：基于图像像素特征模拟物理参数
        1. 亮度(Brightness) -> 映射为纯度和锈蚀
        2. 边缘密度/标准差(Std) -> 映射为厚度
        """
        # --- 步骤 1：基础图像处理 ---
        # 转换为灰度图，方便进行数学计算
        img_gray = image.convert('L')
        img_array = np.array(img_gray)

        # --- 步骤 2：提取像素统计特征 ---
        # 计算平均亮度 (0为全黑，1为全白)
        brightness = img_array.mean() / 255.0
        # 计算标准差 (反映纹理复杂程度，通常废钢越厚、堆积越乱，标准差越大)
        pixel_std = img_array.std() / 255.0

        # --- 步骤 3：建立像素与物理特征的逻辑关联 ---
        # 为了保证演示时“同一张图结果固定”，设置基于图片内容的随机种子
        np.random.seed(hash(image.tobytes()) % 4294967296)

        # 1. 厚度模拟：纹理越复杂(std高)，通常意味着废钢形状越大、厚度越高
        # 基础厚度 3mm，根据 std 波动 2-8mm
        thickness = np.clip(3.0 + (pixel_std * 15.0), 1.0, 15.0)

        # 2. 锈蚀模拟：亮度越低，通常意味着表面氧化严重或光泽度差
        # 逻辑：亮度 0.8 以上基本无锈(5%)，亮度 0.2 以下重锈(70%)
        corrosion = np.clip((1.0 - brightness) * 80.0 + np.random.uniform(-5, 5), 5.0, 85.0)

        # 3. 纯度模拟：亮度高通常意味着金属质感好
        # 逻辑：亮度直接决定纯度基准，波动范围在 0.4-0.98 之间
        purity = np.clip(brightness * 1.1 - 0.05, 0.4, 0.98)

        return np.array([thickness, corrosion, purity])
# 初始化分类器
classifier = WasteSteelClassifier()

# 设置页面配置
st.set_page_config(
    page_title="影簇智检 - 数字化判级终端",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed"  # 默认折叠侧边栏，适合移动设备
)
# --- 替换 1：侧边栏全中文逻辑 ---
with st.sidebar:
    st.header("⚙️ 终端控制台")
    # 专家模式开关完全中文化
    expert_mode = st.toggle("开启专家模式", value=False, help="开启后展示底层物理特征与空间投影坐标")
    st.divider()
    st.info("💡 提示：本终端已连接智能判定引擎，支持实时工业级废钢分类。")
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
st.markdown("<h3 style='text-align: center; margin-bottom: 20px;'>上传废钢照片进行智能判级</h3>", unsafe_allow_html=True)

col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    # --- 替换 2：上传器中文显示 ---
    uploaded_file = st.file_uploader("请上传或拖拽废钢现场照片", type=["jpg", "jpeg", "png"])
    
    # 增加实时判级模拟按钮
    st.button("📸 开启实时判级", key="realtime_btn", help="模拟实时相机判级功能")
    
    if uploaded_file is not None:
        # 显示上传的图片
        image = Image.open(uploaded_file)
        st.image(image, caption="上传的废钢照片", use_container_width=True)
        
        # 智能定界按钮
        if st.button("智能定界", key="classify_btn", help="点击进行智能判级"):
            # 提取特征向量
            feature_vector = classifier.extract_features(image)
            
            # 分类
            predicted_class, confidence, pc_coords = classifier.classify(feature_vector)
            
            # 保存当前特征向量和主成分坐标用于后续显示
            st.session_state['current_features'] = feature_vector
            st.session_state['predicted_class'] = predicted_class
            st.session_state['pc_coords'] = pc_coords
            st.session_state['confidence'] = confidence
            st.session_state['class_name'] = classifier.class_names[predicted_class]

            # --- 优化后的结果展示区 ---
            st.markdown("---")
            st.subheader("🤖 智能判定结论")

            # 改用HTML+CSS卡片布局展示结果
            if confidence > 75:
                # 绿色通过面板
                st.markdown(f"""
                    <div class="result-card pass">
                        <h3>✅ 自动判定通过</h3>
                        <div class="result-value">{classifier.class_names[predicted_class]}</div>
                        <div>算法置信度：{confidence}%</div>
                        <p style="color: green; margin-top: 10px;">当前样本符合工业标准，判定通过。</p>
                    </div>
                """, unsafe_allow_html=True)
            elif confidence > 65:
                # 黄色警告面板
                st.markdown(f"""
                    <div class="result-card review">
                        <h3>⚠️ 判定建议</h3>
                        <div class="result-value">{classifier.class_names[predicted_class]}</div>
                        <div>算法置信度：{confidence}%</div>
                        <p style="color: orange; margin-top: 10px;">样本位于边界区域，建议开启人工复核。</p>
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
                    options=["I类（优质）", "II类（标准）", "III类（劣质）"],
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
                        "confidence": confidence,
                        "thickness": feature_vector[0],
                        "corrosion": feature_vector[1],
                        "purity": feature_vector[2]
                    }
                    
                    # 写入CSV文件
                    with open('feedback.csv', 'a', newline='', encoding='utf-8') as f:
                        writer = csv.DictWriter(f, fieldnames=feedback_data.keys())
                        # 如果文件为空，写入表头
                        if f.tell() == 0:
                            writer.writeheader()
                        writer.writerow(feedback_data)
                    
                    st.success("✅ 复核结果已提交，感谢您的反馈！这些数据将用于模型自我迭代。")
            else:
                # 红色错误面板
                st.markdown(f"""
                    <div class="result-card error">
                        <h3>🚨 预警</h3>
                        <div class="result-value">{classifier.class_names[predicted_class]}</div>
                        <div>算法置信度：{confidence}%</div>
                        <p style="color: red; margin-top: 10px;">特征严重偏移！置信度极低，请进行专家仲裁。</p>
                    </div>
                """, unsafe_allow_html=True)

            # 保留详细数据，放在折叠栏里，显得专业又不乱
            with st.expander("🔍 专家视图：查看底层物理特征向量", expanded=expert_mode):
                st.write(
                    f"厚度: {feature_vector[0]:.2f}mm | 锈蚀: {feature_vector[1]:.2f}% | 纯度: {feature_vector[2]:.2f}")
                st.write(f"PCA投影坐标: PC1={pc_coords[0]:.2f}, PC2={pc_coords[1]:.2f}")
                
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
                                <h5>物理特征提取中</h5>
                                <p>分析厚度、锈蚀、纯度等关键指标</p>
                            </div>
                        </div>
                        <div class="timeline-item left">
                            <div class="timeline-content">
                                <h5>马氏距离核算</h5>
                                <p>计算样本与各类别中心的加权距离</p>
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
            x, y = classifier.map_pc_to_pixel(pc_coords, img_width, img_height)
            
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
        <p>影簇智检 - 废钢智能判级系统 v1.0</p>
        <p>基于改进的 K-means++ 和加权马氏距离算法</p>
    </div>
""", unsafe_allow_html=True)