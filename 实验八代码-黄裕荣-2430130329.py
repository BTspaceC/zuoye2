import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD
import os

# ==========================================
# 0. 页面配置与美化样式设置
# ==========================================
st.set_page_config(
    page_title="在线笑话智能推荐系统 (SVD)",
    page_icon="😄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 注入 CSS 样式，打造高级暗色/明亮融合的极简现代风
st.markdown("""
<style>
    .main {
        background-color: #fcfcfd;
    }
    .stApp {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    .joke-card {
        background: #ffffff;
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05), 0 2px 4px -1px rgba(0, 0, 0, 0.03);
        border-top: 5px solid #4F46E5;
        margin-bottom: 20px;
        min-height: 160px;
        font-size: 0.95rem;
        color: #1F2937;
        line-height: 1.6;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
    }
    .rec-joke-card {
        background: #ffffff;
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05), 0 2px 4px -1px rgba(0, 0, 0, 0.03);
        border-top: 5px solid #10B981;
        margin-bottom: 20px;
        min-height: 160px;
        font-size: 0.95rem;
        color: #1F2937;
        line-height: 1.6;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
    }
    .stat-card {
        background: #F3F4F6;
        border-radius: 8px;
        padding: 10px 15px;
        margin-bottom: 10px;
        border-left: 4px solid #9CA3AF;
    }
    .header-style {
        color: #4F46E5;
        font-weight: 800;
        margin-bottom: 0px;
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 1. 数据加载与预处理 (使用 st.cache_data 缓存)
# ==========================================
@st.cache_data
def load_data():
    """
    加载评分数据和笑话文本数据，并过滤掉无评分的无效笑话。
    processed_ratings.csv 格式：user_id, joke_id, rating
    Dataset4JokeSet.xlsx 格式：每一行代表一个笑话文本，无表头
    """
    # 检查当前目录是否存在数据文件，支持相对路径
    ratings_path = "processed_ratings.csv"
    jokes_path = "Dataset4JokeSet.xlsx"
    
    if not os.path.exists(ratings_path) or not os.path.exists(jokes_path):
        st.error(f"❌ 未能找到必要的数据文件，请检查当前目录下是否有 '{ratings_path}' 和 '{jokes_path}'。")
        st.stop()
        
    # 读取评分数据
    ratings = pd.read_csv(ratings_path)
    # 读取笑话文本，没有表头，列名命名为 "joke_text"
    jokes = pd.read_excel(jokes_path, header=None, names=["joke_text"])
    
    # 调整 joke_id 从 1 开始（Excel 的索引 0 对应评分数据中的 joke_id = 1）
    jokes['joke_id'] = jokes.index + 1
    
    # 过滤掉没有任何评分的笑话 (数据集中有 22 个笑话无评分，若不处理，在降维和相似度计算中会出错)
    valid_joke_ids = ratings['joke_id'].unique()
    valid_jokes = jokes[jokes['joke_id'].isin(valid_joke_ids)].copy()
    
    return ratings, valid_jokes

# ==========================================
# 2. 构建 SVD 潜在空间相似度矩阵 (使用 st.cache_resource 缓存)
# ==========================================
@st.cache_resource
def build_svd_similarity_matrix(ratings):
    """
    1. 将评分数据由长格式 pivot 转换为 宽格式矩阵 (笑话 x 用户)
    2. 使用缺失值填充（填充为0），作为协同过滤的标准预处理
    3. 调用 TruncatedSVD 进行潜在特征空间降维
    4. 在 20 维降维空间中计算余弦相似度，得到笑话之间的相似度矩阵
    """
    # 将长格式转换为宽格式：行(index)为 joke_id，列(columns)为 user_id
    pivot_df = ratings.pivot(index='joke_id', columns='user_id', values='rating')
    # 对缺失值（未评分项）进行 0 填充
    pivot_df_filled = pivot_df.fillna(0)
    
    # 初始化 TruncatedSVD，指定潜在因子维度 k = 20
    svd = TruncatedSVD(n_components=20, random_state=42)
    
    # 拟合并变换：将 (100, N_users) 的评分向量压缩为 (100, 20) 的笑话隐含特征表示
    joke_features_reduced = svd.fit_transform(pivot_df_filled)
    
    # 计算降维后特征向量之间的余弦相似度，形状为 (100, 100)
    sim_matrix = cosine_similarity(joke_features_reduced)
    
    # 转换为 DataFrame，方便基于 joke_id 进行快速索引
    sim_df = pd.DataFrame(sim_matrix, index=pivot_df_filled.index, columns=pivot_df_filled.index)
    
    # 计算这 20 个潜在因子所能解释的总方差比例，供界面显示
    explained_variance = float(svd.explained_variance_ratio_.sum())
    
    return sim_df, explained_variance

# 加载数据与初始化模型
with st.spinner("🔄 正在从本地加载笑话文本与评分数据集，并构建 SVD 隐含模型..."):
    ratings, valid_jokes = load_data()
    sim_df, exp_var = build_svd_similarity_matrix(ratings)

# ==========================================
# 3. 初始化 Session State 状态机
# ==========================================
if 'random_jokes' not in st.session_state:
    # 随机选择 3 个有效的笑话供用户打分
    st.session_state.random_jokes = valid_jokes.sample(3, random_state=None)
if 'user_ratings' not in st.session_state:
    st.session_state.user_ratings = {}
if 'recommendations' not in st.session_state:
    st.session_state.recommendations = None
if 'eval_ratings' not in st.session_state:
    st.session_state.eval_ratings = {}
if 'submitted_evaluation' not in st.session_state:
    st.session_state.submitted_evaluation = False

# ==========================================
# 4. 侧边栏：项目信息与学生配置
# ==========================================
with st.sidebar:
    st.markdown("<h2 class='header-style'>🤖 系统控制台</h2>", unsafe_allow_html=True)
    st.write("---")
    
    # 实验人员信息配置（便于截图和报告编写）
    st.subheader("👤 实验学生信息")
    student_name = st.text_input("学生姓名", value="黄裕荣", key="std_name")
    student_id = st.text_input("学生学号", value="2430130329", key="std_id")
    
    st.write("---")
    
    # 推荐模型参数展示
    st.subheader("⚙️ 模型配置 (SVD)")
    st.info(f"""
    - **算法方法**: TruncatedSVD
    - **隐含维度 (k)**: 20
    - **信息保留比率**: {exp_var*100:.2f}%
    - **相似度度量**: 余弦相似度
    """)
    
    # 流程控制
    st.subheader("🔄 流程重置")
    if st.button("换一批初始笑话 (重置应用)", use_container_width=True):
        st.session_state.random_jokes = valid_jokes.sample(3)
        st.session_state.user_ratings = {}
        st.session_state.recommendations = None
        st.session_state.eval_ratings = {}
        st.session_state.submitted_evaluation = False
        st.rerun()

# ==========================================
# 5. 主页面布局
# ==========================================
st.markdown("<h1 style='text-align: center; color: #4F46E5;'>😄 智能笑话推荐系统</h1>", unsafe_allow_html=True)
st.markdown(f"<p style='text-align: center; color: #6B7280;'>基于 <b>SVD 潜在因子模型</b> 的个性化笑话协同过滤推荐系统<br>实验操作人: <b>{student_name}</b> ({student_id})</p>", unsafe_allow_html=True)
st.write("---")

# ------------------------------------------
# Step 1: 随机笑话打分界面
# ------------------------------------------
st.markdown("### 🎯 第一步：评测你的幽默口味")
st.markdown("请对以下随机选出的 3 个笑话进行打分（-10 代表极不好笑，10 代表极好笑）。系统将通过你的打分在 SVD 压缩的潜在兴趣空间中识别你的偏好：")

cols = st.columns(3)
for i, (_, row) in enumerate(st.session_state.random_jokes.iterrows()):
    joke_id = row['joke_id']
    with cols[i]:
        # 使用自定义的 CSS 类渲染笑话卡片
        st.markdown(
            f"<div class='joke-card'><div><strong>笑话 #{joke_id}</strong><br><br>{row['joke_text']}</div></div>", 
            unsafe_allow_html=True
        )
        # 用 slider 收集评分，范围是 -10.0 到 10.0
        rating = st.slider(
            f"评分 (笑话 #{joke_id})", 
            min_value=-10.0, 
            max_value=10.0, 
            value=0.0, 
            step=0.1, 
            key=f"rate_{joke_id}"
        )
        st.session_state.user_ratings[joke_id] = rating

st.write("")
# 计算推荐按钮
if st.button("🚀 生成个性化推荐笑话", type="primary", use_container_width=True):
    user_joke_ids = list(st.session_state.user_ratings.keys())
    user_joke_ratings = list(st.session_state.user_ratings.values())
    
    with st.spinner("🔮 正在利用 SVD 潜在空间特征计算相关度，寻找您可能最喜欢的笑话..."):
        # 核心算法：
        # 1. 提取用户打过分的这3个笑话在相似度矩阵中的行（每个笑话与其他所有笑话的相似度）
        # 2. 用用户的实际评分作为权重，将相似度向量进行加权求和
        #    如果用户对某个笑话打高分，说明与其相似的笑话应该推荐；若打低分（甚至负分），则排斥与其相似的笑话。
        total_scores = pd.Series(0.0, index=sim_df.index)
        for j_id, r in zip(user_joke_ids, user_joke_ratings):
            total_scores += r * sim_df[j_id]
            
        # 3. 剔除用户已经读过并打过分的这 3 个笑话，避免重复推荐
        total_scores = total_scores.drop(labels=user_joke_ids, errors='ignore')
        
        # 4. 提取综合推荐分数最高的前 5 个笑话的 ID
        top_5_joke_ids = total_scores.nlargest(5).index.tolist()
        
        # 保存推荐笑话的数据框
        st.session_state.recommendations = valid_jokes[valid_jokes['joke_id'].isin(top_5_joke_ids)]
        st.session_state.submitted_evaluation = False  # 重置评价状态
        st.session_state.eval_ratings = {}

# ------------------------------------------
# Step 2: 展示推荐笑话与满意度打分
# ------------------------------------------
if st.session_state.recommendations is not None:
    st.write("---")
    st.markdown("### 🌟 第二步：专属你的个性化推荐笑话")
    st.markdown("根据你的打分及 SVD 隐含因子空间的相关性计算，我们精选了以下 5 个笑话。请给它们评分，看看我们猜的准不准：")
    
    rec_cols = st.columns(5)
    for i, (_, row) in enumerate(st.session_state.recommendations.iterrows()):
        joke_id = row['joke_id']
        with rec_cols[i]:
            st.markdown(
                f"<div class='rec-joke-card'><div><strong>推荐笑话 #{joke_id}</strong><br><br>{row['joke_text']}</div></div>", 
                unsafe_allow_html=True
            )
            eval_rating = st.slider(
                f"为推荐 #{joke_id} 打分", 
                min_value=-10.0, 
                max_value=10.0, 
                value=0.0, 
                step=0.1, 
                key=f"eval_{joke_id}"
            )
            st.session_state.eval_ratings[joke_id] = eval_rating
            
    st.write("")
    if st.button("📊 提交评价并查看满意度", use_container_width=True):
        st.session_state.submitted_evaluation = True

# ------------------------------------------
# Step 3: 满意度展示
# ------------------------------------------
if st.session_state.submitted_evaluation:
    st.write("---")
    st.markdown("### 📈 推荐系统效能评测")
    
    # 满意度算法：将打分归一化到 0 - 100%
    # 计算公式：
    # 每一个笑话的打分取值范围是 [-10, 10]，那么 5 个笑话的总分范围是 [-50, 50]。
    # 归一化公式为：满意度 = (总得分 - 最小值) / (最大值 - 最小值) = (sum + 50) / 100
    total_eval_score = sum(st.session_state.eval_ratings.values())
    satisfaction_percentage = (total_eval_score + 50) / 100 * 100
    
    # 特效：放气球
    st.balloons()
    
    col1, col2 = st.columns([1, 2])
    with col1:
        st.metric(
            label="🎯 推荐总体满意度", 
            value=f"{satisfaction_percentage:.1f}%", 
            delta=f"{total_eval_score:+.1f} (总积分)"
        )
    with col2:
        # 使用进度条直观呈现
        st.write("满意度可视化进度：")
        st.progress(int(satisfaction_percentage))
        
    # 根据满意度级别给用户展示不同的文案提示
    if satisfaction_percentage >= 80:
        st.success("🎉 简直完美！看来我们的 SVD 隐含模型精准抓取到了您的冷幽默和爆笑点！")
    elif satisfaction_percentage >= 50:
        st.info("💡 效果还行！系统大部分情况下能识别出您的笑话口味，还有小部分提升空间。")
    else:
        st.warning("🧐 哎呀，看来 SVD 在你的幽默感面前遭遇了滑铁卢，我们会持续优化因子挖掘算法！")

    # 导出日志以记录本次测试 (方便学生截图证明)
    st.subheader("📝 实验测试运行日志")
    log_df = pd.DataFrame({
        "指标": ["实验人", "学号", "初始打分笑话ID", "初始打分值", "推荐笑话ID", "推荐评分值", "总评价分", "满意度"],
        "数值": [
            student_name,
            student_id,
            str(list(st.session_state.user_ratings.keys())),
            str(list(st.session_state.user_ratings.values())),
            str(list(st.session_state.eval_ratings.keys())),
            str(list(st.session_state.eval_ratings.values())),
            f"{total_eval_score:.1f}",
            f"{satisfaction_percentage:.1f}%"
        ]
    })
    st.table(log_df)
