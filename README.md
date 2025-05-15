# IncomeInsight 
### 是一个集数据预处理、特征可视化、降维分析、聚类建模、分类与回归预测为一体的综合性数据分析平台。 项目以 UCI Adult Income 数据集为基础，通过一系列数据挖掘流程，挖掘收入背后的潜在影响因素，并以图形化方式展现结果，支持命令行与可交互 Web 展示。

## 项目简介

本项目基于成人收入数据集（Adult Dataset），  
采用机器学习方法对个人收入水平（<=50K 或 >50K 美元）进行分类预测，  
并结合降维、聚类、回归等方法进行深入分析与可视化，  
探索不同特征对收入水平的影响关系。  

项目特点：
- 数据清洗与预处理
- 特征统计分析与可视化
- PCA降维与二维可视化
- KMeans无监督聚类分析
- 随机森林分类与回归建模
- 丰富的图表与指标输出

---

##  项目目录结构
```
IncomeInsight/
├── data/                    # 数据集文件夹
│   ├── adult.data            # 训练数据
│   ├── adult.test            # 测试数据
│   ├── adult.names           # 字段描述文件
│   ├── Index                 # 索引文件
│   ├──train_data_processed.csv     # 预处理后用于训练的数据
│   └── old.adult.names       # 旧版字段描述文件
│    
├── models/                          # 模型保存文件
│   ├── classifier.pkl               # 训练好的分类模型
│   ├── kmeans.pkl                   # KMeans 聚类模型
│   ├── regressor.pkl                # 回归模型
│   ├── pca.pkl                      # PCA降维模型
│   └── pca_train.npy                # 降维后训练数据
│ 
├── report/                   # 保存可视化图表与模型评估结果
│   ├── feature_distributions/  # 特征分布直方图
│   ├── correlation_heatmap.png # 特征相关性热力图
│   ├── pca_scatter.png         # PCA降维散点图
│   ├── kmeans_clusters.png     # KMeans聚类散点图
│   ├── classification_report.txt # 分类模型报告
│   ├── classification_confusion_matrix.png # 分类混淆矩阵
│   ├── regression_report.txt   # 回归模型报告
│   └── regression_scatter.png  # 回归真实vs预测散点图
│
├── webapp/                          # 核心应用逻辑和 Web 前端
│   ├── static/                      # 静态资源目录（CSS/JS/图像等）
│   │
│   ├── templates/                   # HTML模板目录
│   │   ├── form.html                # 输入表单与预测结果展示页面
│   │   ├── history.html             # 查询历史记录页
│   │   └── statistics.html          # 展示统计图页面
│   │
│   ├── app.py                       # Flask Web 应用主入口
│ 
├── preprocess.py              # 数据读取与预处理模块
├── visualization.py           # 特征可视化模块
├── dimensionality_reduction.py # PCA降维模块
├── clustering.py              # KMeans聚类模块
├── classification.py          # 随机森林分类模块
├── regression.py              # 随机森林回归模块
├── main.py                    # 主程序（串联各模块一键运行）
│
├── requirements.txt           # 项目依赖列表
└── README.md                  # 项目说明文档
```
---

## 环境与依赖

推荐使用 Python 版本：**Python 3.8+**

安装所需Python依赖包：

```bash
pip install -r requirements.txt
```
核心依赖：
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- 
## 快速开始
### 1. 克隆或下载项目
```bash
git clone https://github.com/你的仓库地址/IncomeInsight.git
cd IncomeInsight
```
或者直接将项目文件夹复制到本地。
### 2. 安装依赖
确保你已经激活虚拟环境后，执行：
```bash
pip install -r requirements.txt
```
### 3. 运行项目主程序
```bash
python main.py
```
执行后，项目将自动完成：
- 数据预处理
- 特征统计与可视化
- PCA降维与散点图绘制
- KMeans聚类分析
- 收入分类建模与评估
- 收入回归建模与评估
所有图表和分析结果会保存到 report/ 文件夹下。

## 项目亮点展示：
- 数据预处理：处理缺失值、标签编码
- 数据分析：直观了解各特征分布及特征间相关性
- 降维可视化：通过PCA将高维数据压缩至2D，便于观察模式
- 无监督聚类：发现数据中的潜在人群分布
- 分类预测：准确预测个人收入水平
- 回归建模：探索收入趋势，评估模型性能
- 图表与报告：结果可视化丰富，便于分析与展示

## 致谢
数据来源：UCI Machine Learning Repository - Adult Data Set

本项目作为机器学习课程设计项目，旨在系统实践机器学习建模全流程。

感谢各开源社区提供的技术支持。