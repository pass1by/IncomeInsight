from flask import Flask, render_template, request, send_from_directory
import joblib
import numpy as np
import pandas as pd
import os
import json
import datetime
import matplotlib
matplotlib.use('Agg') # 强制使用Agg后端，避免Tkinter冲突
import matplotlib.pyplot as plt
from prepare_data import prepare_pca_background
from prepare_data import prepare_all_data

# 设置matplotlib支持中文
matplotlib.rcParams['font.sans-serif'] = ['SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False

app = Flask(__name__)

# 加载聚类分析报告
cluster_report = pd.read_csv('../report/cluster_analysis_report.csv')

# 加载模型
classifier = joblib.load('../models/classifier.pkl')
regressor = joblib.load('../models/regressor.pkl')
pca = joblib.load('../models/pca.pkl')
kmeans = joblib.load('../models/kmeans.pkl')

# 加载PCA背景数据
pca_background = np.load('../models/pca_train.npy')
pca_min = pca_background.min(axis=0)
pca_max = pca_background.max(axis=0)
pca_norm = (pca_background - pca_min) / (pca_max - pca_min) * 20 - 10

# 聚类解释函数
@app.route('/history')
def history():
    # 列出 static/ 目录下所有预测图
    static_dir = 'static'
    pca_images = []

    if os.path.exists(static_dir):
        for filename in os.listdir(static_dir):
            if filename.startswith('predict_pca_position_') and filename.endswith('.png'):
                pca_images.append(filename)

    # 按时间倒序排列（最新的图在最前面）
    pca_images.sort(reverse=True)

    return render_template('history.html', pca_images=pca_images)
# 数据预处理
def encode_input(data):
    education_to_num = {
        'Preschool': 1, '1st-4th': 2, '5th-6th': 3, '7th-8th': 4,
        '9th': 5, '10th': 6, '11th': 7, '12th': 8,
        'HS-grad': 9, 'Some-college': 10, 'Assoc-voc': 11,
        'Assoc-acdm': 12, 'Bachelors': 13, 'Masters': 14,
        'Prof-school': 15, 'Doctorate': 16
    }

    workclass_mapping = {
        'Private': 0,
        'Self-emp-not-inc': 1,
        'Self-emp-inc': 2,
        'Federal-gov': 3,
        'Local-gov': 4,
        'State-gov': 5,
        'Without-pay': 6,
        'Never-worked': 7
    }

    education_mapping = {
        'Preschool': 0,
        '1st-4th': 1,
        '5th-6th': 2,
        '7th-8th': 3,
        '9th': 4,
        '10th': 5,
        '11th': 6,
        '12th': 7,
        'HS-grad': 8,
        'Some-college': 9,
        'Assoc-voc': 10,
        'Assoc-acdm': 11,
        'Bachelors': 12,
        'Masters': 13,
        'Prof-school': 14,
        'Doctorate': 15
    }

    marital_mapping = {
        'Never-married': 0,
        'Married-civ-spouse': 1,
        'Divorced': 2,
        'Separated': 3,
        'Widowed': 4,
        'Married-spouse-absent': 5
    }

    occupation_mapping = {
        'Tech-support': 0,
        'Craft-repair': 1,
        'Other-service': 2,
        'Sales': 3,
        'Exec-managerial': 4,
        'Prof-specialty': 5,
        'Handlers-cleaners': 6,
        'Machine-op-inspct': 7,
        'Adm-clerical': 8,
        'Farming-fishing': 9,
        'Transport-moving': 10,
        'Protective-serv': 11,
        'Priv-house-serv': 12,
        'Armed-Forces': 13
    }

    relationship_mapping = {
        'Wife': 0,
        'Own-child': 1,
        'Husband': 2,
        'Not-in-family': 3,
        'Other-relative': 4,
        'Unmarried': 5
    }

    race_mapping = {
        'White': 0,
        'Asian-Pac-Islander': 1,
        'Amer-Indian-Eskimo': 2,
        'Other': 3,
        'Black': 4
    }

    sex_mapping = {
        'Male': 0,
        'Female': 1
    }

    native_country_mapping = {
        'United-States': 0,
        'Mexico': 1,
        'Philippines': 2,
        'Germany': 3,
        'Canada': 4,
        'Puerto-Rico': 5,
        'India': 6,
        'China': 7,
        'Cuba': 8,
        'England': 9,
        'Japan': 10,
        'Vietnam': 11,
        'Ireland': 12,
        'France': 13,
        'South-Korea': 14,
        'Other': 15
    }

    selected_education = data['education']
    education_num = education_to_num.get(selected_education, 9)

    encoded = {
        'age': int(data['age']),
        'workclass': workclass_mapping.get(data['workclass'], -1),
        'fnlwgt': 100000,
        'education': education_mapping.get(selected_education, -1),
        'education-num': education_num,
        'marital-status': marital_mapping.get(data['marital-status'], -1),
        'occupation': occupation_mapping.get(data['occupation'], -1),
        'relationship': relationship_mapping.get(data['relationship'], -1),
        'race': race_mapping.get(data['race'], -1),
        'sex': sex_mapping.get(data['sex'], -1),
        'capital-gain': int(data['capital-gain']),
        'capital-loss': int(data['capital-loss']),
        'hours-per-week': int(data['hours-per-week']),
        'native-country': native_country_mapping.get(data['native-country'], -1),
    }
    df = pd.DataFrame([encoded])
    df.columns = ['age', 'workclass', 'fnlwgt', 'education', 'education-num',
                  'marital-status', 'occupation', 'relationship', 'race', 'sex',
                  'capital-gain', 'capital-loss', 'hours-per-week', 'native-country']
    return df
@app.route('/statistics')
def statistics():
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')

    # 读取历史记录
    history_file = '../report/history.json'
    if os.path.exists(history_file):
        with open(history_file, 'r', encoding='utf-8') as f:
            history_data = json.load(f)
    else:
        history_data = []

    # 提取回归得分和聚类类别
    regression_scores = [item['regression_score'] for item in history_data]
    cluster_list = [item['cluster'] for item in history_data]

    # 聚类类别统计
    from collections import Counter
    cluster_counts = Counter(cluster_list)

    # 绘制聚类饼图
    pie_filename = f"cluster_pie_chart_{timestamp}.png"
    pie_path = f"static/{pie_filename}"

    plt.figure(figsize=(6,6))
    labels = [f"类别{i}" for i in cluster_counts.keys()]
    sizes = list(cluster_counts.values())
    plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=140)
    plt.axis('equal')
    plt.title('聚类类别分布')
    plt.savefig(pie_path)
    plt.close()

    # 绘制回归得分直方图
    hist_filename = f"regression_score_hist_{timestamp}.png"
    hist_path = f"static/{hist_filename}"

    plt.figure(figsize=(6,4))
    plt.hist(regression_scores, bins=10, color='skyblue', edgecolor='black')
    plt.title('回归趋势得分分布')
    plt.xlabel('得分 (0=低收入，1=高收入)')
    plt.ylabel('人数')
    plt.grid(True)
    plt.savefig(hist_path)
    plt.close()

    return render_template('statistics.html', pie_chart=pie_filename, hist_chart=hist_filename)

@app.route('/', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        form_data = request.form
        input_df = encode_input(form_data)

        # 模型预测
        y_pred_class = classifier.predict(input_df)
        y_pred_class_proba = classifier.predict_proba(input_df)
        y_pred_reg = regressor.predict(input_df)
        X_pca = pca.transform(input_df)
        cluster_id = kmeans.predict(X_pca)

        # 归一化PCA
        X_pca_single_norm = (X_pca - pca_min) / (pca_max - pca_min) * 20 - 10
        # 生成时间戳
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')

        # 确保 static 目录存在
        static_dir = 'static'
        if not os.path.exists(static_dir):
            os.makedirs(static_dir)

        # 图像保存路径，带时间戳
        pca_img_filename = f"predict_pca_position_{timestamp}.png"
        pca_img_path = f"static/{pca_img_filename}"

        # 绘制PCA散点图
        if not os.path.exists('static'):
            os.makedirs('static')
        plt.figure(figsize=(6,6))
        plt.scatter(pca_norm[:,0], pca_norm[:,1], c='lightgray', s=10, label='其他样本')
        plt.scatter(X_pca_single_norm[0,0], X_pca_single_norm[0,1], c='red', s=200, marker='*', label='你的位置')
        plt.title('PCA降维后的位置')
        plt.xlabel('PCA-1')
        plt.ylabel('PCA-2')
        plt.legend()
        plt.grid(True)
        plt.savefig(pca_img_path)
        plt.close()

        # 选取特征画雷达图
        features = ['年龄', '工时', '资本增益', '资本损失']
        values = [
            int(form_data['age']),
            int(form_data['hours-per-week']),
            int(form_data['capital-gain']),
            int(form_data['capital-loss'])
        ]

        # 最大值列表（用于归一化）
        max_values = [100, 100, 99999, 5000]
        values_norm = [v / m for v, m in zip(values, max_values)]

        # 雷达图基础设置
        labels = np.array(features)
        angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False)

        # 闭合曲线
        values_norm = np.concatenate((values_norm, [values_norm[0]]))
        angles_closed = np.concatenate((angles, [angles[0]]))

        # 保存雷达图
        radar_filename = f"radar_chart_{timestamp}.png"
        radar_path = f"static/{radar_filename}"

        # 开始画图
        plt.figure(figsize=(6, 6))
        ax = plt.subplot(111, polar=True)
        plt.figure(figsize=(6, 6))
        ax = plt.subplot(111, polar=True)

        # 绘制雷达线和填充区域
        ax.plot(angles_closed, values_norm, color='red', linewidth=2)
        ax.fill(angles_closed, values_norm, color='skyblue', alpha=0.25)

        # 设置角度标签
        ax.set_thetagrids(angles * 180 / np.pi, labels, fontsize=12, fontweight='bold')

        # 设置极径刻度（0到1之间）
        ax.set_rgrids([0.2, 0.4, 0.6, 0.8, 1.0], labels=['20%', '40%', '60%', '80%', '100%'], angle=0, fontsize=10)

        # 设置雷达图半径范围（统一到1）
        ax.set_ylim(0, 1)

        # 网格线美化
        ax.yaxis.grid(True, linestyle='--', color='gray', alpha=0.7)
        ax.xaxis.grid(True, linestyle='--', color='gray', alpha=0.7)

        # 标题
        plt.title('单人特征雷达图', fontsize=16, fontweight='bold', pad=20)

        # 保存
        plt.savefig(radar_path)
        plt.close()

        # 生成总结
        class_label = '>50K' if y_pred_class[0] == 1 else '<=50K'
        class_prob = f"{y_pred_class_proba[0][1]*100:.2f}%"
        reg_result = y_pred_reg[0]
        cluster = int(cluster_id[0])
        pca1 = X_pca_single_norm[0,0]
        pca2 = X_pca_single_norm[0,1]

        # 生成总结（模块化）
        summary_text = ""

        #  收入分类预测部分
        summary_text += " 收入分类预测结果：\n"
        summary_text += f" 预测结果为：{'高收入 (>50K)' if y_pred_class[0] == 1 else '低收入 (≤50K)'}\n"
        summary_text += f" 高收入概率为：{y_pred_class_proba[0][1] * 100:.2f}%\n\n"

        #  人群聚类结果部分
        row = cluster_report[cluster_report['类别编号'] == int(cluster_id[0])]

        if not row.empty:
            income_ratio = float(row['高收入比例(%)'].values[0])
            avg_pc1 = float(row['PC1均值'].values[0])
            avg_pc2 = float(row['PC2均值'].values[0])
            avg_hours = float(row['hours-per-week均值'].values[0])
            avg_gain = float(row['capital-gain均值'].values[0])

            if income_ratio > 60:
                desc_income = "高收入人群"
            elif income_ratio > 30:
                desc_income = "中等收入人群"
            else:
                desc_income = "低收入人群"

            cluster_description = (
                f"你属于类别{cluster_id[0]}，属于{desc_income}（高收入比例{income_ratio:.1f}%）。\n"
                f"主成分均值：PC1={avg_pc1:.2f}, PC2={avg_pc2:.2f}。\n"
                f"平均工时：{avg_hours:.1f}小时/周，平均资本收益：{avg_gain:.1f}元。"
            )
        else:
            cluster_description = f"你属于类别{cluster_id[0]}，暂无详细统计信息。"

        # 写到summary_text里
        summary_text += " 人群聚类分析：\n"
        summary_text += cluster_description + "\n\n"
        #  回归趋势分析部分
        summary_text += " 回归趋势分析：\n"
        summary_text += f" 你的回归趋势得分是 {reg_result:.4f}，👉 表示你的整体经济水平偏向{'高收入' if reg_result > 0.5 else '中低收入'}群体。\n\n"

        #  PCA降维位置部分
        summary_text += " PCA特征空间位置：\n"
        summary_text += f"你的PCA降维后的坐标是 ({pca1:.4f}, {pca2:.4f})，👉 位于{'中心区域' if abs(pca1) < 5 and abs(pca2) < 5 else '边缘区域'}。\n"

        # 保存总结txt
        if not os.path.exists('../report'):
            os.makedirs('../report')
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        save_filename = f"prediction_summary_{timestamp}.txt"
        save_path = f"../report/{save_filename}"
        with open(save_path, 'w', encoding='utf-8') as f:
            f.write(summary_text)

        result = {
            'summary_text': summary_text,
            'download_link': f"/download/{save_filename}",
            'pca_img_path': f"static/{pca_img_filename}",
            'radar_img_path': f"static/{radar_filename}"
        }
        # 保存预测历史
        history_file = '../report/history.json'

        # 读取已有历史
        if os.path.exists(history_file):
            with open(history_file, 'r', encoding='utf-8') as f:
                history_data = json.load(f)
        else:
            history_data = []

        # 保存回去
        with open(history_file, 'w', encoding='utf-8') as f:
            json.dump(history_data, f, ensure_ascii=False, indent=4)
        # 加一条新记录
        new_record = {
            'timestamp': timestamp,
            'regression_score': float(reg_result),
            'cluster': int(cluster)
        }

        history_data.append(new_record)
        with open(history_file, 'w', encoding='utf-8') as f:
            json.dump(history_data, f, ensure_ascii=False, indent=4)

        return render_template('form.html', result=result)
    return render_template('form.html')

@app.route('/download/<filename>')
def download_file(filename):
    return send_from_directory('../report', filename, as_attachment=True)

if __name__ == '__main__':
    if os.environ.get("WERKZEUG_RUN_MAIN") == "true":
        prepare_all_data()
    app.run(debug=True)
