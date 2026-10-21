# Auto-Glass

Auto-Glass（仓库名 Auto-Glass-Model）是一组眼轴相关的研究原型脚本。三个模块彼此独立，都在程序生成的模拟数据上拟合，不读取临床生物测量记录。

- `axiallength1`：用随机森林，按年龄组与一天中的时刻估计眼轴长度，并画出一日变化曲线。
- `factors2`：用支持向量回归（SVR），分析光照强度、瞳孔直径、环境评分和认知负荷对眼轴变化量的影响，并用 SHAP 查看特征贡献。
- `facialdetect3`：在模拟的面部与环境特征上训练 SVR，估计眼轴长度（AL）和晶状体厚度（LT）；另有一段加载 YOLOv8n 的静态图片演示。

本仓库用于研究演示。临床检测不在这些脚本的范围内。项目已申请专利，请勿用于商业用途。

## 目录

```
.
├── axiallength1/                 # 眼轴长度日变化
│   ├── aldataop.py               # 生成模拟数据；导入时写入固定绝对路径
│   ├── function.py               # 训练随机森林、保存曲线、命令行预测
│   ├── nerturalstudy.py          # 空文件
│   └── simulated_data.csv        # 已保存的模拟表；训练脚本导入时会重新生成，不读此文件
├── factors2/                     # 眼轴变化量与影响因素
│   ├── dataintegrate.py          # 生成整合表并做标准化
│   ├── integrated_eye_data.csv   # svmmodel.py 读取的数据
│   ├── process.py                # 预处理草稿；现有 csv 无 BlueLight_nm
│   └── svmmodel.py               # SVR、MAE/R² 与 SHAP 摘要图
├── facialdetect3/                # 面部特征与检测演示
│   ├── facedetect.py             # 模拟特征上的 SVR，以及摄像头循环
│   ├── demo_detect_and_predict.py # YOLOv8n 静态图演示
│   ├── processdata.py            # 生成模拟样本并划分训练/测试
│   ├── model.py                  # 网格搜索与置换重要性（不能单独运行）
│   ├── traindata.py              # 合成样本草稿（缺少依赖导入）
│   ├── word.py                   # 生成项目工作总结 Word
│   ├── ppt.py                    # 生成算法介绍幻灯片
│   ├── yolov8n.pt                # YOLOv8n 权重
│   ├── cat.jpg                   # 演示脚本读取的图片
│   ├── task3：model and CV-yolo.md # 任务设计稿
│   ├── 算法技术.pptx
│   └── 项目工作总结.docx
└── imgs/                         # 六个年龄组的眼轴日变化图
    ├── Infant_(0-1_yr).png
    ├── Toddler_(2-5_yrs).png
    ├── Preschool_(6-7_yrs).png
    ├── School-aged_(8-12_yrs).png
    ├── Teenager_(13-15_yrs).png
    └── Teenager_(16-18_yrs).png
```

`facialdetect3/` 中还有 `test.jpg`、`test.png`。当前演示脚本没有使用这两张图。

## 模块说明

### 眼轴长度日变化（axiallength1）

`aldataop.py` 生成六个年龄组的模拟眼轴，标签为 `Infant (0-1 yr)`、`Toddler (2-5 yrs)`、`Preschool (6-7 yrs)`、`School-aged (8-12 yrs)`、`Teenager (13-15 yrs)`、`Teenager (16-18 yrs)`。默认 5000 组基线，每组展开为 24 个时刻（共 120000 行）；日内变化由余弦函数生成。仓库里的 `simulated_data.csv` 与这一默认规模一致，但 `function.py` 不读取该文件，而是在导入时重新生成。

`function.py` 只把年龄组（`LabelEncoder`）和小时作为特征，眼轴长度（毫米）作为目标，用 `RandomForestRegressor` 拟合，并按 8:2 划分训练与测试。脚本会为每个年龄组保存一条 24 小时曲线，然后在命令行读取年龄组和时间（`HH:MM:SS`）并打印估计值。年龄组必须与上述英文标签完全一致。

`nerturalstudy.py` 没有代码。

### 影响因素（factors2）

`svmmodel.py` 读取 `factors2/integrated_eye_data.csv`。目标是 `Delta_AL_mm`（当前眼轴减去基线），特征是：

- `Illuminance_Lux`：光照强度
- `Pupil_Mean`：瞳孔直径的 5 行滑动平均
- `Env_Score`：环境评分
- `Cognitive_Load`：认知负荷

年龄组（3～6 岁、7～12 岁、13～18 岁）只用于分层划分，不进入特征。脚本按 8:2 划分后做标准化，训练 RBF 核 SVR（`C=1.0`，`epsilon=0.1`），打印 MAE 和 R²，再用 SHAP 的 `KernelExplainer` 画特征摘要图。已提交的 csv 有 2000 行，特征列在生成时已经标准化，`svmmodel.py` 会再标准化一次。

`dataintegrate.py` 按年龄组、光照场景和认知负荷生成这 2000 行，并把标准化后的表写到**当前目录**的 `integrated_eye_data.csv`。`process.py` 会读取 `factors2/integrated_eye_data.csv`，并访问表中不存在的 `BlueLight_nm`，直接运行会报错。这两个文件都导入了 SMOTE，正文没有调用过采样。

### 面部检测与指标估计（facialdetect3）

`facedetect.py` 与 `processdata.py` 生成 4.0–12.9 岁儿童的模拟会话（200 名儿童、每人 10 次）。SVR 的目标是 `Current_AL_mm` 和 `Current_LT_mm`。输入包括年龄、性别、族裔、基线眼轴、基线晶状体厚度、基线等效球镜，以及环境、任务、光照、色温、阅读/屏幕距离、眨眼频率、瞳孔直径、眼睛开合度（EAR）、AU6/AU7 强度、主观疲劳、认知负荷、距上次休息时间和前夜睡眠时间。数值特征标准化，类别特征独热编码，按 `Child_ID` 划分训练与测试。`facedetect.py` 使用固定超参数（`C=10`，`gamma='scale'`，`epsilon=0.05`）。网格搜索写在 `model.py`，不在这条训练路径里。

摄像头循环里的人脸框是固定比例的模拟框。眨眼、瞳孔、EAR 和 AU 强度由随机数生成，画面本身不参与计算。光照、距离和睡眠等环境量同样是模拟值。预测调用了未定义的 `al_predictor_pipeline` 和 `lt_predictor_pipeline`；上文训练得到的是 `best_al_model` 和 `best_lt_model`。异常被捕获后，画面上的 AL/LT 保持为缺失值，检测框和模拟的光照、眨眼、瞳孔、EAR 仍会绘制。按 `q` 退出。

`demo_detect_and_predict.py` 用 ultralytics 加载 `yolov8n.pt`。这是通用检测权重，脚本注释写明可以换成人脸专用模型。它读取 `facialdetect3/cat.jpg`，在检测框旁标注由瞳孔直径和光照代入公式得到的眼轴演示值。文件中的 `StandardScaler` 和 `SVR` 不参与这次计算。权重路径写成 `yolov8n.pt`，图片路径写成 `facialdetect3/cat.jpg`。在仓库根目录运行时，图片指向现有文件，权重指向根目录下的 `yolov8n.pt`，而仓库中的权重位于 `facialdetect3/yolov8n.pt`。进入 `facialdetect3/` 后，权重文件名一致，图片路径则多了一层目录。

`model.py` 包含 AL/LT 的 SVR 网格搜索、测试集误差和置换重要性，变量来自 `processdata.py` 所在的同一段解释器会话，单独运行会因缺少 `Pipeline`、训练集等名字而失败。`traindata.py` 是合成样本的函数草稿，使用了 `numpy` 和 `random` 但没有导入。

`word.py`、`ppt.py` 以及目录里的 Word、PPT，只生成或存放项目文稿，不参与训练或预测。`task3：model and CV-yolo.md` 讨论微表情、眼动和 YOLO 实时采样的设计，其中的采集设备和网络结构没有对应的实现。

## 依赖

仓库根目录没有 `requirements.txt`。运行对应脚本前，需要已安装该文件所导入的包：

| 脚本 | 需要的第三方包 |
| --- | --- |
| `axiallength1/function.py`（会导入 `aldataop.py`） | pandas、numpy、scikit-learn、matplotlib |
| `factors2/svmmodel.py` | pandas、scikit-learn、shap、matplotlib |
| `factors2/dataintegrate.py` | pandas、numpy、scikit-learn、imbalanced-learn |
| `factors2/process.py` | pandas、scikit-learn、imbalanced-learn |
| `facialdetect3/facedetect.py`、`processdata.py` | pandas、numpy、scikit-learn、matplotlib、seaborn、opencv-python |
| `facialdetect3/demo_detect_and_predict.py` | opencv-python、numpy、ultralytics、pandas、scikit-learn |
| `facialdetect3/word.py` | python-docx |
| `facialdetect3/ppt.py` | python-pptx |

`facedetect.py` 和 `processdata.py` 导入了 seaborn，正文没有调用它。`process.py` 与 `dataintegrate.py` 导入了 imbalanced-learn，正文没有调用 SMOTE。

## 运行

眼轴模块在 `axiallength1` 目录下执行，因为 `function.py` 写的是 `from aldataop import simulated_data`：

```bash
cd axiallength1
python function.py
```

导入 `aldataop.py` 时会立刻生成数据，并写入 `/Environment/engineer/axiallength/simulated_data.csv`。该目录不存在时导入失败，后面的训练不会开始。训练能开始时，曲线图写到**当前目录**下的 `imgs/`，文件名与年龄组标签对应；仓库里已有的图在根目录 `imgs/`。接着脚本等待命令行输入，例如年龄组 `School-aged (8-12 yrs)`、时间 `10:30:00`。图中的中文标题依赖系统里的 SimHei 字体。

影响因素模块在仓库根目录执行，因为它按 `factors2/integrated_eye_data.csv` 找文件：

```bash
python factors2/svmmodel.py
```

SHAP 摘要图会打开绘图窗口，`KernelExplainer` 在这 2000 行上计算量较大。

面部摄像头脚本在仓库根目录执行：

```bash
python facialdetect3/facedetect.py
```

需要本机默认摄像头。打不开摄像头时脚本会直接退出。能打开时，窗口标题为 `Real-time Ocular Data Prediction from Face (Simulated)`。AL/LT 使用了未定义的变量名，画面上显示为缺失值。按 `q` 退出。静态图演示的两条相对路径见上一节。

## 使用限制

- 眼轴、晶状体厚度、光照、瞳孔和面部动作单元均为模拟值。换用临床数据需要另行采集、训练和验证，当前脚本没有这一步。
- 三个模块没有连成同一个监测流程，也没有视力建议、学生用眼监测或硬件联动。
- 项目已申请专利，请勿用于商业用途。
