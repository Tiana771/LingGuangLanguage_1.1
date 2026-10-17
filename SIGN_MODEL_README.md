# 手语小模型基线

## 当前基线范围

本基线使用现有 `.npy` 文件名解析类别，每个文件是一条完整序列，不拆帧。特征为将 `(60, 126)` 序列分成 10 个连续时间段、每段按通道取均值，得到 1260 维向量；分类器为 `StandardScaler + RBF SVC`。预测类别取 `SVC.predict()`，拒识分数取该类别对应的 `predict_proba()` 值，以保持与分类报告相同的类别决策。

首轮默认每类抽取 60% 样本，再分层留出 20% 作验证。随机种子固定为 42。所有数据目前来自一位录制者，因此结果只代表单人数据上的流程验证。SVC 概率和默认拒识阈值 0.35 都是实验配置，须用更多独立数据校准。

## 运行

在已安装 `requirements.txt` 依赖的 Python 环境中：

```powershell
python train_sign_baseline.py --fraction 0.6 --validation-fraction 0.2 --seed 42
python predict_sign.py sign_data/0_你好_0.npy --model model_output/run-<时间戳>/model.joblib --threshold 0.35
python serve_sign_api.py --model model_output/run-<时间戳>/model.joblib --threshold 0.35 --port 5001
python -m unittest -v test_sign_baseline.py
```

训练默认在 `model_output/` 下新建带时间戳的运行目录，不覆盖既有结果。目录内有 `model.joblib`、`metrics.json`、`classification_report.txt`、`confusion_matrix.csv`、`scan_report.json`。原始 `sign_data/` 只读。

## HTTP 接口

健康检查：

```http
GET /api/health
```

模型未加载时返回 HTTP 503；加载成功时返回状态、类别和输入形状。

预测：

```http
POST /api/predict
Content-Type: application/json
```

请求体契约：

```text
sequence: number[60][126]
```

客户端发送 JSON 对象，例如 `{ "sequence": matrix }`，其中 `matrix` 必须是 60 行、每行 126 个有限数值的精确 `(60, 126)` 矩阵。成功响应包含 `label`、`confidence`、`accepted`。低于阈值时 `label` 为 `null`，`message` 为“暂未识别，请重试”。非法形状、非数值或 NaN/Inf 返回 HTTP 400。

## 实时摄像头识别

当前页面通过浏览器本地 `@mediapipe/tasks-vision` Hand Landmarker 提取关键点。任务模型位于 `assets/hand_landmarker.task`，WASM 与 JavaScript 运行库通过 Flask 同源服务从 `node_modules/@mediapipe/tasks-vision/` 提供，不从 Google 或 CDN 下载模型资源。

前处理与采集脚本共用 `sign_features.py` / `camera_features.mjs` 约定：先将输入帧镜像，再按采集时的 handedness 映射把 MediaPipe Left 放入右手槽位、Right 放入左手槽位；减去各手腕点 xyz 后，以每只手最大绝对值加 `1e-6` 归一化；按左手 63 维、右手 63 维拼成 126 维帧。窗口为连续 60 帧，窗口内少于 13 个有效手部帧会拒识。摄像头图像和关键点只在浏览器处理，只有特征窗口发往本机 `/api/predict`。

推理窗口间隔约 450 ms，最近 5 次输出中至少 3 次一致才显示词语；无手、低置信度、权限/API/模型错误都有独立状态。关闭摄像头、离开页面或切换到查询入口会停止视频轨道并释放 Hand Landmarker。摄像头需在 localhost/HTTPS 下由用户授权。

安装前端运行库并启动本地服务：

```powershell
npm install
python serve_sign_api.py --model model_output/multisigner_run-<时间戳>/final_demo_model.joblib --port 5001
```

运行校验：

```powershell
python -m unittest -v test_sign_baseline.py test_multisigner.py test_camera_features.py
```

`test_camera_features.py` 使用同一组合成关键点比较 Python 与浏览器特征向量。真实手势准确率仍受三录制者留一评估结果限制；摄像头识别是限定十词分类器，OOV/无手拒识没有专门负样本性能保证。
