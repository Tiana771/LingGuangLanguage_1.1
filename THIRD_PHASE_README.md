# 第三阶段：摄像头识别演示

默认启动使用已冻结的 B 特征模型与 `calibrated` 拒识配置。Flask 接收一条 `(60,126)` 序列，在后端对 `any_valid` 帧计算 252 维 mean+std，再执行分类。训练 bundle 的样本门槛仍记录为 30；为应对浏览器跟踪掉帧，实时推理门槛单独配置为 15。15--29 帧会进入模型，少于 15 帧拒识，这会带来与离线校准分布不同的风险。

## 启动

在项目根目录运行：

```powershell
.\.venv\Scripts\python.exe serve_sign_api.py --host 127.0.0.1 --port 5001
```

浏览器访问 `http://127.0.0.1:5001/`。启动日志会显示模型路径、拒识模式和实际置信度阈值。健康检查位于 `/api/health`。

## 拒识模式

配置文件：`config/rejection_profiles.json`。默认模式为 `calibrated`，阈值来自此前冻结的校准结果（0.78）。现场可显式选择 `demo`：

```powershell
.\.venv\Scripts\python.exe serve_sign_api.py --host 127.0.0.1 --port 5001 --rejection-mode demo
```

`demo` 阈值 0.60 是未经同等校准的降级选项。页面状态会标出当前模式；切换配置后需要重启 Flask。不要把降级模式的识别结果描述为经校准的可靠结果。

摄像头端 MediaPipe 检测、存在和跟踪阈值为 0.5。浏览器 Tasks Vision 的 handedness 标签会先适配训练采集端约定，再写入左右 63 维槽位。有效帧少于 15 时提示“请将双手完整放入框内”；特征已提取但置信度不足时提示“动作不清晰，暂未识别”。连续窗口投票确认后，结果保持显示 1.5 秒，短暂丢帧不会立即清空。

## 验证

```powershell
.\.venv\Scripts\python.exe -m unittest discover -v
node .\test_prediction_smoother.mjs
.\.venv\Scripts\python.exe .\verify_feature_parity.py
```

特征对照使用真实 `.npy` 的相同 `(60,126)` 序列，验证 Python 与浏览器 JavaScript 的 252 维计算误差。它不验证摄像头视频帧与采集时摄像头帧的完全一致性。

拒识校准只使用已知 10 类样本。项目没有专门无手、静止、背景或词表外负样本，真实负样本拒识能力尚未验证。

## 联调诊断

浏览器开发者工具打开 Console 后筛选 `[sign-predict]`，每个完成的预测会打印 accepted、label、candidate_label、confidence 和 valid_frame_count。Flask 启动终端也会逐个打印对应预测摘要；网页不会显示这些内部调试数据。摄像头缓冲/拒识状态更新做了限频，避免小字快速闪烁。

临时摄像头特征会逐请求保存到 `debug_frames/`，命名包含时间、候选词和置信度。完成短时采集后停止 Flask，再运行：

```powershell
.\.venv\Scripts\python.exe .\analyze_camera_features.py .\sign_data_a\0_你好_A_0.npy
```

脚本将该离线样本与最新摄像头样本按方案 B 比较帧数、左右槽占用和特征数值范围；也可用 `--camera-sample` 指定要检查的调试文件。调试目录被 Git 忽略。连续运行会累积文件，采集完应停止服务。
