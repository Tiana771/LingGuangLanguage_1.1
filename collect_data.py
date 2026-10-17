import cv2
import mediapipe as mp
import numpy as np
import os
from sign_features import pack_hands

# ==================== 1. 基础配置 ====================
DATA_DIR = "sign_data"
os.makedirs(DATA_DIR, exist_ok=True)

# 10个手语词对应的按键（可根据你们实际选词修改）
word_map = {
    ord('0'): "你好", ord('1'): "谢谢", ord('2'): "请问", ord('3'): "医院", ord('4'): "厕所",
    ord('5'): "欢迎", ord('6'): "帮助", ord('7'): "报警", ord('8'): "吃饭", ord('9'): "再见"
}

# 统计每个词已经录了多少次
count_map = {key: 0 for key in word_map.keys()}

# 每个样本录制的帧数（60帧约1秒，包含完整动作）
FRAMES_PER_SAMPLE = 60

# ==================== 2. 初始化 MediaPipe ====================
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# 开启双手检测
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,          # 关键修改：改为2，支持双手
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

# ==================== 3. 开启摄像头 ====================
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
print("摄像头已开启。按 0-9 开始录制，按 Q 退出。")

recording = False
record_frames = []
record_label = None

# ==================== 4. 主循环 ====================
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # 镜像翻转，符合照镜子的直觉
    frame = cv2.flip(frame, 1)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    h, w, _ = frame.shape
    
    # --- 状态提示 ---
    status_text = "Ready (Press 0-9 to record)"
    if recording:
        status_text = f"Recording {word_map[record_label]}... {len(record_frames)}/{FRAMES_PER_SAMPLE}"
    
    cv2.putText(frame, status_text, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # --- 特征提取核心逻辑 ---
    # Frame pixels and handedness mapping match the shared browser preprocessing contract.
    current_features = np.zeros(126)
    if results.multi_hand_landmarks:
        detected_hands = []
        for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            classification = handedness.classification[0]
            detected_hands.append((hand_landmarks.landmark, classification.label, classification.score))
        current_features = pack_hands(detected_hands)

    # --- 录制逻辑 ---
    if recording:
        record_frames.append(current_features)
        
        if len(record_frames) >= FRAMES_PER_SAMPLE:
            label_char = chr(record_label)
            word = word_map[record_label]
            file_name = f"{DATA_DIR}/{label_char}_{word}_{count_map[record_label]}.npy"
            
            # 保存为 (60, 126) 的数组
            np.save(file_name, np.array(record_frames))
            
            print(f"已保存: {file_name}，形状: {np.array(record_frames).shape}")
            count_map[record_label] += 1
            
            # 重置状态
            recording = False
            record_frames = []
            record_label = None

    cv2.imshow("Sign Language Data Collector (Dual Hand)", frame)

    # --- 按键处理 ---
    key = cv2.waitKey(1) & 0xFF

    if key == ord('q') or key == ord('Q'):
        break
    elif key in word_map and not recording:
        recording = True
        record_label = key
        record_frames = []
        print(f"开始录制: {word_map[key]}，请保持动作，不要出画...")

# ==================== 5. 释放资源 ====================
cap.release()
cv2.destroyAllWindows()
print("采集结束！")
