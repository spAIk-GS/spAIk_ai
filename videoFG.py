def generate_posture_feedback(head_down_ratio, arm_movement_ratio):
    # 시선 피드백
    if head_down_ratio > 0.37:
        gaze_level = '나쁨'
    elif head_down_ratio > 0.18:
        gaze_level = "나쁨"
    else:
        gaze_level = "좋음"

    # 제스처 피드백
    if arm_movement_ratio < 0.29:
        gesture_level = "나쁨"
    elif arm_movement_ratio > 0.75:
        gesture_level = "좋음"
    else:
        gesture_level = "좋음"

    return gaze_level, gesture_level
