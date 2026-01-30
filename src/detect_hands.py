import cv2
import numpy as np
import mediapipe as mp
import os

def get_fret_and_string(tx, ty, board_w, board_h):
    L_22 = 0.7184
    relative_pos = (tx / board_w) * L_22
    if 0 <= relative_pos < 1:
        fret_num = -12 * np.log2(1 - relative_pos)
        fret = int(fret_num + 0.5)
        string = 6 - int(ty / (board_h / 6))
        return max(1, min(6, string)), fret
    return None, None

def main():
    # 佐藤さんの取得した座標
    src_pts = np.array([[752, 216], [763, 245], [387, 302], [395, 340]], dtype=np.float32)
    board_w, board_h = 1000, 200
    dst_pts = np.array([[0, 0], [0, board_h], [board_w, 0], [board_w, board_h]], dtype=np.float32)
    M = cv2.getPerspectiveTransform(src_pts, dst_pts)

    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    hands = mp_hands.Hands(static_image_mode=True, max_num_hands=2, min_detection_confidence=0.5, model_complexity=1)
    
    input_path = 'data/test.jpg'
    output_path = 'data/output.jpg'
    image = cv2.imread(input_path)
    if image is None: return
    
    h, w, _ = image.shape
    overlay = image.copy()
    results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            wrist = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]
            if wrist.x < 0.5: continue

            # 1. 骨格の描画設定（緑の点と線をはっきりさせる）
            # 指先（TIP）までしっかり緑の点が出るように指定
            mp_drawing.draw_landmarks(
                overlay, 
                hand_landmarks, 
                mp_hands.HAND_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=3), # 関節の点（緑）
                mp_drawing.DrawingSpec(color=(0, 200, 0), thickness=2)                 # 骨格の線（緑）
            )

            # 判定に使用する指先（TIP）のインデックス
            finger_tips = {
                "Index": mp_hands.HandLandmark.INDEX_FINGER_TIP,
                "Middle": mp_hands.HandLandmark.MIDDLE_FINGER_TIP,
                "Ring": mp_hands.HandLandmark.RING_FINGER_TIP,
                "Pinky": mp_hands.HandLandmark.PINKY_TIP
            }

            for name, tip_idx in finger_tips.items():
                tip_lm = hand_landmarks.landmark[tip_idx]
                
                # --- 純粋に指先（TIP）そのものを判定座標にする ---
                px, py = int(tip_lm.x * w), int(tip_lm.y * h)
                
                # 2. 判定ポイントを薄い赤で描画（緑の点と重ねて確認しやすくする）
                # 半透明レイヤーなので、緑の点の上に赤が乗って「ここを判定した」とわかるようになります
                cv2.circle(overlay, (px, py), 8, (0, 0, 255), -1)
                cv2.putText(overlay, f"{name}_Tip", (px + 10, py), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

                # 座標変換とフレット計算
                pt = cv2.perspectiveTransform(np.array([[[px, py]]], dtype=np.float32), M)[0][0]
                s, f = get_fret_and_string(pt[0], pt[1], board_w, board_h)
                if s is not None:
                    print(f"{name}指: {s}弦 {f}フレット")

    # 3. 合成して保存（透明度 0.5）
    alpha = 0.5
    output_image = cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)

    cv2.imwrite(output_path, output_image)
    print(f"解析完了。output.jpg で緑の点（AI認識）と赤の点（計算位置）が重なっているか確認してください。")
    hands.close()

if __name__ == "__main__":
    main()