import cv2
import numpy as np
import mediapipe as mp
import os

# ギター指板の物理定数
L_22 = 0.7184

def get_fret_and_string(tx, ty, board_w, board_h):
    relative_pos = (tx / board_w) * L_22
    if 0 <= relative_pos < 1:
        fret_num = -12 * np.log2(1 - relative_pos)
        fret = int(fret_num + 0.5)
        # y座標を6分割して弦を判定
        string = 6 - int(ty / (board_h / 6))
        return max(1, min(6, string)), fret
    return None, None

selected_pts = []

def click_event(event, x, y, flags, params):
    if event == cv2.EVENT_LBUTTONDOWN:
        selected_pts.append([x, y])
        # 弦に基づいたラベルを表示
        labels = ["6弦ナット", "1弦ナット", "6弦末端", "1弦末端"]
        current_label = labels[len(selected_pts)-1] if len(selected_pts) <= 4 else ""
        
        print(f"{current_label}を取得: [{x}, {y}]")
        
        # 描画更新
        cv2.circle(params['img'], (x, y), 5, (0, 0, 255), -1)
        cv2.putText(params['img'], current_label, (x+10, y+10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        cv2.imshow('Step 1: Select 4 Points', params['img'])

def main():
    # --- STEP 1: 弦に基づいた4隅の取得 ---
    input_path = 'data/test.jpg'
    image = cv2.imread(input_path)
    if image is None: return
    
    display_img = image.copy()
    cv2.imshow('Step 1: Select 4 Points', display_img)
    cv2.setMouseCallback('Step 1: Select 4 Points', click_event, {'img': display_img})
    
    print("--- 座標取得を開始します ---")
    print("以下の順番で【指板の角】をクリックしてください：")
    print("1. 6弦側のナット付近")
    print("2. 1弦側のナット付近")
    print("3. 6弦側のボディ付近（末端）")
    print("4. 1弦側のボディ付近（末端）")
    print("\n4点選び終わったら、画像ウィンドウで何かキーを押してください。")
    
    cv2.waitKey(0)
    cv2.destroyWindow('Step 1: Select 4 Points')

    if len(selected_pts) != 4:
        print("エラー: 4つの点が選択されませんでした。")
        return

    # 射影変換行列 M の作成
    # selected_pts の順序 [6n, 1n, 6e, 1e] に合わせて dst_pts を対応させる
    src_pts = np.array(selected_pts, dtype=np.float32)
    board_w, board_h = 1000, 200
    dst_pts = np.array([
        [0, 0],         # 6弦ナット (左上)
        [0, board_h],   # 1弦ナット (左下)
        [board_w, 0],   # 6弦末端   (右上)
        [board_w, board_h] # 1弦末端 (右下)
    ], dtype=np.float32)
    M = cv2.getPerspectiveTransform(src_pts, dst_pts)

    # --- STEP 2: MediaPipe解析 ---
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.7)

    h, w, _ = image.shape
    results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            finger_tips = {
                "Index": mp_hands.HandLandmark.INDEX_FINGER_TIP,
                "Middle": mp_hands.HandLandmark.MIDDLE_FINGER_TIP,
                "Ring": mp_hands.HandLandmark.RING_FINGER_TIP,
                "Pinky": mp_hands.HandLandmark.PINKY_TIP
            }

            for name, tip_idx in finger_tips.items():
                tip_lm = hand_landmarks.landmark[tip_idx]
                px, py = int(tip_lm.x * w), int(tip_lm.y * h)
                
                # 指先座標を指板平面に変換
                pt = cv2.perspectiveTransform(np.array([[[px, py]]], dtype=np.float32), M)[0][0]
                s, f = get_fret_and_string(pt[0], pt[1], board_w, board_h)
                
                if s is not None:
                    print(f"{name}指: {s}弦 {f}フレット")
                    cv2.putText(image, f"{s}s {f}f", (px, py - 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    cv2.circle(image, (px, py), 5, (0, 255, 0), -1)

            mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    cv2.imshow('Result', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    hands.close()

if __name__ == "__main__":
    main()