import cv2
import os
import mediapipe as mp

def main():
    # 1. MediaPipeの準備
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils  # 描画用のツール
    
    hands = mp_hands.Hands(
        static_image_mode=True, 
        max_num_hands=2, 
        min_detection_confidence=0.5
    )

    # 2. 画像の読み込み
    input_path = 'data/IMG01.jpg'
    output_path = 'data/output.jpg'
    
    if not os.path.exists(input_path):
        print(f"エラー: {input_path} が見つかりません。")
        return

    image = cv2.imread(input_path)
    # MediaPipe用にRGBへ変換
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # 3. 解析実行
    results = hands.process(image_rgb)

    # 4. 結果の描画と保存
    if results.multi_hand_landmarks:
        print(f"【成功】手が検出されました！ -> {output_path} に保存します")
        
        # 描画用に元の画像をコピー
        annotated_image = image.copy()
        
        for hand_landmarks in results.multi_hand_landmarks:
            # 画像の上に「骨格」と「関節」を描画
            mp_drawing.draw_landmarks(
                annotated_image,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=4), # 関節（緑）
                mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2) # 骨格（青）
            )
            
            # コンソールにも人差し指の座標を表示
            tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            print(f"人差し指先座標: x={tip.x:.4f}, y={tip.y:.4f}")

        # 結果を画像ファイルとして保存
        cv2.imwrite(output_path, annotated_image)
    else:
        print("手が検出されませんでした。")

    hands.close()

if __name__ == "__main__":
    main()