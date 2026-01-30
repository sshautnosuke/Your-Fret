import cv2
import numpy as np
import mediapipe as mp
import os

# 物理定数
L_22 = 0.7184
BOARD_W, BOARD_H = 1000, 200

def get_fret_and_string(tx, ty):
    relative_pos = (tx / BOARD_W) * L_22
    if 0 <= relative_pos < 1:
        fret_num = -12 * np.log2(1 - relative_pos)
        fret = int(fret_num + 0.5)
        string = 6 - int(ty / (BOARD_H / 6))
        return max(1, min(6, string)), fret
    return None, None

selected_pts = []
labels = ["6弦ナット", "1弦ナット", "6弦22フレット", "1弦22フレット"]
finger_names = {8: "Index", 12: "Middle", 16: "Ring", 20: "Pinky"}

def print_next_label():
    if len(selected_pts) < len(labels):
        print(f"\n>>> 【 {labels[len(selected_pts)]} 】をクリックしてください")
    else:
        print("\n>>> 4点のラベリングが完了。解析を開始します。")

def click_event(event, x, y, flags, params):
    global selected_pts
    img_orig = params['img_orig']
    win_name = params['window_name']
    if event == cv2.EVENT_LBUTTONDOWN and len(selected_pts) < len(labels):
        selected_pts.append([x, y])
        print(f"  [取得] {labels[len(selected_pts)-1]}: ({x}, {y})")
        print_next_label()
    elif event == cv2.EVENT_RBUTTONDOWN and len(selected_pts) > 0:
        selected_pts.pop()
        print_next_label()

    display_img = img_orig.copy()
    for i, pt in enumerate(selected_pts):
        cv2.circle(display_img, (pt[0], pt[1]), 5, (0, 0, 255), -1)
        cv2.putText(display_img, labels[i], (pt[0]+10, pt[1]+10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    cv2.imshow(win_name, display_img)

def main():
    video_path = 'data/Fret_test5.mp4'
    if not os.path.exists(video_path):
        print(f"【エラー】ファイルが見つかりません: {video_path}")
        return

    cap = cv2.VideoCapture(video_path)
    ret, first_frame = cap.read()
    if not ret: return

    orig_h, orig_w = first_frame.shape[:2]
    win_name = 'Step 1: Labeling (4 Points)'
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win_name, 1000, int(orig_h * (1000 / orig_w)))
    
    params = {'img_orig': first_frame, 'window_name': win_name}
    cv2.imshow(win_name, first_frame)
    cv2.setMouseCallback(win_name, click_event, params)
    print_next_label()

    cv2.waitKey(0)
    cv2.destroyWindow(win_name)
    if len(selected_pts) < 4: return

    # --- トラッキング・解析セクション ---
    p0 = np.array(selected_pts, dtype=np.float32).reshape(-1, 2)
    old_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
    lk_params = dict(winSize=(25, 25), maxLevel=2, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)

    # 4点ターゲット座標
    dst_pts = np.array([[0,0], [0,BOARD_H], [BOARD_W,0], [BOARD_W,BOARD_H]], dtype=np.float32)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0.reshape(-1,1,2), None, **lk_params)
        
        if p1 is not None:
            p1 = p1.reshape(-1, 2)
            results = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            hand_pts = []
            if results.multi_hand_landmarks:
                for hl in results.multi_hand_landmarks:
                    for lm in hl.landmark:
                        hand_pts.append([lm.x * orig_w, lm.y * orig_h])

            # 信頼度判定
            reliable_idx = [i for i, pt in enumerate(p1) if not hand_pts or np.min(np.linalg.norm(np.array(hand_pts) - pt, axis=1)) > 45]

            final_p = p1.copy()
            if len(reliable_idx) >= 3:
                # 3点以上あれば数学的に補正（推定座標を算出）
                M_rel, _ = cv2.findHomography(p1[reliable_idx], dst_pts[reliable_idx], cv2.RANSAC, 3.0)
                if M_rel is not None:
                    try:
                        M_inv = np.linalg.inv(M_rel)
                        all_ideal = cv2.perspectiveTransform(dst_pts.reshape(-1, 1, 2), M_inv).reshape(-1, 2)
                        for i in range(4):
                            if i not in reliable_idx:
                                final_p[i] = all_ideal[i]
                                cv2.circle(frame, (int(final_p[i][0]), int(final_p[i][1])), 8, (255, 0, 0), 2) # 青: 推定
                            else:
                                cv2.circle(frame, (int(final_p[i][0]), int(final_p[i][1])), 5, (0, 255, 255), -1) # 黄: 実測
                    except: pass

            M_final, _ = cv2.findHomography(final_p, dst_pts, cv2.RANSAC, 2.0)
            
            # 各指の検出と表示
            if results.multi_hand_landmarks:
                for hl in results.multi_hand_landmarks:
                    for idx in [8, 12, 16, 20]: # I, M, R, P
                        lm = hl.landmark[idx]
                        px, py = int(lm.x * orig_w), int(lm.y * orig_h)
                        
                        # 行列Mを使って指の座標を指板上のフレット/弦に変換
                        pt_board = cv2.perspectiveTransform(np.array([[[px, py]]], dtype=np.float32), M_final)[0][0]
                        s, f = get_fret_and_string(pt_board[0], pt_board[1])
                        
                        if s:
                            label = f"{finger_names[idx][0]}: {s}s{f}f"
                            cv2.putText(frame, label, (px, py-10), 0, 0.5, (0, 255, 0), 2)
            
            p0 = final_p
        else:
            cv2.putText(frame, "Tracking Lost!", (50, 80), 0, 1, (0,0,255), 2)

        cv2.imshow('Analysis', frame)
        old_gray = frame_gray.copy()
        if cv2.waitKey(1) & 0xFF == ord('q'): break

    cap.release()
    cv2.destroyAllWindows()
    hands.close()

if __name__ == "__main__":
    main()