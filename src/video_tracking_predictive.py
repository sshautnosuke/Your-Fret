##このコードいいね

import cv2
import numpy as np
import mediapipe as mp
import os

# 物理定数
L_22 = 0.7184
BOARD_W, BOARD_H = 1000, 200

def get_tx_from_fret(f):
    return (1 - 2**(-f/12)) / L_22 * BOARD_W

def get_fret_and_string(tx, ty):
    relative_pos = (tx / BOARD_W) * L_22
    if 0 <= relative_pos < 1:
        fret_num = -12 * np.log2(1 - relative_pos)
        fret = int(fret_num + 0.5)
        string = 6 - int(ty / (BOARD_H / 6))
        return max(1, min(6, string)), fret
    return None, None

selected_pts = []
labels = [
    "6弦ナット (Nut)", "1弦ナット (Nut)", 
    "6弦-3フレット", "1弦-3フレット", 
    "6弦-9フレット", "1弦-9フレット", 
    "6弦-12フレット", "1弦-12フレット", 
    "6弦末端 (End)", "1弦末端 (End)"
]

def print_status():
    """現在の進捗と次の指示をターミナルに表示"""
    os.system('cls' if os.name == 'nt' else 'clear') # ターミナルを掃除して見やすく
    print("="*50)
    print("   ギター指板 10点ラベリング (予測トラッキング版)")
    print("="*50)
    print(" [操作方法]")
    print(" ・左クリック : 座標を確定")
    print(" ・右クリック : 1つ前の点を取り消す (Undo)")
    print("-"*50)
    
    for i, label in enumerate(labels):
        if i < len(selected_pts):
            print(f" [済] {label} : {selected_pts[i]}")
        elif i == len(selected_pts):
            print(f" >>> 次にクリック： 【 {label} 】 <<<")
        else:
            print(f" [未] {label}")
    print("-"*50)
    if len(selected_pts) == len(labels):
        print(" 全ての点が揃いました！画像ウィンドウで【何かキー】を押して開始。")

def click_event(event, x, y, flags, params):
    global selected_pts
    if event == cv2.EVENT_LBUTTONDOWN and len(selected_pts) < len(labels):
        selected_pts.append([x, y])
        print_status()
    elif event == cv2.EVENT_RBUTTONDOWN and len(selected_pts) > 0:
        selected_pts.pop()
        print_status()
    
    # 描画の更新
    display_img = params['img_orig'].copy()
    for i, pt in enumerate(selected_pts):
        cv2.circle(display_img, (pt[0], pt[1]), 5, (0, 0, 255), -1)
        cv2.putText(display_img, f"{i+1}:{labels[i]}", (pt[0]+10, pt[1]+10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    cv2.imshow(params['window_name'], display_img)

def main():
    video_path = 'data/Fret_test4.mp4'
    if not os.path.exists(video_path):
        print(f"Error: {video_path} not found.")
        return

    cap = cv2.VideoCapture(video_path)
    ret, first_frame = cap.read()
    if not ret: return

    orig_h, orig_w = first_frame.shape[:2]
    win_name = 'Step 1: Calibration (Right Click to Undo)'
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win_name, 1000, int(orig_h * (1000 / orig_w)))
    
    params = {'img_orig': first_frame, 'window_name': win_name}
    cv2.imshow(win_name, first_frame)
    cv2.setMouseCallback(win_name, click_event, params)
    
    print_status() # 初回表示
    cv2.waitKey(0)
    cv2.destroyWindow(win_name)

    if len(selected_pts) < 10: return

    # --- 以下、前回の予測トラッキングロジック ---
    p_prev = np.array(selected_pts, dtype=np.float32).reshape(-1, 2)
    old_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
    lk_params = dict(winSize=(20, 20), maxLevel=2, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)

    tx3, tx9, tx12 = get_tx_from_fret(3), get_tx_from_fret(9), get_tx_from_fret(12)
    dst_pts = np.array([
        [0,0],[0,BOARD_H],[tx3,0],[tx3,BOARD_H],
        [tx9,0],[tx9,BOARD_H],[tx12,0],[tx12,BOARD_H],
        [BOARD_W,0],[BOARD_W,BOARD_H]
    ], dtype=np.float32)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        p_current, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p_prev.reshape(-1,1,2), None, **lk_params)
        
        if p_current is not None:
            p_current = p_current.reshape(-1, 2)
            results = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            hand_pts = []
            if results.multi_hand_landmarks:
                for hl in results.multi_hand_landmarks:
                    for lm in hl.landmark: hand_pts.append([lm.x * orig_w, lm.y * orig_h])

            # 信頼度ベースの座標再計算
            reliable_idx = [i for i, pt in enumerate(p_current) if not hand_pts or np.min(np.linalg.norm(np.array(hand_pts) - pt, axis=1)) > 40]

            if len(reliable_idx) >= 4:
                M_rel, _ = cv2.findHomography(p_current[reliable_idx], dst_pts[reliable_idx], cv2.RANSAC, 3.0)
                M_inv = np.linalg.inv(M_rel)
                all_ideal = cv2.perspectiveTransform(dst_pts.reshape(-1, 1, 2), M_inv).reshape(-1, 2)
                
                final_p = p_current.copy()
                for i in range(len(labels)):
                    if i not in reliable_idx:
                        final_p[i] = all_ideal[i]
                        cv2.circle(frame, (int(final_p[i][0]), int(final_p[i][1])), 6, (255, 0, 0), 2)
                    else:
                        cv2.circle(frame, (int(final_p[i][0]), int(final_p[i][1])), 4, (0, 255, 255), -1)
            else:
                final_p = p_current

            M_final, _ = cv2.findHomography(final_p, dst_pts, cv2.RANSAC, 3.0)
            
            if results.multi_hand_landmarks:
                for hl in results.multi_hand_landmarks:
                    for name, idx in [("I", 8), ("M", 12), ("R", 16), ("P", 20)]:
                        lm = hl.landmark[idx]
                        px, py = int(lm.x * orig_w), int(lm.y * orig_h)
                        pt_board = cv2.perspectiveTransform(np.array([[[px, py]]], dtype=np.float32), M_final)[0][0]
                        s, f = get_fret_and_string(pt_board[0], pt_board[1])
                        if s: cv2.putText(frame, f"{s}s{f}f", (px, py-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            p_prev = final_p
        
        cv2.imshow('Predictive Analysis', frame)
        old_gray = frame_gray.copy()
        if cv2.waitKey(1) & 0xFF == ord('q'): break

    cap.release()
    cv2.destroyAllWindows()
    hands.close()

if __name__ == "__main__":
    main()