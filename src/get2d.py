import cv2

# 調べたい画像のパスを指定
IMAGE_PATH = r'C:\Users\sshau\Documents\05_Web_App\Your_Fret\data\test.jpg'

def click_event(event, x, y, flags, params):
    # 左クリックした時に座標を表示
    if event == cv2.EVENT_LBUTTONDOWN:
        print(f"座標を取得しました: [{x}, {y}]")
        
        # 画像上に印を付ける（確認用）
        cv2.circle(img, (x, y), 5, (0, 0, 255), -1)
        cv2.imshow('Coordinate Picker', img)

# 画像の読み込み
img = cv2.imread(IMAGE_PATH)
if img is None:
    print(f"エラー: {IMAGE_PATH} が見つかりません。パスを確認してください。")
    exit()

print("--- 座標取得ツール ---")
print("1. ナット側（6弦→1弦）")
print("2. 末端側（6弦→1弦）")
print("の順でクリックするとスムーズです。")
print("終了するには 'q' キーを押してください。")

cv2.imshow('Coordinate Picker', img)
cv2.setMouseCallback('Coordinate Picker', click_event)

# 'q'が押されるまで待機
while True:
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()