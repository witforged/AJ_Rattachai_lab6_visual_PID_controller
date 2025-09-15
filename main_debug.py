import cv2
import numpy as np
import time
import csv
from pathlib import Path

import robomaster
from robomaster import robot
from robomaster import vision

# ==================== [โหมดดีบัก] ====================
# ตั้งเป็น True เพื่อแสดงหน้าต่าง Mask สีแดง และพิมพ์ค่า score ใน console
DEBUG_MODE = True
# =====================================================


# ==================== [ค่าสำหรับปรับจูน] ====================

# ------ 1. Template ------
# แนะนำให้เริ่มจาก Template ที่ดีที่สุดเพียง 1 ไฟล์ก่อน เพื่อลดความซับซ้อน
TEMPLATE_PATHS = [
    r"template\image.png",  # <-- แก้เป็นชื่อไฟล์ template ของคุณ
]

# ------ 2. Threshold ------
# เกณฑ์รับผลจับคู่ (0.0 - 1.0)
# ค่าที่สูงไป (เช่น 0.6) จะทำให้ไม่เจออะไรเลย, ค่าที่ต่ำไปจะทำให้เจอมั่ว
# **แนะนำให้เริ่มที่ 0.35 - 0.45**
SCORE_THRESH = 0.62

# ------ 3. การกรองสี (HSV Filter) ------
# ปรับค่าเหล่านี้โดยดูจากหน้าต่าง "Red Mask (DEBUG)"
# เป้าหมายคือให้วัตถุเป็นสีขาว และพื้นหลังเป็นสีดำสนิทที่สุด
HSV_LOWER1 = (0, 30, 40)
HSV_UPPER1 = (12, 255, 255)
HSV_LOWER2 = (165, 30, 40)
HSV_UPPER2 = (180, 255, 255)

# ------ 4. การลด Noise ใน Mask ------
# เพิ่มค่าเพื่อลด Noise ที่เป็นจุดๆ ในภาพ Mask
BLUR_KERNEL_SIZE = (3, 3)   # ขนาดเบลอ (เลขคี่) e.g., (9, 9), (11, 11)
MORPH_KERNEL_SIZE = (7, 7)     # ขนาดตัวกรอง Noise (เลขคี่) e.g., (5, 5), (7, 7)
MORPH_ITERATIONS = 2           # จำนวนรอบที่กรอง Noise (เพิ่มเพื่อความสะอาด)

# ------ 5. การติดตาม (Tracking) ------
USE_EMA = True
EMA_ALPHA = 0.35 # ค่าความหน่วง (น้อย = หน่วงมาก, มาก = ตอบสนองเร็ว)

# ===========================================================


# ====== การตั้งค่าระบบ (System Configuration) ======
SCALES = [
    0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.00,
    1.05, 1.10, 1.15, 1.20, 1.25, 1.30, 1.35, 1.40
]
MATCH_METHOD = cv2.TM_CCOEFF_NORMED
MASK_FLOOR_Y = 600

# ====== PID ======
KP_YAW,   KI_YAW,   KD_YAW   = 0.18, 0.0001, 0.01
KP_PITCH, KI_PITCH, KD_PITCH = 0.18, 0.0001, 0.01
SIGN_YAW, SIGN_PITCH = +1, -1
MAX_SPEED = 250.0
INT_CLAMP = 30000.0

# ====== Misc ======
W, H = 1280, 720
CENTER_X, CENTER_Y = W/2, H/2
gimbal_angles = [0.0, 0.0, 0.0, 0.0]

# ===================== Utils =====================
def clamp(x, lo, hi):
    return lo if x < lo else hi if x > hi else x

def _load_masks_from_paths(paths):
    """
    โหลดไฟล์ template และ crop ให้เหลือเฉพาะส่วนของวัตถุเพื่อความแม่นยำ
    """
    masks = []
    for p in paths:
        img = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"[WARN] Failed to load template file: {p}")
            continue
        
        _, thresh = cv2.threshold(img, 1, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(largest_contour)
            cropped_mask = img[y:y+h, x:x+w]
        else:
            cropped_mask = img
        
        _, binm = cv2.threshold(cropped_mask, 127, 255, cv2.THRESH_BINARY)
        masks.append(binm)
    return masks

def red_mask_binarize(bgr, floor_y=MASK_FLOOR_Y):
    """
    ฟังก์ชันสร้าง Mask สีแดงที่ปรับปรุงแล้วเพื่อลด Noise และเพิ่มความแม่นยำ
    """
    m = bgr.copy()
    if floor_y is not None and 0 <= floor_y < m.shape[0]:
        m[floor_y:, :] = 0
    
    # 1. เบลอภาพเพื่อลด Noise เล็กๆ น้อยๆ
    blur = cv2.GaussianBlur(m, BLUR_KERNEL_SIZE, 0)
    
    # 2. แปลงเป็น HSV และกรองเฉพาะช่วงสีแดง
    hsv  = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
    m1 = cv2.inRange(hsv, np.array(HSV_LOWER1), np.array(HSV_UPPER1))
    m2 = cv2.inRange(hsv, np.array(HSV_LOWER2), np.array(HSV_UPPER2))
    mask = cv2.bitwise_or(m1, m2)
    
    # 3. ใช้ Morphology เพื่อกำจัด Noise ที่เป็นจุดๆ และเติมเต็มรูปร่าง
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, MORPH_KERNEL_SIZE)
    # MORPH_OPEN: กำจัดจุดขาวๆ เล็กๆ ที่เป็น Noise ด้านนอก
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  k, iterations=MORPH_ITERATIONS)
    # MORPH_CLOSE: ปิดรูโหว่สีดำเล็กๆ ที่อยู่ในวัตถุสีขาว
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=MORPH_ITERATIONS)
    
    return mask

# ===================== Template-Matching Detector =====================
class RedTemplateDetector:
    def __init__(self, templates, scales=SCALES, score_thresh=SCORE_THRESH):
        self.templates = [t.copy() for t in templates]
        self.scales = list(scales)
        self.score_thresh = float(score_thresh)
        self.last_debug = {}

    def detect(self, bgr):
        bi = red_mask_binarize(bgr, floor_y=MASK_FLOOR_Y)
        best = (-1.0, None, None, None)

        for t in self.templates:
            for s in self.scales:
                tpl = t if abs(s-1.0) < 1e-6 else cv2.resize(t, None, fx=s, fy=s, interpolation=cv2.INTER_NEAREST)
                if bi.shape[0] < tpl.shape[0] or bi.shape[1] < tpl.shape[1]:
                    continue
                
                r = cv2.matchTemplate(bi, tpl, MATCH_METHOD)
                _, score, _, max_loc = cv2.minMaxLoc(r)
                
                if score > best[0]:
                    x1, y1 = max_loc
                    h, w = tpl.shape
                    best = (float(score), (x1, y1, x1+w, y1+h), tpl, s)

        score, bbox, tpl, sc = best
        ok = (bbox is not None) and (score >= self.score_thresh)
        self.last_debug = {"ok": ok, "score": score, "scale": sc}
        
        if DEBUG_MODE:
            return bbox if ok else None, float(score), self.last_debug, bi
        
        return bbox if ok else None, float(score), self.last_debug
    
    @staticmethod
    def draw_debug(frame, bbox, score, ema_pt, show_guides=True, label=""):
        h, w = frame.shape[:2]
        cx, cy = int(w/2), int(h/2)
        if show_guides:
            cv2.drawMarker(frame, (cx, cy), (255, 255, 255), markerType=cv2.MARKER_CROSS, markerSize=20, thickness=2)
        if bbox is not None:
            x1, y1, x2, y2 = map(int, bbox)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
            bb_cx, bb_cy = int(0.5*(x1+x2)), int(0.5*(y1+y2))
            cv2.circle(frame, (bb_cx, bb_cy), 4, (0, 255, 255), -1)
        if ema_pt is not None:
            ex, ey = map(int, ema_pt)
            cv2.circle(frame, (ex, ey), 4, (0, 255, 0), -1)
        if label:
            cv2.putText(frame, label, (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200,255,200), 2)
        cv2.putText(frame, f"score={score:.2f}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200,255,200), 2)

# ===================== PID =====================
class PID:
    def __init__(self, kp, ki, kd, int_clamp=INT_CLAMP):
        self.kp, self.ki, self.kd = kp, ki, kd
        self.int, self.prev_e, self.prev_t, self.d_ema = 0.0, 0.0, None, 0.0
        self.int_clamp, self.d_alpha = abs(int_clamp), 0.5

    def reset(self):
        self.int, self.prev_e, self.prev_t, self.d_ema = 0.0, 0.0, None, 0.0

    def step(self, e, t_now):
        if self.prev_t is None:
            self.prev_t, self.prev_e = t_now, e
            return self.kp * e
        dt = max(1e-3, t_now - self.prev_t)
        de = (e - self.prev_e) / dt
        self.d_ema = self.d_alpha * de + (1 - self.d_alpha) * self.d_ema
        self.int += e * dt
        self.int = clamp(self.int, -self.int_clamp, self.int_clamp)
        u = self.kp * e + self.ki * self.int + self.kd * self.d_ema
        self.prev_e, self.prev_t = e, t_now
        return u

# ===================== Marker SDK glue =====================
class MarkerInfo:
    def __init__(self, x, y, w, h, info):
        self._x, self._y, self._w, self._h, self._info = x, y, w, h, info
    @property
    def pt1(self): return int((self._x - self._w / 2) * W), int((self._y - self._h / 2) * H)
    @property
    def pt2(self): return int((self._x + self._w / 2) * W), int((self._y + self._h / 2) * H)
    @property
    def center(self): return int(self._x * W), int(self._y * H)
    @property
    def text(self): return self._info

markers = []
def on_detect_marker(marker_info):
    markers.clear()
    for x, y, w, h, info in marker_info:
        markers.append(MarkerInfo(x, y, w, h, info))

def sub_angle_cb(angle_info):
    gimbal_angles[:] = angle_info

# ===================== Main =====================
if __name__ == "__main__":
    _ALL_MASKS = _load_masks_from_paths(TEMPLATE_PATHS)
    if not _ALL_MASKS:
        print("[ERROR] No valid templates loaded. Exiting.")
        exit()
        
    detector = RedTemplateDetector(_ALL_MASKS)

    ep = robot.Robot()
    ep.initialize(conn_type="ap")

    cam, gim, ep_vision = ep.camera, ep.gimbal, ep.vision
    
    cam.start_video_stream(display=False)
    gim.sub_angle(freq=50, callback=sub_angle_cb)
    ep_vision.sub_detect_info(name="marker", callback=on_detect_marker)
    gim.recenter(pitch_speed=200, yaw_speed=200).wait_for_completed()

    pid_yaw = PID(KP_YAW, KI_YAW, KD_YAW)
    pid_pitch = PID(KP_PITCH, KI_PITCH, KD_PITCH)

    ema_cx, ema_cy = CENTER_X, CENTER_Y
    latest_bbox, latest_score, latest_src = None, 0.0, "-"
    
    rows, t0 = [], time.time()

    try:
        while True:
            t_now = time.time()
            frame = cam.read_cv2_image(strategy="newest", timeout=0.3)
            if frame is None:
                time.sleep(0.01)
                continue

            cx, cy, bbox, score, src = None, None, None, 0.0, "-"
            binary_mask_for_debug = None

            # Priority 1: Robomaster's built-in markers
            if len(markers) > 0:
                m = markers[-1]
                cx, cy = m.center
                x1, y1 = m.pt1
                x2, y2 = m.pt2
                bbox = (x1, y1, x2, y2)
                score, src = 1.0, "marker"
            # Priority 2: Our custom template detector
            elif detector is not None:
                if DEBUG_MODE:
                    tbbox, tscore, _, binary_mask_for_debug = detector.detect(frame)
                else:
                    tbbox, tscore, _ = detector.detect(frame)

                if tbbox is not None:
                    x1, y1, x2, y2 = tbbox
                    cx, cy = 0.5 * (x1 + x2), 0.5 * (y1 + y2)
                    bbox, score, src = tbbox, float(tscore), "template"
                
                if DEBUG_MODE:
                    # พิมพ์ค่า score สูงสุดที่เจอใน console ตลอดเวลา
                    print(f"Best Match Score: {tscore:.4f} | Threshold: {SCORE_THRESH}")

            latest_bbox, latest_score, latest_src = bbox, score, src

            if bbox is not None:
                if USE_EMA:
                    ema_cx = EMA_ALPHA * cx + (1 - EMA_ALPHA) * ema_cx
                    ema_cy = EMA_ALPHA * cy + (1 - EMA_ALPHA) * ema_cy
                    tgt_x, tgt_y = ema_cx, ema_cy
                else:
                    tgt_x, tgt_y = cx, cy

                err_x, err_y = (tgt_x - CENTER_X), (tgt_y - CENTER_Y)
                u_yaw = clamp(SIGN_YAW * pid_yaw.step(err_x, t_now), -MAX_SPEED, MAX_SPEED)
                u_pitch = clamp(SIGN_PITCH * pid_pitch.step(err_y, t_now), -MAX_SPEED, MAX_SPEED)
                gim.drive_speed(pitch_speed=u_pitch, yaw_speed=u_yaw)
                
                pa, ya, _, _ = gimbal_angles
                rows.append([t_now - t0, pa, ya, err_x, err_y, u_yaw, u_pitch, score, src])
            else:
                gim.drive_speed(pitch_speed=0, yaw_speed=0)
                pid_yaw.reset()
                pid_pitch.reset()

            # แสดงผล Debug
            dbg_frame = frame.copy()
            ema_pt = (ema_cx, ema_cy) if (bbox and USE_EMA) else None
            
            if src == "marker":
                for m in markers:
                    cv2.rectangle(dbg_frame, m.pt1, m.pt2, (0, 255, 0), 2)
                    cv2.putText(dbg_frame, str(m.text), m.center, cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
            
            RedTemplateDetector.draw_debug(dbg_frame, latest_bbox, latest_score, ema_pt, True, f"SRC: {latest_src}")
            cv2.imshow("Robomaster Tracker", dbg_frame)

            if DEBUG_MODE and binary_mask_for_debug is not None:
                cv2.imshow("Red Mask (DEBUG)", binary_mask_for_debug)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        print("\n[INFO] Stopping program...")
        ep_vision.unsub_detect_info(name="marker")
        cv2.destroyAllWindows()
        cam.stop_video_stream()
        gim.drive_speed(pitch_speed=0, yaw_speed=0)
        time.sleep(0.5) # ให้เวลา gimbal หยุดสนิท
        gim.unsub_angle()
        ep.close()

        if rows:
            out = Path("gimbal_response.csv")
            try:
                with out.open('w', newline='') as f:
                    w = csv.writer(f)
                    w.writerow(["t","pitch_angle","yaw_angle","err_x_px","err_y_px","u_yaw_dps","u_pitch_dps","score","src"])
                    w.writerows(rows)
                print(f"[LOG] Saved {len(rows)} samples -> {out.resolve()}")
            except Exception as e:
                print(f"[WARN] CSV not saved: {e}")