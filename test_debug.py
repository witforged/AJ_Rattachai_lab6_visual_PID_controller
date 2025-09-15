import cv2
import numpy as np
import time
import csv
from pathlib import Path

import robomaster
from robomaster import robot
from robomaster import vision

# ===================== Config (ปรับได้) =====================
# ====== Template mask paths ======
TEMPLATE_PATHS = [
    r"C:\Users\snpdp\Desktop\AIE 2_1\Robot\Aj_Rattachai\Vision_and_Image_processing\Lab6_Visual_PID_Controllor\template\Screenshot 2025-09-12 174407.png",
    r"C:\Users\snpdp\Desktop\AIE 2_1\Robot\Aj_Rattachai\Vision_and_Image_processing\Lab6_Visual_PID_Controllor\template\Screenshot 2025-09-12 174708.png",
    r"C:\Users\snpdp\Desktop\AIE 2_1\Robot\Aj_Rattachai\Vision_and_Image_processing\Lab6_Visual_PID_Controllor\template\Screenshot 2025-09-12 174747.png",
    r"C:\Users\snpdp\Desktop\AIE 2_1\Robot\Aj_Rattachai\Vision_and_Image_processing\Lab6_Visual_PID_Controllor\template\Screenshot 2025-09-12 174828.png",
    r"C:\Users\snpdp\Desktop\AIE 2_1\Robot\Aj_Rattachai\Vision_and_Image_processing\Lab6_Visual_PID_Controllor\template\Screenshot 2025-09-12 174828.png",
    r"C:\Users\snpdp\Desktop\AIE 2_1\Robot\Aj_Rattachai\Vision_and_Image_processing\Lab6_Visual_PID_Controllor\template\Screenshot 2025-09-12 175017.png",
]

# ====== Detection params ======
SCALES = [0.70, 0.80, 0.90, 1.00, 1.10, 1.20, 1.30]
SCORE_THRESH = 0.35
MASK_FLOOR_Y = None

# ====== Smoothing / EMA ======
USE_EMA = True
EMA_ALPHA = 0.35

# ====== Fallback flags ======
USE_BLOB_FALLBACK = False
MIN_BLOB_AREA = 800

# ====== Template matching method ======
MATCH_METHOD = 'CCORR'
# วัตถุเป็นทรงชัดๆ ใช้ mask 0/255 → ลอง CCORR ก่อน (เสถียร)
# ถ้าเจอหลอนจากฉากสว่าง/แสงแกว่ง → สลับเป็น CCOEFF

# ====== HSV ranges (สีแดง) ======
HSV_LOWER1 = (0, 30, 40)
HSV_UPPER1 = (12, 255, 255)
HSV_LOWER2 = (165, 30, 40)
HSV_UPPER2 = (180, 255, 255)

# ====== PID params ======
KP_YAW, KI_YAW, KD_YAW = 0.18, 0.0001, 0.01
KP_PITCH, KI_PITCH, KD_PITCH = 0.18, 0.0001, 0.01
SIGN_YAW, SIGN_PITCH = +1, -1
MAX_SPEED = 250.0
INT_CLAMP = 30000.0

# ====== Misc ======
W, H = 1280, 720
CENTER_X, CENTER_Y = W / 2, H / 2

gimbal_angles = [0.0, 0.0, 0.0, 0.0]

# ===================== Utils =====================
def clamp(x, lo, hi):
    return lo if x < lo else hi if x > hi else x

def _load_masks_from_paths(paths):
    masks = []
    for p in paths:
        img = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        _, binm = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)
        masks.append(binm)
    return masks

def _match_method_const():
    return cv2.TM_CCORR_NORMED if MATCH_METHOD.upper() == 'CCORR' else cv2.TM_CCOEFF_NORMED

def red_mask_binarize(bgr, floor_y=MASK_FLOOR_Y):
    m = bgr.copy()
    if floor_y is not None and 0 <= floor_y < m.shape[0]:
        m[floor_y:, :] = 0
    blur = cv2.GaussianBlur(m, (7, 7), 0)
    hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
    m1 = cv2.inRange(hsv, np.array(HSV_LOWER1), np.array(HSV_UPPER1))
    m2 = cv2.inRange(hsv, np.array(HSV_LOWER2), np.array(HSV_UPPER2))
    mask = cv2.bitwise_or(m1, m2)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=1)
    return mask

def _largest_blob_bbox(binary, area_min=MIN_BLOB_AREA):
    found = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = found[-2]
    if not cnts:
        return None
    c = max(cnts, key=cv2.contourArea)
    if cv2.contourArea(c) < area_min:
        return None
    x, y, w, h = cv2.boundingRect(c)
    return (x, y, x + w, y + h)

# ===================== Template-Matching Detector =====================
class RedTemplateDetector:
    def __init__(self, templates, scales=SCALES, score_thresh=SCORE_THRESH):
        self.templates = [t.copy() for t in templates]
        self.scales = list(scales)
        self.score_thresh = float(score_thresh)
        self.last_debug = {}

    def detect(self, bgr, debug_view=False):
        """
        ตรวจจับวัตถุในภาพ BGR พร้อมตัวเลือกในการแสดงผล debug
        :param bgr: ภาพสี BGR
        :param debug_view: True เพื่อแสดงภาพ binary mask และภาพ template ที่ถูก resize
        """
        bi = red_mask_binarize(bgr, floor_y=MASK_FLOOR_Y)
        method = _match_method_const()
        best = (-1.0, None, None, None)

        if debug_view:
            cv2.imshow("Debug: Binarized Mask", bi)

        for t_idx, t in enumerate(self.templates):
            t_bin = t
            for s in self.scales:
                tpl = t_bin if abs(s-1.0) < 1e-6 else cv2.resize(t_bin, None, fx=s, fy=s, interpolation=cv2.INTER_NEAREST)
                if bi.shape[0] < tpl.shape[0] or bi.shape[1] < tpl.shape[1]:
                    continue
                r = cv2.matchTemplate(bi, tpl, method)
                _, score, _, max_loc = cv2.minMaxLoc(r)
                if score > best[0]:
                    x1, y1 = max_loc
                    h, w = tpl.shape
                    best = (float(score), (x1, y1, x1+w, y1+h), tpl, s)
                
                if debug_view:
                    tpl_color = cv2.cvtColor(tpl, cv2.COLOR_GRAY2BGR)
                    cv2.putText(tpl_color, f"Tpl: {t_idx} Scale: {s:.2f}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                    cv2.imshow(f"Debug: Template {t_idx} (Scale: {s:.2f})", tpl_color)

        score, bbox, tpl, sc = best
        ok = (bbox is not None) and (score >= self.score_thresh)
        self.last_debug = {"ok": ok, "score": score, "scale": sc, "method": MATCH_METHOD}
        if not ok:
            return None, float(score), self.last_debug
        return bbox, float(score), self.last_debug
    
    @staticmethod
    def draw_debug(frame, bbox, score, ema_pt, show_guides=True, label=""):
        h, w = frame.shape[:2]
        cx, cy = int(w / 2), int(h / 2)
        if show_guides:
            cv2.drawMarker(frame, (cx, cy), (255, 255, 255), markerType=cv2.MARKER_CROSS, markerSize=20, thickness=2)
        if bbox is not None:
            x1, y1, x2, y2 = map(int, bbox)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
            bb_cx, bb_cy = int(0.5 * (x1 + x2)), int(0.5 * (y1 + y2))
            cv2.circle(frame, (bb_cx, bb_cy), 4, (0, 255, 255), -1)
        if ema_pt is not None:
            ex, ey = map(int, ema_pt)
            cv2.circle(frame, (ex, ey), 4, (0, 255, 0), -1)
        if label:
            cv2.putText(frame, label, (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 255, 200), 2)
        cv2.putText(frame, f"score={score:.2f}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 255, 200), 2)

# ===================== PID =====================
class PID:
    def __init__(self, kp, ki, kd, int_clamp=INT_CLAMP):
        self.kp, self.ki, self.kd = kp, ki, kd
        self.int = 0.0
        self.prev_e = 0.0
        self.prev_t = None
        self.int_clamp = abs(int_clamp)
        self.d_ema = 0.0
        self.d_alpha = 0.5

    def reset(self):
        self.int = 0.0
        self.prev_e = 0.0
        self.prev_t = None
        self.d_ema = 0.0

    def step(self, e, t_now):
        if self.prev_t is None:
            self.prev_t = t_now
            self.prev_e = e
            return self.kp * e
        dt = max(1e-3, t_now - self.prev_t)
        de = (e - self.prev_e) / dt
        self.d_ema = self.d_alpha * de + (1 - self.d_alpha) * self.d_ema
        self.int += e * dt
        self.int = max(-self.int_clamp, min(self.int_clamp, self.int))
        u = self.kp * e + self.ki * self.int + self.kd * self.d_ema
        self.prev_e = e
        self.prev_t = t_now
        return u

# ===================== Marker SDK glue =====================
class MarkerInfo:
    def __init__(self, x, y, w, h, info):
        self._x, self._y, self._w, self._h, self._info = x, y, w, h, info
    @property
    def pt1(self):
        return int((self._x - self._w / 2) * 1280), int((self._y - self._h / 2) * 720)
    @property
    def pt2(self):
        return int((self._x + self._w / 2) * 1280), int((self._y + self._h / 2) * 720)
    @property
    def center(self):
        return int(self._x * 1280), int(self._y * 720)
    @property
    def text(self):
        return self._info

markers = []

def on_detect_marker(marker_info):
    number = len(marker_info)
    markers.clear()
    for i in range(number):
        x, y, w, h, info = marker_info[i]
        markers.append(MarkerInfo(x, y, w, h, info))

def sub_angle_cb(angle_info):
    gimbal_angles[:] = angle_info

def distant_by_cammara(px_x, px_y):
    Kx = 741.176
    Ky = 669.796
    zx = Kx * 5.1 / px_x
    zy = Ky * 5.1 / px_y
    z = (zx ** 2 + zy ** 2) ** 0.5
    return z

# ===================== Main =====================
if __name__ == "__main__":
    _ALL_MASKS = _load_masks_from_paths(TEMPLATE_PATHS)
    detector = RedTemplateDetector(_ALL_MASKS) if len(_ALL_MASKS) > 0 else None

    ep = robot.Robot()
    ep.initialize(conn_type="ap")
    cam = ep.camera
    gim = ep.gimbal
    ep_vision = ep.vision

    cam.start_video_stream(display=False)
    gim.sub_angle(freq=50, callback=sub_angle_cb)
    ep_vision.sub_detect_info(name="marker", callback=on_detect_marker)
    gim.recenter(pitch_speed=200, yaw_speed=200).wait_for_completed()

    pid_yaw = PID(KP_YAW, KI_YAW, KD_YAW)
    pid_pitch = PID(KP_PITCH, KI_PITCH, KD_PITCH)

    ema_cx, ema_cy = CENTER_X, CENTER_Y
    latest_bbox = None
    latest_score = 0.0
    latest_src = "-"

    rows = []
    t0 = time.time()

    try:
        while True:
            t_now = time.time()
            frame = cam.read_cv2_image(strategy="newest", timeout=0.3)
            if frame is None:
                time.sleep(0.01)
                continue

            cx, cy = None, None
            bbox = None
            score = 0.0
            src = "-"
            dist = 0.0

            # 1A) Primary: Template Matching (พร้อมเปิดดีบัก)
            if detector is not None:
                tbbox, tscore, _ = detector.detect(frame, debug_view=True)
                if tbbox is not None:
                    x1, y1, x2, y2 = tbbox
                    cx, cy = 0.5 * (x1 + x2), 0.5 * (y1 + y2)
                    bbox = tbbox
                    score = float(tscore)
                    src = "template"
                    w_px, h_px = x2 - x1, y2 - y1
                    if w_px > 0 and h_px > 0:
                        dist = distant_by_cammara(w_px, h_px)

            # 1B) Fallback: Marker SDK
            if cx is None and len(markers) > 0:
                x_px, y_px = markers[-1].center
                w_px = markers[-1].pt2[0] - markers[-1].pt1[0]
                h_px = markers[-1].pt2[1] - markers[-1].pt1[1]
                cx, cy = x_px, y_px
                bbox = (x_px - w_px // 2, y_px - h_px // 2, x_px + w_px // 2, y_px + h_px // 2)
                score = 1.0
                src = "marker"
                if w_px > 0 and h_px > 0:
                    dist = distant_by_cammara(w_px, h_px)

            latest_bbox = bbox
            latest_score = score
            latest_src = src

            # 2) ควบคุม PID และสั่งงาน Gimbal
            if bbox is not None and cx is not None:
                if USE_EMA:
                    ema_cx = EMA_ALPHA * cx + (1 - EMA_ALPHA) * ema_cx
                    ema_cy = EMA_ALPHA * cy + (1 - EMA_ALPHA) * ema_cy
                    tgt_x, tgt_y = ema_cx, ema_cy
                else:
                    tgt_x, tgt_y = cx, cy

                err_x = (tgt_x - CENTER_X)
                err_y = (tgt_y - CENTER_Y)
                u_yaw = SIGN_YAW * pid_yaw.step(err_x, t_now)
                u_pitch = SIGN_PITCH * pid_pitch.step(err_y, t_now)
                u_yaw = clamp(u_yaw, -MAX_SPEED, MAX_SPEED)
                u_pitch = clamp(u_pitch, -MAX_SPEED, MAX_SPEED)
                gim.drive_speed(pitch_speed=u_pitch, yaw_speed=u_yaw)
                pa, ya, _, _ = gimbal_angles
                rows.append([t_now - t0, pa, ya, err_x, err_y, u_yaw, u_pitch, latest_score, latest_src])
            else:
                gim.drive_speed(pitch_speed=0, yaw_speed=0)
                pid_yaw.reset()
                pid_pitch.reset()

            # 3) แสดงผล Debug
            dbg = frame.copy()
            ema_pt = (ema_cx, ema_cy) if (bbox is not None and USE_EMA) else None
            RedTemplateDetector.draw_debug(dbg, latest_bbox, latest_score, ema_pt, True, f"SRC: {latest_src}")
            if bbox is not None and dist > 0:
                dist_text = f"Distance: {dist:.2f} cm"
                text_pos = (int(bbox[0]), int(bbox[1]) - 10)
                cv2.putText(dbg, dist_text, text_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

            if latest_src == "marker":
                for m in markers:
                    cv2.rectangle(dbg, m.pt1, m.pt2, (0, 255, 0), 2)
                    cv2.putText(dbg, str(m.text), m.center, cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

            cv2.imshow("Template-First PID", dbg)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        try:
            ep_vision.unsub_detect_info(name="marker")
        except Exception:
            pass
        cv2.destroyAllWindows()
        try:
            cam.stop_video_stream()
        except Exception:
            pass
        gim.drive_speed(pitch_speed=0, yaw_speed=0)
        try:
            gim.unsub_angle()
        except Exception:
            pass
        ep.close()