import cv2
import numpy as np
import time
import csv
from pathlib import Path

import robomaster
from robomaster import robot
from robomaster import vision


# ====== Template mask paths ======
TEMPLATE_PATHS = [r"C:\Users\snpdp\Desktop\AIE 2_1\Robot\Aj_Rattachai\Vision_and_Image_processing\Lab6_Visual_PID_Controllor\template\Screenshot 2025-09-12 174747.png",
]

# ====== Detection params ======
SCORE_THRESH = 0.30
MASK_FLOOR_Y = 600

# ====== Smoothing / EMA ======
USE_EMA = True
EMA_ALPHA = 0.35

# ====== Fallback flags ======
USE_BLOB_FALLBACK = False
MIN_BLOB_AREA = 800

# ====== Template matching method ======
MATCH_METHOD = 'CCORR'

# ====== HSV ranges (สีแดง) ======
HSV_LOWER1 = (0,   30, 40)
HSV_UPPER1 = (12, 255,255)
HSV_LOWER2 = (165, 30, 40)
HSV_UPPER2 = (180,255,255)

# ====== PID params ======
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
    masks = []
    for p in paths:
        img = cv2.imread(p, cv2.IMREAD_COLOR)
        binm = red_mask_binarize(img, floor_y=None)
        masks.append(binm)
    return masks

def _match_method_const():
    return cv2.TM_CCORR_NORMED if MATCH_METHOD.upper() == 'CCORR' else cv2.TM_CCOEFF_NORMED

def red_mask_binarize(bgr, floor_y=MASK_FLOOR_Y):
    m = bgr.copy()
    if floor_y is not None and 0 <= floor_y < m.shape[0]:
        m[floor_y:, :] = 0
    blur = cv2.GaussianBlur(m, (7,7), 0)
    hsv  = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
    # --------------------
    m1 = cv2.inRange(hsv, np.array(HSV_LOWER1), np.array(HSV_UPPER1))
    m2 = cv2.inRange(hsv, np.array(HSV_LOWER2), np.array(HSV_UPPER2))
    mask = cv2.bitwise_or(m1, m2)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  k, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=1)
    return mask

def _largest_blob_bbox(binary, area_min=MIN_BLOB_AREA):
    found = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = found[-2]
    if not cnts: return None
    c = max(cnts, key=cv2.contourArea)
    if cv2.contourArea(c) < area_min: return None
    x, y, w, h = cv2.boundingRect(c)
    return (x, y, x+w, y+h)

# ===================== Template-Matching Detector (IMPROVED WITH IMAGE PYRAMID) =====================
class RedTemplateDetector:
    def __init__(self, templates, score_thresh=SCORE_THRESH):
        if not templates:
            self.template = None
        else:
            self.template = templates[0].copy()
        self.score_thresh = float(score_thresh)
        self.last_debug = {}

    def detect(self, bgr, pyramid_scale=1.5, min_size=(40, 40)):
        if self.template is None:
            return None, 0.0, {}
            
        bi = red_mask_binarize(bgr, floor_y=MASK_FLOOR_Y)
        method = _match_method_const()
        best_match = (-1.0, None, 1.0)
        tpl_h, tpl_w = self.template.shape[:2]
        current_image = bi
        current_scale = 1.0

        while True:
            if current_image.shape[0] < tpl_h or current_image.shape[1] < tpl_w:
                break

            res = cv2.matchTemplate(current_image, self.template, method)
            _, score, _, max_loc = cv2.minMaxLoc(res)

            if score > best_match[0]:
                x1 = int(max_loc[0] * current_scale)
                y1 = int(max_loc[1] * current_scale)
                x2 = int((max_loc[0] + tpl_w) * current_scale)
                y2 = int((max_loc[1] + tpl_h) * current_scale)
                best_match = (score, (x1, y1, x2, y2), current_scale)

            if current_image.shape[0] < min_size[1] or current_image.shape[1] < min_size[0]:
                break

            new_width = int(current_image.shape[1] / pyramid_scale)
            new_height = int(current_image.shape[0] / pyramid_scale)
            if new_width > 0 and new_height > 0:
                 current_image = cv2.resize(current_image, (new_width, new_height), interpolation=cv2.INTER_AREA)
            else:
                 break
            current_scale *= pyramid_scale

        final_score, final_bbox, final_scale = best_match
        ok = (final_bbox is not None) and (final_score >= self.score_thresh)
        self.last_debug = {"ok": ok, "score": final_score, "scale": final_scale, "method": MATCH_METHOD}

        if not ok:
            return None, final_score, self.last_debug

        return final_bbox, final_score, self.last_debug

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
        self.int = clamp(self.int + e * dt, -self.int_clamp, self.int_clamp)
        u = self.kp * e + self.ki * self.int + self.kd * self.d_ema
        self.prev_e = e
        self.prev_t = t_now
        return u

# ===================== Marker SDK glue =====================
class MarkerInfo:
    def __init__(self, x, y, w, h, info):
        self._x, self._y, self._w, self._h, self._info = x, y, w, h, info
    @property
    def center(self): return int(self._x * W), int(self._y * H)
    @property
    def pt1(self): return int((self._x - self._w / 2) * W), int((self._y - self._h / 2) * H)
    @property
    def pt2(self): return int((self._x + self._w / 2) * W), int((self._y + self._h / 2) * H)
    @property
    def text(self): return self._info

markers = []
def on_detect_marker(marker_info):
    markers.clear(); markers.extend([MarkerInfo(x, y, w, h, info) for x, y, w, h, info in marker_info])
def sub_angle_cb(angle_info): gimbal_angles[:] = angle_info

def distant_by_cammara(px_x,px_y):
    if px_x == 0 or px_y == 0: return 0
    Kx, Ky = 741.176, 669.796
    zx = Kx * 5.1 / px_x; zy = Ky * 5.1 / px_y
    return (zx**2 + zy**2)**0.5

# ===================== Main =====================
if __name__ == "__main__":
    _ALL_MASKS = _load_masks_from_paths(TEMPLATE_PATHS)
    detector = RedTemplateDetector(_ALL_MASKS) if _ALL_MASKS else None

    ep = robot.Robot(); ep.initialize(conn_type="ap")
    cam, gim, ep_vision = ep.camera, ep.gimbal, ep.vision
    cam.start_video_stream(display=False)
    gim.sub_angle(freq=50, callback=sub_angle_cb)
    ep_vision.sub_detect_info(name="marker", callback=on_detect_marker)
    gim.recenter(pitch_speed=200, yaw_speed=200).wait_for_completed()

    pid_yaw, pid_pitch = PID(KP_YAW, KI_YAW, KD_YAW), PID(KP_PITCH, KI_PITCH, KD_PITCH)
    ema_cx, ema_cy = CENTER_X, CENTER_Y
    latest_bbox, latest_score, latest_src = None, 0.0, "-"
    rows, t0 = [], time.time()

    try:
        while True:
            t_now, frame = time.time(), cam.read_cv2_image(strategy="newest", timeout=0.3)
            if frame is None: time.sleep(0.01); continue

            cx, cy, bbox, score, src, dist = None, None, None, 0.0, "-", 0.0
            
            if detector:
                tbbox, tscore, _ = detector.detect(frame)
                if tbbox:
                    x1, y1, x2, y2 = tbbox
                    cx, cy, bbox, score, src = 0.5*(x1+x2), 0.5*(y1+y2), tbbox, float(tscore), "template_pyramid"
                    w_px, h_px = x2-x1, y2-y1
                    if w_px > 0 and h_px > 0: dist = distant_by_cammara(w_px, h_px)

            if cx is None and markers:
                m = markers[-1]
                cx, cy, bbox = m.center[0], m.center[1], (m.pt1[0], m.pt1[1], m.pt2[0], m.pt2[1])
                score, src = 1.0, "marker"
                w_px, h_px = bbox[2]-bbox[0], bbox[3]-bbox[1]
                if w_px > 0 and h_px > 0: dist = distant_by_cammara(w_px, h_px)

            latest_bbox, latest_score, latest_src = bbox, score, src

            if bbox and cx:
                if USE_EMA:
                    ema_cx, ema_cy = (EMA_ALPHA*cx + (1-EMA_ALPHA)*ema_cx), (EMA_ALPHA*cy + (1-EMA_ALPHA)*ema_cy)
                    tgt_x, tgt_y = ema_cx, ema_cy
                else: tgt_x, tgt_y = cx, cy
                
                err_x, err_y = (tgt_x - CENTER_X), (tgt_y - CENTER_Y)
                u_yaw = clamp(SIGN_YAW * pid_yaw.step(err_x, t_now), -MAX_SPEED, MAX_SPEED)
                u_pitch = clamp(SIGN_PITCH * pid_pitch.step(err_y, t_now), -MAX_SPEED, MAX_SPEED)
                gim.drive_speed(pitch_speed=u_pitch, yaw_speed=u_yaw)
                
                rows.append([t_now-t0, gimbal_angles[0], gimbal_angles[1], err_x, err_y, u_yaw, u_pitch, score, src])
            else:
                gim.drive_speed(pitch_speed=0, yaw_speed=0)
                pid_yaw.reset(); pid_pitch.reset()

            dbg = frame.copy()
            ema_pt = (ema_cx, ema_cy) if (bbox and USE_EMA) else None
            RedTemplateDetector.draw_debug(dbg, latest_bbox, latest_score, ema_pt, True, f"SRC: {latest_src}")

            if bbox and dist > 0:
                cv2.putText(dbg, f"Dist: {dist:.1f} cm", (int(bbox[0]), int(bbox[1])-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)
            if src == "marker":
                for m in markers:
                    cv2.rectangle(dbg, m.pt1, m.pt2, (0, 255, 0), 2)
                    cv2.putText(dbg, str(m.text), m.center, cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

            cv2.imshow("Template Pyramid PID", dbg)
            if cv2.waitKey(1) & 0xFF == ord('q'): break
    finally:
        if 'ep' in locals() and ep.is_connected():
            ep.close()
        cv2.destroyAllWindows()