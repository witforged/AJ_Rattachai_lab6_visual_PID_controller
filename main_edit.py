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
    r"template/near.png",
    r"template/mid.png",
    r"template/far.png",
]

# ====== Detection params ======
SCALES = [0.70, 0.80, 0.90, 1.00, 1.10, 1.20, 1.30]   # ลองหลาย scale ของ template เพื่อรับมือระยะใกล้/ไกล
SCORE_THRESH = 0.35                                    # เกณฑ์รับผลจับคู่จาก template (ยิ่งสูงยิ่งเข้มงวด)
MASK_FLOOR_Y = 600                                     # ปิดส่วนล่างของภาพ (พรม/พื้น) เพื่อลดสัญญาณรบกวน

# ====== Smoothing / EMA ======
USE_EMA = True                                         # เปิดใช้ EMA ให้ตำแหน่งเป้านิ่งขึ้น
EMA_ALPHA = 0.35                                       # ค่าน้ำหนัก EMA (สูง = ตามไวขึ้น แต่นิ่งน้อยลง)

# ====== Fallback flags ======
USE_BLOB_FALLBACK = True                               # ถ้าไม่เจอทั้ง marker/template ให้ลองหาก้อนสีแดงที่ใหญ่สุด
MIN_BLOB_AREA = 800                                    # พื้นที่ขั้นต่ำของ blob ที่จะยอมรับ

# ====== Template matching method ======
MATCH_METHOD = 'CCORR'  # 'CCORR' or 'CCOEFF'
# วัตถุเป็นทรงชัดๆ ใช้ mask 0/255 → ลอง CCORR ก่อน (เสถียร)
# ถ้าเจอหลอนจากฉากสว่าง/แสงแกว่ง → สลับเป็น CCOEFF

# ====== HSV ranges (สีแดง) ======
HSV_LOWER1 = (0,   30, 40)
HSV_UPPER1 = (12, 255,255)
HSV_LOWER2 = (165, 30, 40)
HSV_UPPER2 = (180,255,255)

# ====== PID params ======
KP_YAW,   KI_YAW,   KD_YAW   = 0.18, 0.0001, 0.01     # yaw แก้ไขค่า PID
KP_PITCH, KI_PITCH, KD_PITCH = 0.18, 0.0001, 0.01     # pitch แก้ไขค่า PID
SIGN_YAW, SIGN_PITCH = +1, -1                          # ทิศสัญญาณให้ match กับ ep_gimbal.drive_speed 
MAX_SPEED = 250.0                                      # deg/s จำกัดความเร็ว gimbal
INT_CLAMP = 30000.0                                    # anti-windup จำกัดอินทิกรัล

# ====== Misc ======
W, H = 1280, 720                                       # ขนาดเฟรม
CENTER_X, CENTER_Y = W/2, H/2                          # จุดกึ่งกลางภาพ (พิกเซล)

# มุมกิมบอล (จาก callback)  pitch, yaw, pitch_ground, yaw_ground
gimbal_angles = [0.0, 0.0, 0.0, 0.0]

# ===================== Utils =====================
def clamp(x, lo, hi):
    return lo if x < lo else hi if x > hi else x

def _load_masks_from_paths(paths): # นำเข้า template masks
    masks = []
    for p in paths:
        img = cv2.imread(p, cv2.IMREAD_GRAYSCALE) # อ่านภาพเป็นขาวดำ
        # ให้เป็น binary 0/255
        _, binm = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
        masks.append(binm)
    return masks

def _match_method_const(): # คืนค่า flag ให้ cv2.matchTemplate
    return cv2.TM_CCORR_NORMED if MATCH_METHOD.upper() == 'CCORR' else cv2.TM_CCOEFF_NORMED

def red_mask_binarize(bgr, floor_y=MASK_FLOOR_Y):
    """
    สร้าง binary mask โทนแดงจากภาพ BGR:
      1) ปิดท่อนล่าง (ลดผลจากพื้น)
      2) Blur ลด noise
      3) แปลงเป็น HSV + คัดช่วงแดง 2 ช่วง
      4) เปิด-ปิด morphology ให้ก้อนเรียบขึ้น
    """
    m = bgr.copy()
    if floor_y is not None and 0 <= floor_y < m.shape[0]: # ถ้ากำหนด floor_y → ปิดท่อนล่าง
        m[floor_y:, :] = 0
    blur = cv2.GaussianBlur(m, (7,7), 0) # เบลอเพื่อลด noise
    hsv  = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV) # BGR -> HSV
    m1 = cv2.inRange(hsv, np.array(HSV_LOWER1), np.array(HSV_UPPER1)) # mask แดงช่วงที่ 1
    m2 = cv2.inRange(hsv, np.array(HSV_LOWER2), np.array(HSV_UPPER2)) # mask แดงช่วงที่ 2
    mask = cv2.bitwise_or(m1, m2) # รวมสองช่วง
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5)) # kernel ทรงวงรี
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  k, iterations=1) # เปิด = ลบ noise เล็กๆ
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=1) # ปิด = เติมรูพรุนเล็กๆ
    return mask

def _largest_blob_bbox(binary, area_min=MIN_BLOB_AREA): # หา bbox ของก้อนแดงใหญ่สุด (เป็น fallback ที่เสี่ยงเจอสิ่งอื่นสีแดง)
    found = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = found[-2]
    if not cnts: return None
    c = max(cnts, key=cv2.contourArea)
    if cv2.contourArea(c) < area_min: return None
    x, y, w, h = cv2.boundingRect(c)
    return (x, y, x+w, y+h)

# ===================== Template-Matching Detector =====================
class RedTemplateDetector: # ตรวจจับวัตถุสีแดงโดยใช้ template matching
    # เก็บพารามิเตอร์สำหรับวนหาคู่แมตช์
    def __init__(self, templates, scales=SCALES, score_thresh=SCORE_THRESH): # กำหนด template, scales และ score_thresh
        self.templates = [t.copy() for t in templates] # ทำสำเนาของ template เพื่อป้องกันการเปลี่ยนแปลงต้นฉบับ
        self.scales = list(scales)                     # กำหนดสเกลที่ใช้ในการตรวจจับ
        self.score_thresh = float(score_thresh)        # เกณฑ์คะแนนขั้นต่ำสำหรับการยอมรับการตรวจจับ
        self.last_debug = {}                           # เก็บข้อมูลสำหรับการ Debug

    # ตรวจจับวัตถุในภาพ BGR
    def detect(self, bgr):
        bi = red_mask_binarize(bgr, floor_y=MASK_FLOOR_Y) # สร้าง binary image โดยใช้ mask สีแดง
        method = _match_method_const()                     # เลือกวิธีการจับคู่ template (CCORR หรือ CCOEFF)
        best = (-1.0, None, None, None)  # score เริ่มจาก -1 ต่ำสุด, bbox, tpl_used, scale

        # วนหาคู่แมตช์ที่ดีที่สุด
        for t in self.templates:
            t_bin = t
            for s in self.scales:
                # ปรับขนาด template ตามสเกล (nearest เพื่อคงบิต 0/255 ชัดๆ)
                tpl = t_bin if abs(s-1.0) < 1e-6 else cv2.resize(t_bin, None, fx=s, fy=s, interpolation=cv2.INTER_NEAREST)
                # ข้ามถ้า template ใหญ่กว่า binary image
                if bi.shape[0] < tpl.shape[0] or bi.shape[1] < tpl.shape[1]:
                    continue
                r = cv2.matchTemplate(bi, tpl, method)     # ทำการจับคู่ template กับ binary image
                _, score, _, max_loc = cv2.minMaxLoc(r)    # เอาค่าสูงสุด (ยิ่งสูงยิ่งเหมือน)
                # อัปเดต best ถ้าคะแนนดีกว่า เพื่อเก็บค่าที่ดีที่สุด
                if score > best[0]:
                    x1, y1 = max_loc                       # ตำแหน่งซ้ายบนของการจับคู่ที่ดีที่สุด
                    h, w = tpl.shape                       # ขนาดของ template ที่ใช้
                    best = (float(score), (x1, y1, x1+w, y1+h), tpl, s)

        score, bbox, tpl, sc = best
        ok = (bbox is not None) and (score >= self.score_thresh)
        self.last_debug = {"ok": ok, "score": score, "scale": sc, "method": MATCH_METHOD}
        if not ok:
            return None, float(score), self.last_debug
        return bbox, float(score), self.last_debug
    
    @staticmethod
    def draw_debug(frame, bbox, score, ema_pt, show_guides=True, label=""):
        """
        วาด crosshair กลางภาพ + bbox + จุด EMA (เขียว) + ข้อความ debug
        """
        h, w = frame.shape[:2]
        cx, cy = int(w/2), int(h/2)
        if show_guides:
            cv2.drawMarker(frame, (cx, cy), (255, 255, 255), markerType=cv2.MARKER_CROSS, markerSize=20, thickness=2) # จุดกึ่งกลางจอ
        if bbox is not None:
            x1, y1, x2, y2 = map(int, bbox)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2) # กรอบเป้า (เหลือง)
            bb_cx, bb_cy = int(0.5*(x1+x2)), int(0.5*(y1+y2))
            cv2.circle(frame, (bb_cx, bb_cy), 4, (0, 255, 255), -1)    # จุดกึ่งกลางกรอบ
        if ema_pt is not None:
            ex, ey = map(int, ema_pt)
            cv2.circle(frame, (ex, ey), 4, (0, 255, 0), -1)            # จุด EMA (เขียว)
        if label:
            cv2.putText(frame, label, (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200,255,200), 2)
        cv2.putText(frame, f"score={score:.2f}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200,255,200), 2)

# ===================== PID =====================
class PID:
    def __init__(self, kp, ki, kd, int_clamp=INT_CLAMP):
        self.kp, self.ki, self.kd = kp, ki, kd
        self.int = 0.0                 # สะสมอินทิกรัล
        self.prev_e = 0.0              # error ก่อนหน้า
        self.prev_t = None             # เวลาอ่านก่อนหน้า
        self.int_clamp = abs(int_clamp)# ขอบเขต anti-windup
        self.d_ema = 0.0               # ค่า derivative ที่กรองด้วย EMA
        self.d_alpha = 0.5             # น้ำหนัก EMA สำหรับอนุพันธ์

    def reset(self):                   # รีเซ็ตสถานะ PID เมื่อเป้า “หาย”
        self.int = 0.0
        self.prev_e = 0.0
        self.prev_t = None
        self.d_ema = 0.0

    def step(self, e, t_now):
        """
        คำนวณ u = Kp*e + Ki*∫e dt + Kd*d(e)/dt (แต่ d(e)/dt ผ่าน EMA เพื่อลด noise)
        """
        if self.prev_t is None:        # รอบแรก → ยังไม่มี dt ใช้ P อย่างเดียว
            self.prev_t = t_now
            self.prev_e = e
            return self.kp * e
        dt = max(1e-3, t_now - self.prev_t)  # กัน dt == 0
        de = (e - self.prev_e) / dt
        self.d_ema = self.d_alpha * de + (1 - self.d_alpha) * self.d_ema  # derivative smoothing
        self.int += e * dt
        # anti-windup
        self.int = max(-self.int_clamp, min(self.int_clamp, self.int))
        # PID output
        u = self.kp * e + self.ki * self.int + self.kd * self.d_ema
        # เก็บสถานะไว้ใช้รอบถัดไป
        self.prev_e = e
        self.prev_t = t_now
        return u

# ===================== Marker SDK glue =====================
class MarkerInfo:
    """
    โครงข้อมูล marker จาก SDK (x,y,w,h ∈ [0,1] อิงขนาดภาพ), พร้อม helper ให้เป็นพิกเซล
    """
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

markers = []  # เก็บ marker ล่าสุดจาก SDK

def on_detect_marker(marker_info):
    """
    callback จาก ep_vision: รับ list ของ marker (x,y,w,h,info) → แปลงเก็บเป็น MarkerInfo
    """
    number = len(marker_info)
    markers.clear()
    for i in range(number):
        x, y, w, h, info = marker_info[i]
        markers.append(MarkerInfo(x, y, w, h, info))  # x,y,w,h ∈ [0,1]

def sub_angle_cb(angle_info):
    """
    callback มุมกิมบอล: pitch, yaw, pitch_ground, yaw_ground
    เก็บลง global เพื่อ log
    """
    gimbal_angles[:] = angle_info

def distant_by_cammara(px_x,px_y):
    Kx = 741.176
    Ky = 669.796
    zx = Kx*5.1/px_x
    zy = Ky*5.1/px_y
    z = (zx**2 + zy**2)**0.5
    return z

# ===================== Main =====================
if __name__ == "__main__":
    # โหลด template (ถ้ามีไฟล์)
    _ALL_MASKS = _load_masks_from_paths(TEMPLATE_PATHS)
    detector = RedTemplateDetector(_ALL_MASKS) if len(_ALL_MASKS) > 0 else None

    ep = robot.Robot()
    ep.initialize(conn_type="ap")

    cam = ep.camera
    gim = ep.gimbal
    ep_vision = ep.vision      

    # เริ่มสตรีม + subscribe callbacks
    cam.start_video_stream(display=False)                         # เปิดวิดีโอ (ไม่ต้องโชว์หน้าต่างจาก SDK)
    gim.sub_angle(freq=50, callback=sub_angle_cb)                 # รับค่ามุมกิมบอลที่ 50Hz
    ep_vision.sub_detect_info(name="marker", callback=on_detect_marker) # เปิดตรวจจับ marker โดย SDK
    gim.recenter(pitch_speed=200, yaw_speed=200).wait_for_completed()   # รีเซ็นเตอร์ก่อนเริ่ม

    # สร้าง PID สองแกน
    pid_yaw = PID(KP_YAW, KI_YAW, KD_YAW)
    pid_pitch = PID(KP_PITCH, KI_PITCH, KD_PITCH)

    # ตัวแปรสถานะสำหรับ EMA และ Debug
    ema_cx, ema_cy = CENTER_X, CENTER_Y
    latest_bbox = None
    latest_score = 0.0
    latest_src = "-"   # "marker" | "template" | "blob"

    # เตรียม log CSV
    rows = []
    t0 = time.time()

    try:
        # ==================================
        # ======     Main Loop      ======
        # ==================================
        while True:
            t_now = time.time()
            frame = cam.read_cv2_image(strategy="newest", timeout=0.3) # ดึงเฟรมล่าสุดจากกล้อง
            if frame is None:
                time.sleep(0.01)
                continue

            # 1) ตรวจจับเป้า: ใช้ Marker SDK เป็นหลัก → template → blob
            cx, cy = None, None
            bbox = None
            score = 0.0
            src = "-"

            # 1A) Primary: Marker SDK
            if len(markers) > 0:  # เจอ marker อย่างน้อย 1 อัน
                x_px, y_px = markers[-1].center  # เลือก marker ล่าสุด (จะปรับนโยบายเลือกทีหลังได้)
                w_px = markers[-1].pt2[0] - markers[-1].pt1[0]
                h_px = markers[-1].pt2[1] - markers[-1].pt1[1]
                cx, cy = x_px, y_px
                bbox = (x_px - w_px//2, y_px - h_px//2, x_px + w_px//2, y_px + h_px//2)
                score = 1.0  # จาก SDK ถือว่ามั่นใจ
                src = "marker"

            # 1B) Fallback: Template Matching
            if cx is None and detector is not None:
                tbbox, tscore, _ = detector.detect(frame)
                if tbbox is not None:
                    x1, y1, x2, y2 = tbbox
                    cx, cy = 0.5*(x1+x2), 0.5*(y1+y2)
                    bbox = tbbox
                    score = float(tscore)
                    src = "template"

            # 1C) Fallback: Largest Red Blob
            if cx is None and USE_BLOB_FALLBACK:
                bi = red_mask_binarize(frame, floor_y=MASK_FLOOR_Y)
                fb = _largest_blob_bbox(bi, area_min=MIN_BLOB_AREA)
                if fb is not None:
                    x1, y1, x2, y2 = fb
                    cx, cy = 0.5*(x1+x2), 0.5*(y1+y2)
                    bbox = fb
                    score = 0.50  # ให้คะแนนกลางๆ
                    src = "blob"

            latest_bbox = bbox
            latest_score = score
            latest_src = src

            # 2) ควบคุม PID และสั่งงาน Gimbal
            if bbox is not None and cx is not None:
                # ใช้ EMA เพื่อให้เป้านิ่งขึ้น
                if USE_EMA:
                    ema_cx = EMA_ALPHA * cx + (1 - EMA_ALPHA) * ema_cx
                    ema_cy = EMA_ALPHA * cy + (1 - EMA_ALPHA) * ema_cy
                    tgt_x, tgt_y = ema_cx, ema_cy
                else:
                    tgt_x, tgt_y = cx, cy

                # error พิกเซล (ขวา/ล่าง = บวก)
                err_x = (tgt_x - CENTER_X)  # เป้าอยู่ขวา → err_x > 0 → yaw ขวา
                err_y = (tgt_y - CENTER_Y)  # เป้าอยู่ล่าง → err_y > 0 → pitch ลง

                # PID สองแกน + กลับทิศตาม SIGN_*
                u_yaw   = SIGN_YAW   * pid_yaw.step(err_x, t_now)
                u_pitch = SIGN_PITCH * pid_pitch.step(err_y, t_now)

                # จำกัดความเร็ว gimbal
                u_yaw   = clamp(u_yaw,   -MAX_SPEED, MAX_SPEED)
                u_pitch = clamp(u_pitch, -MAX_SPEED, MAX_SPEED)

                # สั่งกิมบอล (รูปแบบเดียวกับตัวอย่าง: pitch_speed, yaw_speed)
                gim.drive_speed(pitch_speed=u_pitch, yaw_speed=u_yaw)

                # อยากยิงอินฟราเรดใส่เป้า? เปิดบรรทัดล่างนี้ (และ import blaster)
                # ep_blaster.fire(fire_type=blaster.INFRARED_FIRE)

                # log (t, มุมกิมบอล, error, สัญญาณสั่ง, score, source)
                pa, ya, _, _ = gimbal_angles
                rows.append([t_now - t0, pa, ya, err_x, err_y, u_yaw, u_pitch, latest_score, latest_src])
            else:
                # ไม่เจอเป้า: หยุดนิ่งและรีเซ็ต PID (กันสะสมค่าอินทิกรัล)
                gim.drive_speed(pitch_speed=0, yaw_speed=0)
                pid_yaw.reset()
                pid_pitch.reset()

            # 3) แสดงผล Debug
            dbg = frame.copy()
            ema_pt = (ema_cx, ema_cy) if (bbox is not None and USE_EMA) else None
            if latest_src == "marker":
                # วาดกรอบ marker ทั้งหมด + label
                for m in markers:
                    cv2.rectangle(dbg, m.pt1, m.pt2, (0,255,0), 2) # marker SDK → กรอบเขียว
                    cv2.putText(dbg, str(m.text), m.center, cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,0), 2)

                    # ==== เพิ่มการคำนวณและแสดงระยะทาง ====
                    w_px = m.pt2[0] - m.pt1[0]
                    h_px = m.pt2[1] - m.pt1[1]
                    if w_px > 0 and h_px > 0:  # กันหารศูนย์
                        dist = distant_by_cammara(w_px, h_px)
                        cv2.putText(dbg, f"Dist={dist:.2f}", (m.pt1[0], m.pt1[1]-10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)

                RedTemplateDetector.draw_debug(dbg, latest_bbox, latest_score, ema_pt, True, "SRC: marker")
            else:
                # template / blob → วาดกรอบเหลืองจาก draw_debug
                RedTemplateDetector.draw_debug(dbg, latest_bbox, latest_score, ema_pt, True, f"SRC: {latest_src}")

            cv2.imshow("Merged Marker+Visual PID", dbg)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        # ===== Cleanup & Save =====
        try:
            ep_vision.unsub_detect_info(name="marker") # ยกเลิก sub marker
        except Exception:
            pass
        cv2.destroyAllWindows()
        try:
            cam.stop_video_stream()
        except Exception:
            pass
        gim.drive_speed(pitch_speed=0, yaw_speed=0)   # สั่งหยุดก่อนปิด
        try:
            gim.unsub_angle()
        except Exception:
            pass
        ep.close()

        # Save CSV
        # out = Path("gimbal_response.csv")
        # try:
        #     with out.open('w', newline='') as f:
        #         w = csv.writer(f)
        #         w.writerow(["t","pitch_angle","yaw_angle","err_x_px","err_y_px","u_yaw_dps","u_pitch_dps","score","src"])
        #         w.writerows(rows)
        #     print(f"[LOG] Saved {len(rows)} samples → {out.resolve()}")
        # except Exception as e:
        #     print(f"[WARN] CSV not saved: {e}")