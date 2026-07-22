from flask import Flask, render_template_string, Response, request, jsonify
import cv2
import math
import time
import numpy as np
from ultralytics import YOLO

app = Flask(__name__)
camera = cv2.VideoCapture(0)  # 제트슨 카메라 가동

# ======================================================================
# ⚙️ [글로벌 시스템 파라미터 및 알람/로그 상태 변수]
# ======================================================================
system_config = {
    # --- [AI 모델 신뢰도 및 성능 파라미터] ---
    "CONF_HELMET_VEST": 0.60,
    "CONF_PERSON": 0.60,
    "FRAME_INTERVAL": 5,

    # --- [시간 및 동작 감지 파라미터] ---
    "SOS_LIMIT_TIME": 5.0,
    "SOS_COOLDOWN": 0.3,
    "FALL_MOTIONLESS_TIME": 20.0,
    "FALL_MOVE_THRESHOLD": 30.0,

    # --- [비율 및 자세 판정 파라미터] ---
    "FALL_ANGLE_LIMIT": 45.0,
    "FALL_ASPECT_RATIO": 0.8,
    "HORIZONTAL_RATIO_LIMIT": 1.5,
    "SOS_EXTEND_RATIO": 0.8,
    "SOS_CROSS_DEADZONE": 0.2,
    "SOS_HEIGHT_RATIO": 0.2
}

# 🚨 알람 및 실시간 프론트엔드 로그 연동용 상태 관리 객체
alarm_state = {
    "is_triggered": False,
    "msg": ""
}

# 📋 로그 중복 생성 방지를 위한 상태 추적 플래그
event_log_tracker = {
    "pending_logs": [], # 웹 프론트엔드가 가져갈 미출력 로그 큐
    "last_helmet_state": True, # True: 정상 장착, False: 미착용 중
    "last_vest_state": True,
    "last_fall_state": False,  # True: 쓰러짐 발생 중
    "last_sos_state": False    # True: SOS 요청 중
}

def add_system_log(message):
    """서버 내부 및 웹 대시보드 로그에 이벤트를 추가하는 헬퍼 함수"""
    timestamp = time.strftime('%H:%M:%S', time.localtime())
    log_text = f"[{timestamp}] {message}"
    event_log_tracker["pending_logs"].append(log_text)
    print(f"[SYSTEM LOG] {log_text}")

MODEL_PPE  = "best.pt"           
MODEL_POSE = "yolo26m-pose.pt"   

# ======================================================================
# 🧠 [단위 클래스 및 안전 관리 엔진 정의]
# ======================================================================
class PPEItem:
    def __init__(self, coords, conf, type_name):
        self.coords = coords          
        self.conf = conf              
        self.type_name = type_name    
        self.is_claimed = False       

class PersonState:
    def __init__(self):
        self.state = 'uncrossed'      
        self.count = 0                
        self.first_time = 0           
        self.is_sos_active = False    
        self.sos_trigger_time = 0     
        self.last_uncross_time = 0    
        
        self.is_fall_active = False   
        self.fall_pose_start = 0      
        self.last_move_time = 0       
        self.last_center = None       
        
    def update_sos(self, is_crossed, current_time):
        if self.is_sos_active:
            if current_time - self.sos_trigger_time > system_config["SOS_LIMIT_TIME"]:
                self.is_sos_active = False
                self.count = 0
            return True

        if self.state == 'uncrossed' and is_crossed:
            if current_time - self.last_uncross_time > system_config["SOS_COOLDOWN"]:
                self.state = 'crossed'
                if self.count == 0:
                    self.first_time = current_time 
                
        elif self.state == 'crossed' and not is_crossed:
            self.state = 'uncrossed'
            self.count += 1
            self.last_uncross_time = current_time 
            
            if current_time - self.first_time <= system_config["SOS_LIMIT_TIME"]:
                if self.count >= 3: 
                    self.is_sos_active = True           
                    self.sos_trigger_time = current_time
                    self.count = 0                                            
                    return True
            else:
                self.count = 0
                
        return False

    def update_fall(self, is_fallen_pose, center, current_time):
        if self.is_fall_active:
            if not is_fallen_pose:
                self.is_fall_active = False 
            return self.is_fall_active

        if is_fallen_pose:
            if self.fall_pose_start == 0:
                self.fall_pose_start = current_time
                self.last_move_time = current_time
                self.last_center = center
            else:
                if self.last_center:
                    dist = math.hypot(center[0] - self.last_center[0], center[1] - self.last_center[1])
                    if dist > system_config["FALL_MOVE_THRESHOLD"]:
                        self.last_move_time = current_time 
                self.last_center = center
                
                if current_time - self.last_move_time >= system_config["FALL_MOTIONLESS_TIME"]:
                    self.is_fall_active = True
        else:
            self.fall_pose_start = 0
            self.last_move_time = 0
            self.last_center = None
            self.is_fall_active = False
            
        return self.is_fall_active

class SafetySystem:
    def __init__(self):
        self.pose_model = YOLO(MODEL_POSE) 
        self.ppe_model = YOLO(MODEL_PPE)
        self.helmet_id = 0
        self.vest_id = 1
        self.person_tracker = {} 
        
        self.cached_persons = []
        self.cached_helmets = []
        self.cached_vests = []
        self.global_flags = (False, False, False, False) 

    def update_ai(self, frame):
        pose_results = self.pose_model.track(frame, persist=True, verbose=False)
        ppe_results = self.ppe_model(frame, verbose=False)
        current_time = time.time()
        
        all_helmets = []
        all_vests = []
        
        for r in ppe_results:
            for box in r.boxes:
                conf = float(box.conf[0])
                if conf >= system_config["CONF_HELMET_VEST"]:  
                    cls_id = int(box.cls[0])
                    coords = list(map(int, box.xyxy[0]))
                    if cls_id == self.helmet_id:
                        all_helmets.append(PPEItem(coords, conf, 'Helmet'))
                    elif cls_id == self.vest_id:
                        all_vests.append(PPEItem(coords, conf, 'Vest'))
                        
        global_fall = False
        global_no_helmet = False
        global_no_vest = False
        global_sos = False 
        
        cached_persons_temp = []
        
        for r in pose_results:
            if r.boxes is None or r.keypoints is None: continue
            
            track_ids = r.boxes.id.int().cpu().tolist() if r.boxes.id is not None else []
            
            valid_indices = []
            for i, box_i in enumerate(r.boxes):
                if float(box_i.conf[0]) < system_config["CONF_PERSON"]: continue
                
                is_duplicate = False
                xi1, yi1, xi2, yi2 = map(int, box_i.xyxy[0])
                area_i = (xi2 - xi1) * (yi2 - yi1)
                
                for j in valid_indices:
                    box_j = r.boxes[j]
                    xj1, yj1, xj2, yj2 = map(int, box_j.xyxy[0])
                    area_j = (xj2 - xj1) * (yj2 - yj1)
                    
                    x_left = max(xi1, xj1)
                    y_top = max(yi1, yj1)
                    x_right = min(xi2, xj2)
                    y_bottom = min(yi2, yj2)
                    
                    if x_right > x_left and y_bottom > y_top:
                        inter_area = (x_right - x_left) * (y_bottom - y_top)
                        min_area = min(area_i, area_j)
                        if min_area > 0 and (inter_area / min_area) > 0.5:
                            is_duplicate = True
                            break
                            
                if not is_duplicate:
                    valid_indices.append(i)
            
            for i in valid_indices:
                box = r.boxes[i]
                person_conf = float(box.conf[0])
                
                track_id = track_ids[i] if i < len(track_ids) else -1
                if track_id != -1:
                    if track_id not in self.person_tracker:
                        self.person_tracker[track_id] = PersonState() 
                    person_obj = self.person_tracker[track_id]
                else:
                    person_obj = PersonState()
                
                px1, py1, px2, py2 = map(int, box.xyxy[0])
                width = px2 - px1
                height = py2 - py1
                
                kpts = r.keypoints.xy[i].cpu().numpy() 
                
                ls = kpts[5]; rs = kpts[6]; lh = kpts[11]; rh = kpts[12] 
                le = kpts[7]; re = kpts[8]                               
                lw = kpts[9]; rw = kpts[10]                              
                
                is_crossed = False
                left_hand_pt = None
                right_hand_pt = None
                
                if ls[0] > 0 and rs[0] > 0 and lw[0] > 0 and rw[0] > 0 and le[0] > 0 and re[0] > 0:
                    left_hand_x = lw[0] + (lw[0] - le[0]) * system_config["SOS_EXTEND_RATIO"]
                    left_hand_y = lw[1] + (lw[1] - le[1]) * system_config["SOS_EXTEND_RATIO"]
                    right_hand_x = rw[0] + (rw[0] - re[0]) * system_config["SOS_EXTEND_RATIO"]
                    right_hand_y = rw[1] + (rw[1] - re[1]) * system_config["SOS_EXTEND_RATIO"]
                    
                    left_hand_pt = (int(left_hand_x), int(left_hand_y))
                    right_hand_pt = (int(right_hand_x), int(right_hand_y))

                    shoulder_diff = ls[0] - rs[0]             
                    hand_diff = left_hand_x - right_hand_x    
                    shoulder_width = abs(shoulder_diff)
                    
                    is_x_crossed = (shoulder_diff * hand_diff < 0) and (abs(hand_diff) > shoulder_width * system_config["SOS_CROSS_DEADZONE"])
                    hands_up = (left_hand_y < le[1]) and (right_hand_y < re[1])
                    shoulder_y_avg = (ls[1] + rs[1]) / 2
                    hands_high = (left_hand_y < shoulder_y_avg + height * system_config["SOS_HEIGHT_RATIO"]) and (right_hand_y < shoulder_y_avg + height * system_config["SOS_HEIGHT_RATIO"])

                    if is_x_crossed and hands_up and hands_high:
                        is_crossed = True 

                is_sos = person_obj.update_sos(is_crossed, current_time)
                if is_sos:
                    global_sos = True

                is_fallen_pose = False
                if ls[0] > 0 and rs[0] > 0 and lh[0] > 0 and rh[0] > 0:
                    shoulder_x = (ls[0] + rs[0]) / 2
                    shoulder_y = (ls[1] + rs[1]) / 2
                    hip_x = (lh[0] + rh[0]) / 2
                    hip_y = (lh[1] + rh[1]) / 2
                    
                    dx = hip_x - shoulder_x
                    dy = hip_y - shoulder_y
                    if dx != 0: angle = math.degrees(math.atan2(dy, dx))
                    else: angle = 90
                    
                    is_aspect_ratio_wide = width > (height * system_config["FALL_ASPECT_RATIO"])

                    if (abs(angle) <= system_config["FALL_ANGLE_LIMIT"] or abs(angle) >= (180 - system_config["FALL_ANGLE_LIMIT"])) and is_aspect_ratio_wide:
                        is_fallen_pose = True

                person_center = ((px1 + px2) / 2, (py1 + py2) / 2)
                is_fall = person_obj.update_fall(is_fallen_pose, person_center, current_time)
                
                if is_fall:
                    global_fall = True
                    is_sos = False 
                    person_obj.is_sos_active = False 

                has_helmet = False
                has_vest = False
                margin_x = width * 0.2
                margin_y = height * 0.1

                for helmet in all_helmets:
                    if not helmet.is_claimed: 
                        cx = (helmet.coords[0] + helmet.coords[2]) / 2 
                        cy = (helmet.coords[1] + helmet.coords[3]) / 2 
                        if px1 - margin_x <= cx <= px2 + margin_x and py1 - margin_y <= cy <= py2 + margin_y:
                            has_helmet = True
                            helmet.is_claimed = True 
                            break
                            
                for vest in all_vests:
                    if not vest.is_claimed:
                        cx = (vest.coords[0] + vest.coords[2]) / 2
                        cy = (vest.coords[1] + vest.coords[3]) / 2
                        if px1 - margin_x <= cx <= px2 + margin_x and py1 - margin_y <= cy <= py2 + margin_y:
                            has_vest = True
                            vest.is_claimed = True
                            break
                            
                if not has_helmet: global_no_helmet = True
                if not has_vest: global_no_vest = True
                    
                cached_persons_temp.append({
                    'coords': (px1, py1, px2, py2),
                    'width': width, 'height': height,
                    'kpts': kpts, 'conf': person_conf,
                    'is_sos': is_sos, 'is_fall': is_fall, 'is_crossed': is_crossed,
                    'sos_count': person_obj.count,
                    'has_helmet': has_helmet, 'has_vest': has_vest,
                    'left_hand_pt': left_hand_pt, 'right_hand_pt': right_hand_pt
                })

        self.cached_persons = cached_persons_temp
        self.cached_helmets = all_helmets
        self.cached_vests = all_vests
        self.global_flags = (global_no_helmet, global_no_vest, global_fall, global_sos)

    def draw(self, frame):
        for p in self.cached_persons:
            px1, py1, px2, py2 = p['coords']
            kpts = p['kpts']
            
            if p['left_hand_pt'] and p['right_hand_pt']:
                lw = kpts[9]; rw = kpts[10]
                cv2.line(frame, (int(lw[0]), int(lw[1])), p['left_hand_pt'], (255, 0, 255), 2)
                cv2.line(frame, (int(rw[0]), int(rw[1])), p['right_hand_pt'], (255, 0, 255), 2)
                cv2.circle(frame, p['left_hand_pt'], 6, (255, 0, 255), -1)
                cv2.circle(frame, p['right_hand_pt'], 6, (255, 0, 255), -1)

            if p['is_fall']:
                cv2.rectangle(frame, (px1, py1), (px2, py2), (0, 0, 255), 3)
                self.draw_skeleton(frame, kpts, color=(0, 0, 255))   
                cv2.putText(frame, f"Fallen {p['conf']*100:.0f}%", (px1, py1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            elif p['is_sos']:
                cv2.rectangle(frame, (px1, py1), (px2, py2), (0, 0, 255), 3) 
                self.draw_skeleton(frame, kpts, color=(0, 0, 255))   
                cv2.putText(frame, "SOS REQUEST!", (px1, py1 - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 3)
            else:
                self.draw_skeleton(frame, kpts, color=(0, 255, 0))   
                
                status_text = f"Worker {p['conf']*100:.0f}%"
                if p['sos_count'] > 0 and not p['is_sos']:
                    status_text += f" | SOS: {p['sos_count']}/3"
                cv2.putText(frame, status_text, (px1, py2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                
                if p['is_crossed'] and not p['is_sos']:
                    cv2.putText(frame, "Arms Crossed!", (px1, py1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
            
            if not p['is_sos'] and not p['is_fall']:
                width, height = p['width'], p['height']
                ls = kpts[5]; rs = kpts[6]; lh = kpts[11]; rh = kpts[12]
                is_horizontal = False
                
                if ls[0] > 0 and rs[0] > 0 and lh[0] > 0 and rh[0] > 0:
                    shoulder_center_x, shoulder_y = (ls[0] + rs[0]) / 2, (ls[1] + rs[1]) / 2
                    hip_x, hip_y = (lh[0] + rh[0]) / 2, (lh[1] + rh[1]) / 2
                    
                    if abs(hip_x - shoulder_center_x) > abs(hip_y - shoulder_y):
                        if width > height * 1.2:
                            is_horizontal = True
                            
                elif width > height * system_config["HORIZONTAL_RATIO_LIMIT"]:
                    is_horizontal = True

                if is_horizontal:
                    if ls[0] > 0 and rs[0] > 0: shoulder_center_x = (ls[0] + rs[0]) / 2
                    elif ls[0] > 0: shoulder_center_x = ls[0]
                    elif rs[0] > 0: shoulder_center_x = rs[0]
                    else: shoulder_center_x = px1 
                    
                    is_head_right = shoulder_center_x > (px1 + px2) / 2

                    if is_head_right: 
                        hx1, hy1 = px2 - int(width*0.2), py1 + int(height*0.3)
                        hx2, hy2 = px2, py2 - int(height*0.3)
                        vx1, vy1 = px1 + int(width*0.1), py1 + int(height*0.1)
                        vx2, vy2 = px2 - int(width*0.2), py2 - int(height*0.1)
                    else:             
                        hx1, hy1 = px1, py1 + int(height*0.3)
                        hx2, hy2 = px1 + int(width*0.2), py2 - int(height*0.3)
                        vx1, vy1 = px1 + int(width*0.2), py1 + int(height*0.1)
                        vx2, vy2 = px2 - int(width*0.1), py2 - int(height*0.1)
                else: 
                    hx1, hy1 = px1 + int(width*0.3), py1
                    hx2, hy2 = px2 - int(width*0.3), py1 + int(height*0.2)
                    vx1, vy1 = px1 + int(width*0.1), py1 + int(height*0.2)
                    vx2, vy2 = px2 - int(width*0.1), py2 - int(height*0.1)

                if not p['has_helmet']:
                    cv2.rectangle(frame, (hx1, hy1), (hx2, hy2), (0, 165, 255), 3) 
                    cv2.putText(frame, 'No Helmet', (hx1, hy1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)
                if not p['has_vest']:
                    cv2.rectangle(frame, (vx1, vy1), (vx2, vy2), (0, 165, 255), 3) 
                    cv2.putText(frame, 'No Vest', (vx1, vy1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)

        for helmet in self.cached_helmets:
            x1, y1, x2, y2 = helmet.coords
            if helmet.is_claimed:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"Helmet {helmet.conf*100:.0f}%", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                
        for vest in self.cached_vests:
            x1, y1, x2, y2 = vest.coords
            if vest.is_claimed:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"Vest {vest.conf*100:.0f}%", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        return frame, self.global_flags

    def draw_skeleton(self, frame, kpts, color=(0,255,0)):
        skeleton = [(5,6), (5,7), (7,9), (6,8), (8,10), (5,11), (6,12), (11,12), (11,13), (13,15), (12,14), (14,16)]
        for pt1, pt2 in skeleton:
            if pt1 < len(kpts) and pt2 < len(kpts):
                x1, y1 = int(kpts[pt1][0]), int(kpts[pt1][1])
                x2, y2 = int(kpts[pt2][0]), int(kpts[pt2][1])
                if x1 > 0 and y1 > 0 and x2 > 0 and y2 > 0:
                    cv2.line(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.circle(frame, (x1, y1), 3, color, -1)
                    cv2.circle(frame, (x2, y2), 3, color, -1)

# ========================================================
# 🚨 [알람 발생 및 락킹 그리기 레이어]
# ========================================================
def render_alarm_overlay(frame, msg_text):
    cv2.rectangle(frame, (0, 0), (frame.shape[1], 50), (0, 0, 255), -1)
    cv2.putText(frame, f"ALARM: {msg_text}", (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

system = SafetySystem()

def gen_frames():  
    global system, alarm_state, event_log_tracker
    frame_count = 0
    
    while True:
        success, frame = camera.read()
        if not success: break
        
        frame_count += 1
        if frame_count % int(system_config["FRAME_INTERVAL"]) == 0:
            system.update_ai(frame)
        
        frame, flags = system.draw(frame)
        danger_no_helmet, danger_no_vest, is_fall, is_sos = flags

        # ========================================================
        # 📋 [실시간 웹 관제 이벤트 로그 감지 및 트리거 장치]
        # ========================================================
        # 1. 보호구 미착용 상태 변화 추적 로그
        if danger_no_helmet and event_log_tracker["last_helmet_state"]:
            add_system_log("⚠️ [위험] 작업자 미확인 안전모(No Helmet) 상태 감지됨!")
            event_log_tracker["last_helmet_state"] = False
        elif not danger_no_helmet and not event_log_tracker["last_helmet_state"]:
            event_log_tracker["last_helmet_state"] = True

        if danger_no_vest and event_log_tracker["last_vest_state"]:
            add_system_log("⚠️ [위험] 작업자 미확인 안전조끼(No Vest) 상태 감지됨!")
            event_log_tracker["last_vest_state"] = False
        elif not danger_no_vest and not event_log_tracker["last_vest_state"]:
            event_log_tracker["last_vest_state"] = True

        # 2. 쓰러짐 감지 상태 변화 추적 로그 및 알람 트리거
        if is_fall and not event_log_tracker["last_fall_state"]:
            add_system_log("🚨 [비상] 작업자 쓰러짐(Fall) 비상 사태가 발생했습니다!!")
            event_log_tracker["last_fall_state"] = True
        elif not is_fall and event_log_tracker["last_fall_state"]:
            event_log_tracker["last_fall_state"] = False

        # 3. SOS 신호 상태 변화 추적 로그 및 알람 트리거
        if is_sos and not event_log_tracker["last_sos_state"]:
            add_system_log("🚨 [요청] 작업자 SOS 수신호 구조 요청이 발생했습니다!!")
            event_log_tracker["last_sos_state"] = True
        elif not is_sos and event_log_tracker["last_sos_state"]:
            event_log_tracker["last_sos_state"] = False

        # ✅ 1. 알람 락킹(Locking) 작동 조건 변경: 보호구 미착용 제외, 오직 'SOS'와 '쓰러짐' 발생 시에만 켜짐
        if is_fall or is_sos:
            alarm_state["is_triggered"] = True
            active_errors = []
            if is_sos: active_errors.append("[SOS DETECTED]")
            if is_fall: active_errors.append("[FALL DETECTED]")
            alarm_state["msg"] = " ".join(active_errors)

        # 알람 조건이 충족되어 락이 걸린 경우 화면 상단 알람 오버레이 렌더링
        if alarm_state["is_triggered"]:
            render_alarm_overlay(frame, alarm_state["msg"])

        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

# ========================================================
# 🖥️ [웹 관제 대시보드 - HTML/CSS/JS]
# ========================================================
@app.route('/')
def index():
    html_layout = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Industrial Safety AI Vision Center</title>
        <style>
            body { font-family: 'Malgun Gothic', Arial, sans-serif; background-color: #121212; color: #e0e0e0; margin: 30px; }
            .main-header { display: flex; align-items: center; justify-content: space-between; border-bottom: 3px solid #00adb5; padding-bottom: 15px; }
            
            .container { display: flex; gap: 30px; margin-top: 25px; align-items: flex-start; }
            .video-section { background: #1e1e1e; padding: 20px; border-radius: 14px; box-shadow: 0 6px 15px rgba(0,0,0,0.6); border: 1px solid #2d2d2d; flex: 1; }
            .video-feed { border: 5px solid #2d2d2d; border-radius: 8px; display: block; background: #000; width: 100%; height: auto; max-width: 640px; }
            
            .control-section { background: #1e1e1e; padding: 20px; border-radius: 14px; width: 400px; box-shadow: 0 6px 15px rgba(0,0,0,0.6); border: 1px solid #2d2d2d; }
            h3 { margin-top: 0; color: #00adb5; border-bottom: 1px solid #333; padding-bottom: 10px; font-size: 18px; }
            
            button { padding: 12px; margin: 5px 0; font-size: 14px; font-weight: bold; cursor: pointer; border-radius: 6px; border: none; transition: 0.2s; }
            
            .btn-alarm-off { background-color: #b71c1c; color: #ffebee; width: 100%; font-size: 16px; padding: 15px; box-shadow: 0 0 15px rgba(183,28,28,0.4); }
            .btn-alarm-off:hover { background-color: #d32f2f; box-shadow: 0 0 20px rgba(211,47,47,0.6); }
            
            .status-log { margin-top: 15px; background: #0a0a0a; padding: 12px; border-radius: 8px; font-family: 'Courier New', monospace; font-size: 12px; color: #00ff55; height: 140px; overflow-y: auto; border: 1px solid #222; }
            
            .group-title { margin-top: 25px; margin-bottom: 8px; color: #00adb5; font-size: 15px; font-weight: bold; }
            .group-wrapper { display: flex; gap: 5px; margin-bottom: 15px; }
            .btn-group { background: #333; color: #ccc; flex: 1; font-size: 11px; padding: 10px 3px; border-radius: 4px; }
            .btn-group.active { background: #00adb5; color: #fff; box-shadow: 0 0 8px #00adb5; font-weight: bold; }
            
            .param-container { background: #252525; border-radius: 8px; padding: 15px; border: 1px solid #3a3a3a; display: none; }
            .param-container.active { display: block; }
            .param-row { display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px; }
            .param-row:last-child { margin-bottom: 0; }
            .param-row label { font-size: 13px; color: #b3b3b3; line-height: 1.3; }
            .param-row input { width: 80px; background: #111; border: 1px solid #555; color: #fff; padding: 6px; border-radius: 4px; text-align: center; font-size: 13px; }
            .btn-apply { background: #00adb5; color: #fff; width: 100%; font-size: 15px; margin-top: 15px; border-radius: 6px; padding: 12px; }
            .btn-apply:hover { background: #008c9e; }
        </style>
        <script>
            // 🔄 실시간 로그 갱신을 위해 0.5초마다 백엔드 서버로부터 새로운 로그 데이터를 수집(폴링)
            setInterval(function() {
                fetch('/get_live_logs')
                    .then(response => response.json())
                    .then(data => {
                        if (data.logs && data.logs.length > 0) {
                            let logBox = document.getElementById('log');
                            data.logs.forEach(msg => {
                                logBox.innerHTML += msg + "<br>";
                            });
                            logBox.scrollTop = logBox.scrollHeight;
                        }
                    });
            }, 500);

            function clearAlarm() {
                fetch('/clear_alarm')
                    .then(response => response.json())
                    .then(data => {
                        // 알람 OFF 해제 로그는 수동 명령 즉시 출력하도록 직접 처리
                        let logBox = document.getElementById('log');
                        logBox.innerHTML += "[" + new Date().toLocaleTimeString() + "] " + data.message + "<br>";
                        logBox.scrollTop = logBox.scrollHeight;
                    });
            }

            function switchGroup(groupName, element) {
                document.querySelectorAll('.param-container').forEach(el => el.classList.remove('active'));
                document.querySelectorAll('.btn-group').forEach(el => el.classList.remove('active'));
                
                document.getElementById('group-' + groupName).classList.add('active');
                element.classList.add('active');
            }

            function applyParameters() {
                let formData = new FormData();
                document.querySelectorAll('.param-container input').forEach(input => {
                    formData.append(input.name, input.value);
                });

                fetch('/update_params', {
                    method: 'POST',
                    body: formData
                })
                .then(response => response.json())
                .then(data => {
                    let logBox = document.getElementById('log');
                    logBox.innerHTML += "[" + new Date().toLocaleTimeString() + "] " + data.message + "<br>";
                    logBox.scrollTop = logBox.scrollHeight;
                });
            }
        </script>
    </head>
    <body>
        <div class="main-header">
            <h2>🛡️ 산업안전 AI 비전 모니터링 시스템 v3.5</h2>
            <span style="color: #00adb5; font-weight: bold; background: #222; padding: 6px 12px; border-radius: 20px;">Jetson Cluster Mode</span>
        </div>
        
        <div class="container">
            <div class="video-section">
                <h3>📷 실시간 관제 및 지능형 추론 스트림 (Live AI Inference Feed)</h3>
                <img class="video-feed" src="/video_feed">
            </div>
            
            <div class="control-section">
                <h3>🕹️ 원격 안전 관제 콘솔</h3>
                
                <button class="btn-alarm-off" onclick="clearAlarm()">🚨 알람 OFF (정상 복구)</button>
                
                <h4>📋 시스템 보안 및 안전 이벤트 로그</h4>
                <div class="status-log" id="log">[안내] 인공지능 영상 관제 레이어가 가동되었습니다.<br></div>

                <div class="group-title">⚙️ Parameter Group 설정 및 튜닝</div>
                <div class="group-wrapper">
                    <button class="btn-group active" onclick="switchGroup('model', this)">AI 모델 신뢰도</button>
                    <button class="btn-group" onclick="switchGroup('time', this)">시간 및 동작 감지</button>
                    <button class="btn-group" onclick="switchGroup('pose', this)">비율 및 자세 판정</button>
                </div>

                <div id="group-model" class="param-container active">
                    <div class="param-row">
                        <label>CONF_HELMET_VEST<br><small style="color:#777;">장비 인식 임계값</small></label>
                        <input type="text" name="CONF_HELMET_VEST" value="{{ config['CONF_HELMET_VEST'] }}">
                    </div>
                    <div class="param-row">
                        <label>CONF_PERSON<br><small style="color:#777;">사람 감지 임계값</small></label>
                        <input type="text" name="CONF_PERSON" value="{{ config['CONF_PERSON'] }}">
                    </div>
                    <div class="param-row">
                        <label>FRAME_INTERVAL<br><small style="color:#777;">AI 연산 스킵 주기</small></label>
                        <input type="text" name="FRAME_INTERVAL" value="{{ config['FRAME_INTERVAL'] }}">
                    </div>
                </div>

                <div id="group-time" class="param-container">
                    <div class="param-row">
                        <label>SOS_LIMIT_TIME<br><small style="color:#777;">SOS 완료 제한시간 (초)</small></label>
                        <input type="text" name="SOS_LIMIT_TIME" value="{{ config['SOS_LIMIT_TIME'] }}">
                    </div>
                    <div class="param-row">
                        <label>SOS_COOLDOWN<br><small style="color:#777;">손 풀림 유예시간 (초)</small></label>
                        <input type="text" name="SOS_COOLDOWN" value="{{ config['SOS_COOLDOWN'] }}">
                    </div>
                    <div class="param-row">
                        <label>FALL_MOTIONLESS_TIME<br><small style="color:#777;">무동작 대기시간 (초)</small></label>
                        <input type="text" name="FALL_MOTIONLESS_TIME" value="{{ config['FALL_MOTIONLESS_TIME'] }}">
                    </div>
                    <div class="param-row">
                        <label>FALL_MOVE_THRESHOLD<br><small style="color:#777;">움직임 임계거리 (px)</small></label>
                        <input type="text" name="FALL_MOVE_THRESHOLD" value="{{ config['FALL_MOVE_THRESHOLD'] }}">
                    </div>
                </div>

                <div id="group-pose" class="param-container">
                    <div class="param-row">
                        <label>FALL_ANGLE_LIMIT<br><small style="color:#777;">쓰러짐 허용 각도 (도)</small></label>
                        <input type="text" name="FALL_ANGLE_LIMIT" value="{{ config['FALL_ANGLE_LIMIT'] }}">
                    </div>
                    <div class="param-row">
                        <label>FALL_ASPECT_RATIO<br><small style="color:#777;">가로/세로 박스 비율</small></label>
                        <input type="text" name="FALL_ASPECT_RATIO" value="{{ config['FALL_ASPECT_RATIO'] }}">
                    </div>
                    <div class="param-row">
                        <label>HORIZONTAL_RATIO_LIMIT<br><small style="color:#777;">강제 누움 처리 비율</small></label>
                        <input type="text" name="HORIZONTAL_RATIO_LIMIT" value="{{ config['HORIZONTAL_RATIO_LIMIT'] }}">
                    </div>
                    <div class="param-row">
                        <label>SOS_EXTEND_RATIO<br><small style="color:#777;">손끝 가상 연장 비율</small></label>
                        <input type="text" name="SOS_EXTEND_RATIO" value="{{ config['SOS_EXTEND_RATIO'] }}">
                    </div>
                    <div class="param-row">
                        <label>SOS_CROSS_DEADZONE<br><small style="color:#777;">양손 최소 교차 깊이</small></label>
                        <input type="text" name="SOS_CROSS_DEADZONE" value="{{ config['SOS_CROSS_DEADZONE'] }}">
                    </div>
                    <div class="param-row">
                        <label>SOS_HEIGHT_RATIO<br><small style="color:#777;">손 거치 최소 높이 비율</small></label>
                        <input type="text" name="SOS_HEIGHT_RATIO" value="{{ config['SOS_HEIGHT_RATIO'] }}">
                    </div>
                </div>

                <button class="btn-apply" onclick="applyParameters()">⚡ 파라미터 즉시 적용</button>
            </div>
        </div>
    </body>
    </html>
    """
    return render_template_string(html_layout, config=system_config)

# ========================================================
# [원격 제어 및 동적 변수 업데이트 통신 API 엔드포인트]
# ========================================================
@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# ✅ 2. 파이썬 백엔드에서 생성된 최신 실시간 이벤트 로그들을 웹 브라우저로 리턴하는 폴링 API
@app.route('/get_live_logs')
def get_live_logs():
    global event_log_tracker
    logs = list(event_log_tracker["pending_logs"])
    event_log_tracker["pending_logs"].clear() # 반환 후 큐 비우기
    return jsonify(logs=logs)

# ✅ 2. 웹 대시보드에서 알람 수동 OFF 해제 요청을 처리하는 API (해제 로그 반영)
@app.route('/clear_alarm')
def clear_alarm():
    global alarm_state
    alarm_state["is_triggered"] = False
    alarm_state["msg"] = ""
    # 수동 복구 발생 시 로그 시스템 연동
    add_system_log("🟢 [정상] 관제자 명령: 마스터 알람을 정상 해제 상태로 원격 리셋했습니다.")
    return jsonify(status="success", message="마스터 알람 원격 복구 완료")

@app.route('/update_params', methods=['POST'])
def update_params():
    global system_config
    try:
        updated_keys = []
        for key in system_config.keys():
            if key in request.form:
                if key in ["FRAME_INTERVAL"]:
                    system_config[key] = int(request.form[key])
                else:
                    system_config[key] = float(request.form[key])
                updated_keys.append(key)
        
        return jsonify(status="success", message=f"[설정 변경] {len(updated_keys)}개의 코어 파라미터 즉시 반영 완료.")
    except Exception as e:
        return jsonify(status="fail", message=f"[오류] 데이터 형식이 올바르지 않습니다: {str(e)}")

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)