import cv2
import numpy as np
from djitellopy import Tello
import time

#tello.move_up()
# tello.rotate_clockwise()
# tello.move_forward()

# --- KONFIGURASI ---
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
aruco_params = cv2.aruco.DetectorParameters()

focal_length = 900 
tag_width = 0.16  # 16 cm

# Initialize Tello
tello = Tello()
tello.connect()
print(f"Bateri: {tello.get_battery()}%")
tello.streamon()

print("Drone berlepas...")
tello.takeoff()
time.sleep(2)

def cari_sampai_stabil(target_id, target_dist):
    """Fungsi bantu: Cari tag, rapatkan drone sampai jarak betul, kemudian berhenti"""
    print(f"\n--- Mula Cari ID: {target_id} (Target Jarak: {target_dist}cm) ---")
    
    while True:
        frame = tello.get_frame_read().frame
        if frame is None: 
            continue
        
        frame = cv2.resize(frame, (960, 720))
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, rejected = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=aruco_params)
        
        found_target = False

        if ids is not None:
            for i, marker_id in enumerate(ids.flatten()):
                if marker_id == target_id:
                    found_target = True
                    cv2.aruco.drawDetectedMarkers(frame, corners, ids)
                    
                    # Kira kedudukan tengah & Jarak sebenar
                    c = corners[i][0]
                    center_x = int((c[0][0] + c[2][0]) / 2)
                    center_y = int((c[0][1] + c[2][1]) / 2)
                    pixel_width = np.linalg.norm(c[0] - c[1])
                    
                    if pixel_width == 0:
                        continue
                    
                    dist_cm = (tag_width * focal_length) / pixel_width * 100

                    info = f"Cari ID:{target_id} | Target:{target_dist}cm | Semasa:{int(dist_cm)}cm"
                    cv2.putText(frame, info, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                    # Kawalan halus untuk letak tengah & jarak tepat
                    lr, fb, ud, yaw = 0, 0, 0, 0
                    
                    if center_x < 400: lr = -15
                    elif center_x > 560: lr = 15
                    
                    if center_y < 300: ud = 15
                    elif center_y > 420: ud = -15
                    
                    if dist_cm > (target_dist + 15): fb = 15
                    elif dist_cm < (target_dist - 15): fb = -15

                    # Semak sama ada drone dah stabil depan tag
                    is_centered = (400 <= center_x <= 560) and (300 <= center_y <= 420)
                    is_dist_stable = abs(target_dist - dist_cm) < 15

                    if is_centered and is_dist_stable:
                        print(f"ID {target_id} sudah stabil pada jarak {int(dist_cm)}cm.")
                        tello.send_rc_control(0, 0, 0, 0) # Hentikan kawalan halus
                        cv2.imshow("Step-by-Step Navigation", frame)
                        cv2.waitKey(500)
                        return # Selesai langkah ni, keluar dari fungsi!

                    tello.send_rc_control(lr, fb, ud, yaw)
                    break

        if not found_target:
            cv2.putText(frame, f"CARI ID: {target_id}", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            tello.send_rc_control(0, 0, 0, 20) # Pusing perlahan cari tag

        cv2.imshow("Step-by-Step Navigation", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Manual abort...")
            tello.land()
            exit()

try:
    # ====================================================
    # --- SENARAI MISI DI PECAHKAN SATU SATU (STEP-BY-STEP) ---
    # ====================================================

    # --- LANGKAH 1 ---
    cari_sampai_stabil(target_id=1, target_dist=150) # Cari ID 1 sampai jarak 150cm
    tello.send_rc_control(0, 0, 0, 0)
    time.sleep(1)
    print("Tindakan 1: Naik atas 30cm")
    tello.move_up(30)
    tello.move_forward(370)
    time.sleep(1)

    # --- LANGKAH 2 ---
    cari_sampai_stabil(target_id=2, target_dist=200) # Cari ID 2 sampai jarak 200cm
    tello.send_rc_control(0, 0, 0, 0)
    time.sleep(1)
    print("Tindakan 2: Maju ke depan 50cm")
    tello.move_forward(300)
    tello.move_down(50)
    time.sleep(1)

    # --- LANGKAH 3 ---
    cari_sampai_stabil(target_id=3, target_dist=180) # Cari ID 3 sampai jarak 180cm
    tello.send_rc_control(0, 0, 0, 0)
    time.sleep(1)
    print("Tindakan 3: Pusing kanan 90 darjah")
    tello.move_forward(300)
    time.sleep(1)

    # --- LANGKAH 4 ---
    cari_sampai_stabil(target_id=4, target_dist=200) # Cari ID 4 sampai jarak 200cm
    tello.send_rc_control(0, 0, 0, 0)
    time.sleep(1)
    tello.move_up(30)
    tello.move_forward(300)
    print("Tindakan 4: Mendarat")
    tello.land()

    print("Semua misi berjaya diselesaikan!")

except Exception as e:
    print(f"Ralat berlaku: {e}")
    tello.land()

finally:
    cv2.destroyAllWindows()
    tello.streamoff()