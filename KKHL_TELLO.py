#TELLO_mrbotart

import time
import cv2
import numpy as np
from djitellopy import Tello

# =================== LOW-WIFI SETTINGS ===================
FRAME_SIZE = (640, 480)          # Keep tags reasonably large for detection
CX, CY = FRAME_SIZE[0] // 2, FRAME_SIZE[1] // 2
TAG_WIDTH_M = 0.16              # Measure the black AprilTag square, not paper
FOCAL_LENGTH_PX = 600.0         # ESTIMATE: calibrate using a known distance
MIN_BATTERY = 10

VIDEO_FPS = 15                   # Reduced from 15 FPS for weak Wi-Fi
VIDEO_BITRATE_MBPS = 1         # 1 Mbps reduces network load (may blur tags)
FRAME_LOOP_SEC = 0.10          # ~10 control decisions/second
VIDEO_HOVER_AFTER_SEC = 0.7    # No fresh video -> stop sending motion


CENTER_X_TOL = 60              # All IDs: +/-60 px horizontally
CENTER_Y_TOL = 55              # All IDs: +/-55 px vertically
DIST_TOL_CM = 25               # All IDs: +/-25 cm from target
STABLE_FRAMES = 3              # Require three DIFFERENT video frames
STABLE_SECONDS = 0.20

LR_SPEED = 12                   # Slow sideways movement
UD_SPEED = 12                   # Slow up/down movement
FB_SPEED = 12                   # Slow forward/backward adjustment
YAW_ALIGN_SPEED = 12           # Slow yaw when tag is far from screen center
YAW_FIRST_PX = 110
SCAN_YAW_SPEED = 12            # Slow continuous rotation

SEARCH_STEP_CM = 30
MIN_HEIGHT_CM = 50
# Height search cycles repeat until tag appears (no search timeout)
SCAN_FALLBACK_SEC = 55          # Timed fallback is NOT exact 360 degrees
SCAN_LIMIT_SEC = 75

TAG_LOST_GRACE_SEC = 2.0

MISSION_SPEED_CM_S = 25         # Explicit forward speed in cm/s
RIGHT_TURN_DEG = 85             # Clockwise; change to 90 if needed
GATE_UP_CM = 30                 # Height increase AFTER alignment, BEFORE passing
# IMPORTANT: Up 30 cm at every gate means a cumulative +120 cm (not counting search shifts).
# Check gate aperture and ceiling clearance before flying.
# Gate entries: (AprilTag ID, target alignment distance, forward pass distance)
GATES = [
    (1, 200, 290),
    (2, 200, 300),
    (3, 180, 300),
    (4, 200, 300),
]
WINDOW = 'Tello Low WiFi / No Auto Land - Q to land'

# =================== APRILTAG DETECTOR ===================
cv2.setUseOptimized(True)
if not hasattr(cv2, 'aruco') or not hasattr(cv2.aruco, 'DICT_APRILTAG_36h11'):
    raise RuntimeError('Missing AprilTag support: install opencv-contrib-python')
A = cv2.aruco
D = A.getPredefinedDictionary(A.DICT_APRILTAG_36h11)
P = A.DetectorParameters()
DETECTOR = A.ArucoDetector(D, P) if hasattr(A, 'ArucoDetector') else None

T = Tello()
reader = None
stream_started = False
last_raw_ref = None             # Keep reference to detect newly decoded frame
last_fresh_frame_at = None
fps_time_start = time.monotonic()
fps_frames = 0
measured_fps = 0.0
last_status_at = 0.0


def stop_motion(force=False):
    """Send zero RC controls (twice on important transitions)."""
    if not T.is_flying:
        return
    T.send_rc_control(0, 0, 0, 0)
    if force:
        time.sleep(0.12)       # DJITelloPy rate-limits RC updates
        T.send_rc_control(0, 0, 0, 0)


def check_key():
    if cv2.waitKey(1) & 0xFF in (ord('q'), ord('Q'), 27):
        raise KeyboardInterrupt('Q/ESC pressed')


def video_age():
    if last_fresh_frame_at is None:
        return 0.0
    return time.monotonic() - last_fresh_frame_at


def check_video_stall():
    """React to a decoder stall; this CANNOT detect every kind of network failure."""
    global last_status_at
    elapsed = video_age()
    if elapsed >= VIDEO_HOVER_AFTER_SEC:
        stop_motion()
        if time.monotonic() - last_status_at > 1.0:
            print(f'WARNING: no new video frame for {elapsed:.1f}s, hovering')
            last_status_at = time.monotonic()
    return elapsed < VIDEO_HOVER_AFTER_SEC


def show_waiting_frame(target_id):
    """Keep UI alive even when video stream stops decoding."""
    if video_age() >= VIDEO_HOVER_AFTER_SEC:
        canvas = np.zeros((FRAME_SIZE[1], FRAME_SIZE[0], 3), dtype=np.uint8)
        cv2.putText(canvas, f'WAITING FOR VIDEO - ID {target_id}', (22, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 200, 255), 2)
        cv2.putText(canvas, 'HOVER / NO AUTO LANDING', (22, 165),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)
        cv2.putText(canvas, 'Press Q or ESC to request LANDING', (22, 210),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
        cv2.imshow(WINDOW, canvas)


def read_tag(target_id, status=''):
    """Return (tag, fresh). Only newly decoded frames count toward alignment."""
    global last_raw_ref, last_fresh_frame_at, fps_time_start, fps_frames, measured_fps
    raw = reader.frame if reader is not None else None
    if raw is None or raw.size == 0 or raw is last_raw_ref:
        check_video_stall()
        show_waiting_frame(target_id)
        check_key()
        return None, False

    # BackgroundFrameRead sets a NEW numpy frame object each decoder update.
    last_raw_ref = raw
    now = time.monotonic()
    last_fresh_frame_at = now
    fps_frames += 1
    if now - fps_time_start >= 2.0:
        measured_fps = fps_frames / (now - fps_time_start)
        fps_frames = 0
        fps_time_start = now

    image = cv2.resize(raw, FRAME_SIZE, interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if DETECTOR is not None:
        corners, ids, _ = DETECTOR.detectMarkers(gray)
    else:
        corners, ids, _ = A.detectMarkers(gray, D, parameters=P)

    tag = None
    selected_corners = None
    if ids is not None:
        for i, this_id in enumerate(ids.flatten()):
            if int(this_id) != target_id:
                continue
            points = corners[i][0]
            width = (np.linalg.norm(points[0] - points[1]) +
                     np.linalg.norm(points[2] - points[3])) / 2.0
            if width < 2.0:
                continue
            x, y = np.mean(points, axis=0)
            distance = 100.0 * TAG_WIDTH_M * FOCAL_LENGTH_PX / width
            tag = (float(x), float(y), float(distance))
            selected_corners = corners[i]
            break

    if selected_corners is not None:
        A.drawDetectedMarkers(image, [selected_corners])
    cv2.rectangle(image, (CX - CENTER_X_TOL, CY - CENTER_Y_TOL),
                  (CX + CENTER_X_TOL, CY + CENTER_Y_TOL), (0, 255, 255), 2)
    cv2.drawMarker(image, (CX, CY), (255, 255, 0), cv2.MARKER_CROSS, 20, 1)
    cv2.putText(image, f'{status}  FPS:{measured_fps:.1f}', (12, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 255, 255), 2)
    if tag is not None:
        x, y, dist = tag
        cv2.putText(image, f'ID {target_id}  X:{x:.0f} Y:{y:.0f} DIST:{dist:.0f}cm',
                    (12, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.53, (0, 255, 0), 2)
    cv2.imshow(WINDOW, image)
    check_key()
    return tag, True


def yaw_heading():
    try:
        value = T.get_current_state().get('yaw')
        return None if value is None else float(value)
    except (TypeError, ValueError, AttributeError):
        return None


def scan_slowly(target_id, level):
    """Continuous slow yaw; immediately hand a detected tag to alignment."""
    print(f'SCAN ID {target_id}: {level}')
    stop_motion(force=True)
    start = time.monotonic()
    previous = yaw_heading()
    total_turn = 0.0
    last_real_turn = start
    has_turn_data = False
    try:
        while True:
            tag, fresh = read_tag(target_id, f'SCAN ID {target_id} / {level}')
            if tag is not None:
                print(f'FOUND ID {target_id}: stop scanning; center now')
                return tag

            now = time.monotonic()
            if now - start >= SCAN_LIMIT_SEC and video_age() < VIDEO_HOVER_AFTER_SEC:
                print('Scan timed out; estimated angle may be incomplete')
                return None

            if not check_video_stall():
                time.sleep(FRAME_LOOP_SEC)
                continue

            new_heading = yaw_heading()
            if new_heading is not None and previous is not None:
                delta = (new_heading - previous + 180) % 360 - 180
                if abs(delta) < 35:
                    total_turn += abs(delta)
                    if abs(delta) >= 0.1:
                        last_real_turn = now
                        has_turn_data = True
                previous = new_heading
            elif new_heading is not None:
                previous = new_heading

            if has_turn_data and total_turn >= 355:
                print('Completed approximately 360 degrees based on telemetry')
                return None
            if now - start >= SCAN_FALLBACK_SEC and now - last_real_turn > 2 and video_age() < VIDEO_HOVER_AFTER_SEC:
                print('Timed scan ended (not guaranteed to be 360 degrees)')
                return None

            T.send_rc_control(0, 0, 0, SCAN_YAW_SPEED)
            time.sleep(FRAME_LOOP_SEC)
    finally:
        stop_motion(force=True)


def search_heights(target_id):
    cycle = 0
    while True:
        cycle += 1
        print(f'ID {target_id}: search cycle {cycle} (no timeout)')
        tag = scan_slowly(target_id, 'START')
        if tag is not None:
            return tag

        T.move_up(SEARCH_STEP_CM)
        tag = scan_slowly(target_id, 'UP 30CM')
        if tag is not None:
            return tag

        T.move_down(SEARCH_STEP_CM)  # First down: original height
        try:
            altitude = T.get_height()
        except Exception as exc:
            print(f'Height unavailable; skip lower scan: {exc}')
            altitude = 0
        if altitude >= MIN_HEIGHT_CM + SEARCH_STEP_CM:
            T.move_down(SEARCH_STEP_CM)  # Second down: -30 cm from start
            tag = scan_slowly(target_id, 'DOWN 30CM')
            if tag is not None:
                return tag
            T.move_up(SEARCH_STEP_CM)  # Restore starting height
        else:
            print('Lower scan skipped: insufficient height')
    return None


def align_to_tag(target_id, target_dist_cm):
    count = 0
    stable_since = None
    loss_since = None
    seen = False
    last_log = 0.0
    print(f'ALIGN ID {target_id} at {target_dist_cm}cm')

    while True:
        tag, fresh = read_tag(target_id, f'ALIGN ID {target_id}')
        now = time.monotonic()

        if not fresh:
            # No repeated/stale video frame is used for alignment.
            time.sleep(FRAME_LOOP_SEC)
            continue

        if tag is None:
            stop_motion()
            count = 0
            stable_since = None
            if loss_since is None:
                loss_since = now
            if not seen or now - loss_since >= TAG_LOST_GRACE_SEC:
                print(f'ID {target_id} absent: scanning slowly')
                tag = search_heights(target_id)
                if tag is None:
                    continue  # Search indefinitely; no automatic landing
                seen = True
                # Scan already saw this frame, but centering will use the next new one.
                loss_since = None
                stop_motion()
            time.sleep(FRAME_LOOP_SEC)
            continue

        seen = True
        loss_since = None
        x, y, distance = tag
        ex, ey = x - CX, y - CY
        ed = distance - target_dist_cm
        ok_x = abs(ex) <= CENTER_X_TOL
        ok_y = abs(ey) <= CENTER_Y_TOL
        ok_d = abs(ed) <= DIST_TOL_CM

        if ok_x and ok_y and ok_d:
            stop_motion()
            if count == 0:
                stable_since = now
            count += 1
            if count >= STABLE_FRAMES and now - stable_since >= STABLE_SECONDS:
                stop_motion(force=True)
                print(f'ID {target_id} ACCEPTED: X={x:.0f}, Y={y:.0f}, dist={distance:.0f}cm')
                return
        else:
            count = 0
            stable_since = None
            lr = fb = ud = yaw = 0
            if abs(ex) > YAW_FIRST_PX:
                yaw = YAW_ALIGN_SPEED if ex > 0 else -YAW_ALIGN_SPEED
            elif not ok_x:
                lr = LR_SPEED if ex > 0 else -LR_SPEED
            if not ok_y:
                ud = UD_SPEED if ey < 0 else -UD_SPEED
                if ud < 0:
                    try:
                        if T.get_height() <= MIN_HEIGHT_CM:
                            ud = 0
                    except Exception:
                        ud = 0
            if ok_x and ok_y:
                if ed > DIST_TOL_CM:
                    fb = FB_SPEED
                elif ed < -DIST_TOL_CM:
                    fb = -FB_SPEED

            T.send_rc_control(lr, fb, ud, yaw)
            if now - last_log >= 1.0:
                print(f'ID {target_id}: dx={ex:+.0f}px dy={ey:+.0f}px '
                      f'dist={distance:.0f}cm RC=({lr},{fb},{ud},{yaw})')
                last_log = now
        time.sleep(FRAME_LOOP_SEC)


def wait_for_fresh_video(label):
    """Do not start a new scripted manoeuvre while video is stalled.

    This only verifies the decoder has provided a new frame; it cannot
    guarantee that a command sent over Wi-Fi will actually be received.
    """
    print(f'WAIT FOR VIDEO: {label}')
    stop_motion(force=True)
    while True:
        _tag, fresh = read_tag(-1, label)
        if fresh:
            return
        stop_motion()
        time.sleep(FRAME_LOOP_SEC)


def one_forward_pass(distance_cm):
    """Send ONE forward move, not repeated segments or a remaining-cm loop.

    DJI Tello may reply only after finishing its movement. Use a timeout
    long enough for distance/speed and do not resend if the reply is lost:
    the drone may have performed the manoeuvre despite the missing ACK.
    """
    distance_cm = int(distance_cm)
    if not (20 <= distance_cm <= 500):
        raise ValueError('Forward distance must be between 20 and 500 cm')
    speed = int(MISSION_SPEED_CM_S)
    if not (10 <= speed <= 100):
        raise ValueError('Forward speed must be between 10 and 100 cm/s')

    command = f'go {distance_cm} 0 0 {speed}'
    timeout = max(20.0, distance_cm / speed + 12.0)
    print(f'ONE FORWARD PASS: {distance_cm} cm at {speed} cm/s'
          f' (timeout {timeout:.1f}s; no retries)')
    stop_motion(force=True)
    # send_command_with_return sends the instruction once, without DJITelloPy's
    # automatic command retry loop. An error/timeout means position is UNKNOWN.
    reply = T.send_command_with_return(command, timeout=timeout)
    if str(reply).strip().lower() != 'ok':
        raise RuntimeError(
            f'Forward completion unconfirmed ({reply!r}). '
            'Do not proceed to yaw or the next gate.'
        )
    stop_motion(force=True)


def pass_gate(tag_id, align_distance_cm, forward_distance_cm):
    """For every gate: align -> up -> pass forward -> yaw right."""
    print(f'\n========== GATE {tag_id} =========')
    align_to_tag(tag_id, align_distance_cm)
    stop_motion(force=True)

    # A height change AFTER alignment changes the previously verified aperture
    # position: use only if the gate is large enough for +GATE_UP_CM.
    wait_for_fresh_video(f'GATE {tag_id}: READY TO GO UP')
    print(f'GATE {tag_id}: UP {GATE_UP_CM} cm')
    T.move_up(GATE_UP_CM)
    stop_motion(force=True)

    wait_for_fresh_video(f'GATE {tag_id}: READY TO PASS')
    one_forward_pass(forward_distance_cm)

    wait_for_fresh_video(f'GATE {tag_id}: READY TO TURN')
    print(f'GATE {tag_id}: YAW RIGHT {RIGHT_TURN_DEG} degrees')
    stop_motion(force=True)
    T.rotate_clockwise(RIGHT_TURN_DEG)
    stop_motion(force=True)
    print(f'GATE {tag_id} DONE: facing roughly toward next gate')


def try_video_setting(fn, value, label):
    try:
        fn(value)
        print(f'Video setting accepted: {label}')
    except Exception as exc:
        print(f'Video setting not accepted ({label}): {exc}')


def wait_for_q():
    """Remain airborne and stationary until the user presses Q / ESC."""
    while True:
        stop_motion()
        if reader is not None:
            read_tag(-1, 'MISSION FINISHED - Q TO LAND')
        else:
            show_waiting_frame(-1)
            check_key()
        time.sleep(0.15)


def mission():
    global reader, stream_started, last_fresh_frame_at
    T.connect()
    battery = T.get_battery()
    print(f'BATTERY: {battery}%')
    if battery < MIN_BATTERY:
        raise RuntimeError(f'Charge battery to at least {MIN_BATTERY}%')

    # These commands need SDK/firmware support. They do NOT set OpenCV processing FPS.
    try_video_setting(T.set_video_resolution, T.RESOLUTION_480P, '480p')
    camera_fps = T.FPS_5 if VIDEO_FPS == 5 else T.FPS_15
    try_video_setting(T.set_video_fps, camera_fps, f'{VIDEO_FPS} FPS')
    try_video_setting(T.set_video_bitrate, T.BITRATE_1MBPS, '1 Mbps')
    try:
        T.set_speed(MISSION_SPEED_CM_S)
        print(f'Requested default flight speed: {MISSION_SPEED_CM_S} cm/s')
    except Exception as exc:
        print(f'Default flight speed command unsupported: {exc}')

    T.streamon()
    stream_started = True
    reader = T.get_frame_read()
    last_fresh_frame_at = time.monotonic()
    time.sleep(1.5)

    print('AUTO TAKEOFF')
    T.takeoff()
    time.sleep(1.0)
    T.move_up(40)

    for tag_id, align_distance, forward_distance in GATES:
        pass_gate(tag_id, align_distance, forward_distance)

    print('MISSION COMPLETE: hovering until Q/ESC is pressed')
    wait_for_q()


if __name__ == '__main__':
    try:
        mission()
    except KeyboardInterrupt:
        print('USER ABORT: land requested')
    except Exception as exc:
        print(f'ERROR: {exc}. Hovering until Q/ESC; no automatic landing.')
        if T.is_flying:
            try:
                wait_for_q()
            except KeyboardInterrupt:
                print('USER REQUESTED LANDING')
    finally:
        if T.is_flying:
            try:
                stop_motion(force=True)
                print('LANDING')
                T.land()
            except Exception as exc:
                print(f'Unable to confirm landing: {exc}')
        if stream_started:
            try:
                T.streamoff()
            except Exception:
                pass
        cv2.destroyAllWindows()
        T.end()
