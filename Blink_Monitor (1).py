import streamlit as st
import cv2
import mediapipe as mp
import time
import numpy as np

st.set_page_config(page_title="Blink Monitor - Smart Tracking", layout="centered")

# ------------------ Session State Initialization ------------------
if 'blink_count' not in st.session_state:
    st.session_state.blink_count = 0
if 'eyes_closed' not in st.session_state:
    st.session_state.eyes_closed = False
if 'open_eye_reference' not in st.session_state:
    st.session_state.open_eye_reference = None
if 'minute_start' not in st.session_state:
    st.session_state.minute_start = time.time()
if 'start_time' not in st.session_state:
    st.session_state.start_time = time.time()
if 'show_reminder' not in st.session_state:
    st.session_state.show_reminder = False
if 'reminder_start' not in st.session_state:
    st.session_state.reminder_start = 0
if 'camera_active' not in st.session_state:
    st.session_state.camera_active = False

# ------------------ Constants ------------------
BLINK_RATIO = 0.4
TOTAL_TIME = 5 * 60  # 5 minutes
REMINDER_DURATION = 10  # seconds
NORMAL_MAX = 20

# ------------------ MediaPipe Setup ------------------
mp_face = mp.solutions.face_mesh

# ------------------ Streamlit UI ------------------

if st.button("← Back to Home"):
    st.switch_page("sci_fair.py")

st.title("👁️ Blink Monitor")
st.markdown("### Monitor your blink rate to reduce eye strain")

col1, col2, col3 = st.columns([1, 1, 1])

with col1:
    st.metric("Current Blinks", st.session_state.blink_count)

with col2:
    total_elapsed = time.time() - st.session_state.start_time
    remaining = TOTAL_TIME - int(total_elapsed)
    minutes_left = remaining // 60
    seconds_left = remaining % 60
    st.metric("Time Remaining", f"{minutes_left}:{seconds_left:02d}")

with col3:
    st.metric("Target", f"{NORMAL_MAX} blinks/min")

st.markdown("---")

# ------------------ Control Buttons ------------------
col_btn1, col_btn2 = st.columns(2)

with col_btn1:
    if st.button("🎥 Start Camera", disabled=st.session_state.camera_active, type="primary"):
        st.session_state.camera_active = True
        st.session_state.blink_count = 0
        st.session_state.eyes_closed = False
        st.session_state.open_eye_reference = None
        st.session_state.minute_start = time.time()
        st.session_state.start_time = time.time()
        st.session_state.show_reminder = False
        st.rerun()

with col_btn2:
    if st.button("🛑 Stop Camera", disabled=not st.session_state.camera_active, type="secondary"):
        st.session_state.camera_active = False
        st.rerun()

# ------------------ Camera Feed ------------------
if st.session_state.camera_active:
    # Create placeholder for video
    video_placeholder = st.empty()
    
    # Open camera
    camera = cv2.VideoCapture(0)
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    # Initialize face mesh
    face_mesh = mp_face.FaceMesh(refine_landmarks=True)
    
    # Stop button for real-time control
    stop_button = st.button("⏹️ Stop", key="stop_realtime")
    
    while st.session_state.camera_active and not stop_button:
        success, frame = camera.read()
        if not success:
            st.error("Failed to access camera")
            break
        
        frame = cv2.flip(frame, 1)
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(image)
        
        # ------------------ Blink Detection ------------------
        if results.multi_face_landmarks:
            face = results.multi_face_landmarks[0]
            
            top = face.landmark[159]
            bottom = face.landmark[145]
            
            eye_opening = abs(top.y - bottom.y)
            
            if st.session_state.open_eye_reference is None or eye_opening > st.session_state.open_eye_reference:
                st.session_state.open_eye_reference = eye_opening
            
            if st.session_state.open_eye_reference:
                if eye_opening < st.session_state.open_eye_reference * BLINK_RATIO:
                    if not st.session_state.eyes_closed:
                        st.session_state.blink_count += 1
                        st.session_state.eyes_closed = True
                else:
                    st.session_state.eyes_closed = False
        
        # ------------------ Minute Handling ------------------
        if time.time() - st.session_state.minute_start >= 60:
            if st.session_state.blink_count < NORMAL_MAX:
                st.session_state.show_reminder = True
                st.session_state.reminder_start = time.time()
            
            st.session_state.blink_count = 0
            st.session_state.minute_start = time.time()
        
        # ------------------ Total Timer ------------------
        total_elapsed = time.time() - st.session_state.start_time
        if total_elapsed >= TOTAL_TIME:
            st.session_state.camera_active = False
            st.success("✅ Session completed! Great job!")
            break
        
        remaining = TOTAL_TIME - int(total_elapsed)
        minutes_left = remaining // 60
        seconds_left = remaining % 60
        
        # ------------------ Display on Frame ------------------
        cv2.putText(
            frame,
            f"Blinks/min: {st.session_state.blink_count} / 20",
            (10, frame.shape[0] - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2
        )
        
        cv2.putText(
            frame,
            f"Time left: {minutes_left}:{seconds_left:02d}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (200, 200, 200),
            2
        )
        
        # ------------------ Gentle Reminder ------------------
        if st.session_state.show_reminder:
            if time.time() - st.session_state.reminder_start <= REMINDER_DURATION:
                cx = frame.shape[1] - 60
                cy = 50
                
                # Open eye
                cv2.ellipse(frame, (cx, cy), (22, 11), 0, 0, 360, (255, 255, 255), 2)
                
                # Blinking pupil
                blink_phase = int(time.time() * 2) % 2
                if blink_phase == 0:
                    cv2.circle(frame, (cx, cy), 3, (255, 255, 255), -1)
                
                cv2.putText(
                    frame,
                    "Blink",
                    (cx - 20, cy + 25),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    2
                )
            else:
                st.session_state.show_reminder = False
        
        # Convert BGR to RGB for Streamlit
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Display frame with controlled size
        video_placeholder.image(frame_rgb, channels="RGB", width=640, caption="Your Video Feed")
        
        time.sleep(0.03)  # ~30 fps
    
    camera.release()
    face_mesh.close()
    
    if stop_button:
        st.session_state.camera_active = False
        st.rerun()

else:
    st.info("👆 Click 'Start Camera' to begin monitoring your blink rate")
    st.markdown("""
    **How it works:**
    - The app tracks your blinks per minute in real-time
    - Aim for at least 20 blinks per minute to keep your eyes healthy
    - You'll see a gentle reminder (blinking eye icon) if you blink too little
    - Session duration: 5 minutes
    """)

# ------------------ Footer ------------------
st.markdown("---")
st.markdown("💡 **Tip**: Remember to take breaks and blink regularly to prevent eye strain!")
