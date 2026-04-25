import streamlit as st
import cv2
import numpy as np
import tempfile
import os

# ─── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Lane Detection AI",
    page_icon="🛣️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&family=Rajdhani:wght@300;400;600&display=swap');

    * { box-sizing: border-box; }

    html, body, [data-testid="stAppViewContainer"] {
        background: #0a0a0f;
        color: #e0e0e0;
    }

    [data-testid="stSidebar"] {
        background: #0f0f1a !important;
        border-right: 1px solid #1a1a2e;
    }

    .main-title {
        font-family: 'Orbitron', monospace;
        font-size: 2.8rem;
        font-weight: 900;
        background: linear-gradient(90deg, #00ff88, #00ccff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        letter-spacing: 4px;
        margin-bottom: 0.2rem;
    }

    .sub-title {
        font-family: 'Rajdhani', sans-serif;
        font-size: 1rem;
        color: #444466;
        text-align: center;
        letter-spacing: 6px;
        text-transform: uppercase;
        margin-bottom: 2rem;
    }

    .mode-card {
        background: linear-gradient(135deg, #0f0f1a, #1a1a2e);
        border: 1px solid #1e1e3a;
        border-radius: 12px;
        padding: 1.5rem;
        margin-bottom: 1rem;
        transition: border-color 0.3s;
    }

    .mode-card:hover { border-color: #00ff88; }

    .stat-box {
        background: #0f0f1a;
        border: 1px solid #1e1e3a;
        border-radius: 8px;
        padding: 1rem;
        text-align: center;
        font-family: 'Orbitron', monospace;
    }

    .stat-number {
        font-size: 2rem;
        font-weight: 700;
        color: #00ff88;
    }

    .stat-label {
        font-size: 0.7rem;
        color: #555577;
        letter-spacing: 2px;
        text-transform: uppercase;
    }

    .section-header {
        font-family: 'Orbitron', monospace;
        font-size: 0.85rem;
        color: #00ff88;
        letter-spacing: 3px;
        text-transform: uppercase;
        border-bottom: 1px solid #1e1e3a;
        padding-bottom: 0.5rem;
        margin-bottom: 1rem;
    }

    .stButton > button {
        background: linear-gradient(90deg, #00ff88, #00ccff) !important;
        color: #000 !important;
        font-family: 'Orbitron', monospace !important;
        font-weight: 700 !important;
        font-size: 0.8rem !important;
        letter-spacing: 2px !important;
        border: none !important;
        border-radius: 6px !important;
        padding: 0.6rem 1.5rem !important;
        width: 100% !important;
        transition: opacity 0.2s !important;
    }

    .stButton > button:hover { opacity: 0.85 !important; }

    .stSlider label, .stRadio label, .stFileUploader label {
        font-family: 'Rajdhani', sans-serif !important;
        color: #aaaacc !important;
        font-size: 0.9rem !important;
        letter-spacing: 1px !important;
    }

    .lane-badge {
        display: inline-block;
        background: #00ff8822;
        border: 1px solid #00ff88;
        color: #00ff88;
        font-family: 'Orbitron', monospace;
        font-size: 0.7rem;
        padding: 0.2rem 0.7rem;
        border-radius: 20px;
        letter-spacing: 2px;
        margin-bottom: 1rem;
    }

    .video-badge {
        display: inline-block;
        background: #00ccff22;
        border: 1px solid #00ccff;
        color: #00ccff;
        font-family: 'Orbitron', monospace;
        font-size: 0.7rem;
        padding: 0.2rem 0.7rem;
        border-radius: 20px;
        letter-spacing: 2px;
        margin-bottom: 1rem;
    }

    .info-box {
        background: #001a0a;
        border: 1px solid #00ff88;
        border-radius: 8px;
        padding: 1rem;
        color: #00ff88;
        font-family: 'Rajdhani', sans-serif;
        font-size: 0.9rem;
        margin-bottom: 1rem;
    }

    .warning-box {
        background: #1a1000;
        border: 1px solid #ffaa00;
        border-radius: 8px;
        padding: 1rem;
        color: #ffaa00;
        font-family: 'Rajdhani', sans-serif;
        font-size: 0.9rem;
    }

    img { border-radius: 8px; }
    [data-testid="stImage"] { border-radius: 8px; overflow: hidden; }
    [data-testid="stCameraInput"] > div {
        border: 1px solid #1e1e3a !important;
        border-radius: 12px !important;
        background: #0f0f1a !important;
    }
</style>
""", unsafe_allow_html=True)


# ─── Core Processing ───────────────────────────────────────────────────────────
def process_frame(image, canny_low, canny_high, hough_threshold, min_line_length, max_line_gap):
    image = cv2.resize(image, (800, 500))
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, canny_low, canny_high)

    height, width = edges.shape
    mask = np.zeros_like(edges)
    polygon = np.array([[
        (0, height),
        (width, height),
        (width // 2, int(height * 0.6))
    ]], np.int32)
    cv2.fillPoly(mask, polygon, 255)
    cropped_edges = cv2.bitwise_and(edges, mask)

    lines = cv2.HoughLinesP(cropped_edges, 2, np.pi / 180, hough_threshold,
                            np.array([]), minLineLength=min_line_length, maxLineGap=max_line_gap)

    line_image = np.zeros_like(image)
    line_count = 0
    if lines is not None:
        line_count = len(lines)
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(line_image, (x1, y1), (x2, y2), (0, 255, 136), 5)

    combo = cv2.addWeighted(image, 0.8, line_image, 1, 1)
    return combo, cropped_edges, line_count


def bgr_to_rgb(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def show_result(original, result, edges, line_count):
    r1, r2, r3 = st.columns([2, 2, 1])
    with r1:
        st.caption("ORIGINAL")
        st.image(bgr_to_rgb(cv2.resize(original, (800, 500))), use_container_width=True)
    with r2:
        st.caption("LANE DETECTION")
        st.image(bgr_to_rgb(result), use_container_width=True)
    with r3:
        st.caption("EDGES")
        st.image(edges, use_container_width=True)
        st.markdown(f"""
        <div class="stat-box" style="margin-top:0.5rem">
            <div class="stat-number">{line_count}</div>
            <div class="stat-label">Lines Found</div>
        </div>
        """, unsafe_allow_html=True)


def process_video(video_path, canny_low, canny_high, hough_threshold, min_line_length, max_line_gap):
    cap = cv2.VideoCapture(video_path)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    duration = round(total_frames / fps, 1) if fps > 0 else 0

    # Output video setup
    out_path = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False).name
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(out_path, fourcc, fps, (800, 500))

    progress_bar = st.progress(0)
    status_text = st.empty()
    preview_slot = st.empty()

    frame_idx = 0
    total_lines_all = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        result, edges, line_count = process_frame(
            frame, canny_low, canny_high, hough_threshold, min_line_length, max_line_gap
        )
        total_lines_all += line_count
        out.write(result)

        # Update every 15 frames
        if frame_idx % 15 == 0:
            progress = int((frame_idx / max(total_frames, 1)) * 100)
            progress_bar.progress(progress)
            status_text.markdown(f"""
            <div style="font-family:'Rajdhani',sans-serif; color:#00ff88; font-size:0.85rem; letter-spacing:2px;">
                PROCESSING FRAME {frame_idx} / {total_frames} — {progress}%
            </div>
            """, unsafe_allow_html=True)
            preview_slot.image(bgr_to_rgb(result), caption=f"Preview — Frame {frame_idx}", use_container_width=True)

        frame_idx += 1

    cap.release()
    out.release()
    progress_bar.progress(100)
    status_text.markdown("""
    <div style="font-family:'Rajdhani',sans-serif; color:#00ff88; font-size:0.85rem; letter-spacing:2px;">
        ✅ PROCESSING COMPLETE
    </div>
    """, unsafe_allow_html=True)
    preview_slot.empty()

    return out_path, total_frames, fps, duration, total_lines_all


# ─── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="section-header">⚙ Detection Parameters</div>', unsafe_allow_html=True)

    canny_low    = st.slider("Canny Low Threshold",  10,  150,  50)
    canny_high   = st.slider("Canny High Threshold", 50,  300, 150)
    hough_threshold   = st.slider("Hough Threshold",      20,  200, 100)
    min_line_length   = st.slider("Min Line Length",       10,  150,  40)
    max_line_gap      = st.slider("Max Line Gap",           1,   50,   5)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-header">Mode:</div>', unsafe_allow_html=True)
    mode = st.radio("", ["🖼️  Image Upload", "🎥  Live Camera", "🎬  Video Upload"],
                    label_visibility="collapsed")

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
    <div style="font-family:'Rajdhani',sans-serif; color:#333355; font-size:0.75rem; letter-spacing:1px;">
        LANE DETECTION AI · CV PROJECT<br>
        Hough Transform · Canny Edge
    </div>
    """, unsafe_allow_html=True)


# ─── Header ───────────────────────────────────────────────────────────────────
st.markdown('<div class="main-title">LANE DETECT</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Computer Vision · Real-Time · Hough Transform</div>', unsafe_allow_html=True)


# ─── Image Mode ───────────────────────────────────────────────────────────────
if "🖼️" in mode:
    st.markdown('<div class="lane-badge">IMAGE MODE</div>', unsafe_allow_html=True)

    uploaded_files = st.file_uploader(
        "Upload up to 5 road images",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True
    )

    if uploaded_files:
        uploaded_files = uploaded_files[:5]
        total_lines = 0
        processed_count = 0

        col1, col2, col3 = st.columns(3)
        stat_placeholders = [col1.empty(), col2.empty(), col3.empty()]
        st.markdown("<br>", unsafe_allow_html=True)

        for i, uploaded_file in enumerate(uploaded_files):
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            if image is None:
                continue

            result, edges, line_count = process_frame(
                image, canny_low, canny_high, hough_threshold, min_line_length, max_line_gap
            )
            total_lines += line_count
            processed_count += 1

            st.markdown(f'<div class="section-header">IMAGE {i+1} — {uploaded_file.name}</div>', unsafe_allow_html=True)
            show_result(image, result, edges, line_count)
            st.markdown("<hr style='border-color:#1e1e3a; margin:1.5rem 0'>", unsafe_allow_html=True)

        with stat_placeholders[0]:
            st.markdown(f"""<div class="stat-box">
                <div class="stat-number">{processed_count}</div>
                <div class="stat-label">Images Processed</div>
            </div>""", unsafe_allow_html=True)
        with stat_placeholders[1]:
            st.markdown(f"""<div class="stat-box">
                <div class="stat-number">{total_lines}</div>
                <div class="stat-label">Total Lines</div>
            </div>""", unsafe_allow_html=True)
        with stat_placeholders[2]:
            avg = round(total_lines / processed_count, 1) if processed_count else 0
            st.markdown(f"""<div class="stat-box">
                <div class="stat-number">{avg}</div>
                <div class="stat-label">Avg Per Image</div>
            </div>""", unsafe_allow_html=True)

    else:
        st.markdown("""
        <div class="mode-card" style="text-align:center; padding:3rem;">
            <div style="font-size:3rem">🛣️</div>
            <div style="font-family:'Orbitron',monospace; color:#333355; font-size:0.9rem; letter-spacing:3px; margin-top:1rem">
                UPLOAD UP TO 5 IMAGES TO BEGIN
            </div>
        </div>
        """, unsafe_allow_html=True)


# ─── Camera Mode ──────────────────────────────────────────────────────────────
elif "🎥" in mode:
    st.markdown('<div class="lane-badge">LIVE CAMERA MODE</div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="info-box">
        📷 Allow camera access when your browser asks — point at a road and click capture!
    </div>
    """, unsafe_allow_html=True)

    camera_image = st.camera_input("Point your camera at a road and take a snapshot")

    if camera_image is not None:
        file_bytes = np.asarray(bytearray(camera_image.read()), dtype=np.uint8)
        frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        if frame is not None:
            result, edges, line_count = process_frame(
                frame, canny_low, canny_high, hough_threshold, min_line_length, max_line_gap
            )
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown('<div class="section-header">🔍 Detection Result</div>', unsafe_allow_html=True)
            show_result(frame, result, edges, line_count)
        else:
            st.markdown("""
            <div class="warning-box">⚠️ Could not process image. Please try capturing again.</div>
            """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="mode-card" style="text-align:center; padding:2rem;">
            <div style="font-size:3rem">📷</div>
            <div style="font-family:'Orbitron',monospace; color:#333355; font-size:0.9rem; letter-spacing:3px; margin-top:1rem">
                CAPTURE A FRAME TO DETECT LANES
            </div>
        </div>
        """, unsafe_allow_html=True)


# ─── Video Mode ───────────────────────────────────────────────────────────────
elif "🎬" in mode:
    st.markdown('<div class="video-badge">VIDEO MODE</div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="info-box">
        🎬 Upload a dashcam or road video — lane detection will be applied to every frame
        and you can download the processed video!
    </div>
    """, unsafe_allow_html=True)

    uploaded_video = st.file_uploader(
        "Upload a road video",
        type=["mp4", "avi", "mov", "mkv"]
    )

    if uploaded_video is not None:
        # Save uploaded video to temp file
        tmp_input = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
        tmp_input.write(uploaded_video.read())
        tmp_input.flush()
        tmp_input.close()

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="section-header">⚙ Processing Video...</div>', unsafe_allow_html=True)

        out_path, total_frames, fps, duration, total_lines = process_video(
            tmp_input.name,
            canny_low, canny_high, hough_threshold, min_line_length, max_line_gap
        )

        # Stats
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="section-header">📊 Video Stats</div>', unsafe_allow_html=True)

        s1, s2, s3, s4 = st.columns(4)
        with s1:
            st.markdown(f"""<div class="stat-box">
                <div class="stat-number">{total_frames}</div>
                <div class="stat-label">Total Frames</div>
            </div>""", unsafe_allow_html=True)
        with s2:
            st.markdown(f"""<div class="stat-box">
                <div class="stat-number">{round(fps)}</div>
                <div class="stat-label">FPS</div>
            </div>""", unsafe_allow_html=True)
        with s3:
            st.markdown(f"""<div class="stat-box">
                <div class="stat-number">{duration}s</div>
                <div class="stat-label">Duration</div>
            </div>""", unsafe_allow_html=True)
        with s4:
            st.markdown(f"""<div class="stat-box">
                <div class="stat-number">{total_lines}</div>
                <div class="stat-label">Lines Detected</div>
            </div>""", unsafe_allow_html=True)

        # Play processed video
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="section-header">🎬 Processed Video</div>', unsafe_allow_html=True)
        st.video(out_path)

        # Download button
        with open(out_path, "rb") as f:
            st.download_button(
                label="⬇ DOWNLOAD PROCESSED VIDEO",
                data=f,
                file_name="lane_detection_output.mp4",
                mime="video/mp4"
            )

        # Cleanup temp files
        os.unlink(tmp_input.name)

    else:
        st.markdown("""
        <div class="mode-card" style="text-align:center; padding:3rem;">
            <div style="font-size:3rem">🎬</div>
            <div style="font-family:'Orbitron',monospace; color:#333355; font-size:0.9rem; letter-spacing:3px; margin-top:1rem">
                UPLOAD A ROAD VIDEO TO BEGIN
            </div>
        </div>
        """, unsafe_allow_html=True)
