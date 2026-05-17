import streamlit as st
import cv2
import numpy as np
import tempfile
import os
import urllib.request

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


# ─── Lane Smoother for Video (keeps lines stable across frames) ────────────────
class LaneSmoother:
    def __init__(self, smooth_frames=8):
        self.smooth_frames = smooth_frames
        self.left_lines  = []
        self.right_lines = []

    def add(self, left, right):
        if left is not None:
            self.left_lines.append(left)
            if len(self.left_lines) > self.smooth_frames:
                self.left_lines.pop(0)
        if right is not None:
            self.right_lines.append(right)
            if len(self.right_lines) > self.smooth_frames:
                self.right_lines.pop(0)

    def get(self):
        left  = np.mean(self.left_lines,  axis=0).astype(int) if self.left_lines  else None
        right = np.mean(self.right_lines, axis=0).astype(int) if self.right_lines else None
        return left, right

lane_smoother = LaneSmoother()


# ─── Core Processing ───────────────────────────────────────────────────────────
def make_coords(image, line_params):
    """Convert slope/intercept to full lane segment coordinates."""
    slope, intercept = line_params
    height = image.shape[0]
    y1 = height
    y2 = int(height * 0.58)
    if abs(slope) < 1e-6:
        return None
    x1 = int((y1 - intercept) / slope)
    x2 = int((y2 - intercept) / slope)
    return np.array([x1, y1, x2, y2])


def average_lines(image, lines):
    """Separate Hough segments into one averaged left + one right lane line."""
    left_fit, right_fit = [], []
    if lines is None:
        return None, None
    for line in lines:
        x1, y1, x2, y2 = line[0]
        if x1 == x2:
            continue
        slope = (y2 - y1) / (x2 - x1)
        intercept = y1 - slope * x1
        angle = abs(np.degrees(np.arctan(slope)))
        if angle < 25 or angle > 85:   # ignore near-horizontal / near-vertical noise
            continue
        if slope < 0:
            left_fit.append((slope, intercept))
        else:
            right_fit.append((slope, intercept))
    left_line  = make_coords(image, np.mean(left_fit,  axis=0)) if left_fit  else None
    right_line = make_coords(image, np.mean(right_fit, axis=0)) if right_fit else None
    return left_line, right_line


def process_frame(image, canny_low, canny_high, hough_threshold, min_line_length, max_line_gap):
    image = cv2.resize(image, (800, 500))
    height, width = image.shape[:2]

    # ── Night-vision boost: CLAHE on L channel ─────────────────────────────────
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    l = clahe.apply(l)
    enhanced = cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)

    # ── Bilateral filter: removes noise, preserves lane edges ──────────────────
    gray = cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY)
    blur = cv2.bilateralFilter(gray, 9, 75, 75)

    # ── Canny edges ────────────────────────────────────────────────────────────
    edges = cv2.Canny(blur, canny_low, canny_high)

    # ── Trapezoid ROI (better than old triangle) ───────────────────────────────
    mask = np.zeros_like(edges)
    polygon = np.array([[
        (int(width * 0.05), height),
        (int(width * 0.95), height),
        (int(width * 0.58), int(height * 0.58)),
        (int(width * 0.42), int(height * 0.58)),
    ]], np.int32)
    cv2.fillPoly(mask, polygon, 255)
    cropped_edges = cv2.bitwise_and(edges, mask)

    # ── Hough transform ────────────────────────────────────────────────────────
    lines = cv2.HoughLinesP(
        cropped_edges, 1, np.pi / 180,
        threshold=hough_threshold,
        minLineLength=min_line_length,
        maxLineGap=max_line_gap
    )

    # ── Average into 2 solid lane lines + smooth across video frames ───────────
    left_line, right_line = average_lines(image, lines)
    lane_smoother.add(left_line, right_line)
    left_line, right_line = lane_smoother.get()

    # ── Draw lanes ─────────────────────────────────────────────────────────────
    line_image = np.zeros_like(image)
    line_count = 0

    # Green tint fill between the two lanes
    if left_line is not None and right_line is not None:
        lane_poly = np.array([[
            (left_line[0],  left_line[1]),
            (left_line[2],  left_line[3]),
            (right_line[2], right_line[3]),
            (right_line[0], right_line[1]),
        ]], np.int32)
        cv2.fillPoly(line_image, lane_poly, (0, 80, 0))

    for lane in [left_line, right_line]:
        if lane is not None:
            x1, y1, x2, y2 = lane
            cv2.line(line_image, (x1, y1), (x2, y2), (0, 255, 136), 12)   # thick green
            cv2.line(line_image, (x1, y1), (x2, y2), (255, 255, 255),  3)  # thin white centre
            line_count += 1

    combo = cv2.addWeighted(image, 0.85, line_image, 0.6, 0)
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
    fps          = cap.get(cv2.CAP_PROP_FPS)
    duration     = round(total_frames / fps, 1) if fps > 0 else 0

    # ── Speed optimisation: cap processing at 15 fps max ──────────────────────
    # If source is 24/30 fps we skip every other frame → 2x faster
    # Output still plays at original fps (duplicating skipped frames)
    TARGET_PROCESS_FPS = 15
    frame_skip = max(1, int(round(fps / TARGET_PROCESS_FPS)))

    out_path = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False).name
    fourcc   = cv2.VideoWriter_fourcc(*"mp4v")
    out      = cv2.VideoWriter(out_path, fourcc, fps, (800, 500))

    progress_bar = st.progress(0)
    status_text  = st.empty()
    preview_slot = st.empty()

    frame_idx       = 0
    total_lines_all = 0
    last_result     = None   # reuse for skipped frames

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % frame_skip == 0:
            # process this frame
            result, _, line_count = process_frame(
                frame, canny_low, canny_high, hough_threshold, min_line_length, max_line_gap
            )
            total_lines_all += line_count
            last_result = result
        else:
            # reuse last processed result for skipped frames (saves ~50% time)
            result = last_result if last_result is not None else cv2.resize(frame, (800, 500))

        out.write(result)

        # Update UI every 30 frames
        if frame_idx % 30 == 0:
            progress = int((frame_idx / max(total_frames, 1)) * 100)
            progress_bar.progress(progress)
            elapsed_frames = frame_idx + 1
            eta_frames = total_frames - elapsed_frames
            eta_sec = round(eta_frames / fps) if fps > 0 else 0
            status_text.markdown(f"""
            <div style="font-family:'Rajdhani',sans-serif; color:#00ff88; font-size:0.85rem; letter-spacing:2px;">
                ⚡ PROCESSING FRAME {frame_idx} / {total_frames} — {progress}% — ETA {eta_sec}s
            </div>
            """, unsafe_allow_html=True)
            preview_slot.image(bgr_to_rgb(result), caption=f"Live Preview — Frame {frame_idx}", use_container_width=True)

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


# ─── Sample Image Loader ───────────────────────────────────────────────────────
# Replace these URLs with your actual raw GitHub URLs after uploading the images
# Format: https://raw.githubusercontent.com/YOUR_USERNAME/YOUR_REPO/main/samples/image1.jpg
SAMPLE_IMAGES = [
    {
        "name": "Solid White Curve",
        "url": "https://raw.githubusercontent.com/Theprogramergt/computervision/main/tusimple_images/solidWhiteCurve.jpg"
    },
    {
        "name": "Solid White Right",
        "url": "https://raw.githubusercontent.com/Theprogramergt/computervision/main/tusimple_images/solidWhiteRight.jpg"
    },
    {
        "name": "Solid Yellow Left",
        "url": "https://raw.githubusercontent.com/Theprogramergt/computervision/main/tusimple_images/solidYellowLeft.jpg"
    },
]


@st.cache_data(show_spinner=False)
def load_image_from_url(url: str):
    """Download image from URL and decode into a BGR numpy array. Returns None on failure."""
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = np.frombuffer(resp.read(), dtype=np.uint8)
        img = cv2.imdecode(data, cv2.IMREAD_COLOR)
        return img
    except Exception:
        return None


# ─── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="section-header">⚙ Detection Parameters</div>', unsafe_allow_html=True)

    canny_low         = st.slider("Canny Low Threshold",   10,  150,  50)
    canny_high        = st.slider("Canny High Threshold",  50,  300, 150)
    hough_threshold   = st.slider("Hough Threshold",       20,  200,  50)
    min_line_length   = st.slider("Min Line Length",       10,  200,  80)
    max_line_gap      = st.slider("Max Line Gap",           1,  100,  50)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-header">Mode:</div>', unsafe_allow_html=True)
    mode = st.radio("Select Mode", ["🖼️  Image Upload", "🎥  Live Camera", "🎬  Video Upload"],
                    label_visibility="hidden")

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
        # ── User uploaded their own images ──
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
        # ── No upload yet → show sample images ──
        st.markdown("""
        <div class="info-box">
            🛣️ No image uploaded yet — try the <strong>3 sample road images</strong> below,
            or upload your own above!
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<div class="section-header">🔬 Sample Images — Try Immediately</div>', unsafe_allow_html=True)

        total_lines = 0
        processed_count = 0

        for i, sample in enumerate(SAMPLE_IMAGES):
            with st.spinner(f"Loading sample: {sample['name']}..."):
                image = load_image_from_url(sample["url"])

            if image is None:
                st.markdown(f"""
                <div class="warning-box">
                    ⚠️ Could not load sample "{sample['name']}". 
                    Check that the GitHub URL is correct and the image is publicly accessible.
                </div>
                """, unsafe_allow_html=True)
                continue

            result, edges, line_count = process_frame(
                image, canny_low, canny_high, hough_threshold, min_line_length, max_line_gap
            )
            total_lines += line_count
            processed_count += 1

            st.markdown(f'<div class="section-header">SAMPLE {i+1} — {sample["name"]}</div>', unsafe_allow_html=True)
            show_result(image, result, edges, line_count)
            st.markdown("<hr style='border-color:#1e1e3a; margin:1.5rem 0'>", unsafe_allow_html=True)

        if processed_count > 0:
            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown(f"""<div class="stat-box">
                    <div class="stat-number">{processed_count}</div>
                    <div class="stat-label">Samples Shown</div>
                </div>""", unsafe_allow_html=True)
            with col2:
                st.markdown(f"""<div class="stat-box">
                    <div class="stat-number">{total_lines}</div>
                    <div class="stat-label">Total Lines</div>
                </div>""", unsafe_allow_html=True)
            with col3:
                avg = round(total_lines / processed_count, 1) if processed_count else 0
                st.markdown(f"""<div class="stat-box">
                    <div class="stat-number">{avg}</div>
                    <div class="stat-label">Avg Per Sample</div>
                </div>""", unsafe_allow_html=True)


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

        lane_smoother.__init__()  # reset smoother for each new video

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
