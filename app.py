import streamlit as st
import cv2
import numpy as np
import tempfile
import os
import urllib.request
import subprocess

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


# ─── FFmpeg Re-encoder (mp4v → H.264 so browsers can play it) ─────────────────
def reencode_for_browser(input_path: str) -> str:
    """Re-encode mp4v → H.264/yuv420p so every browser can play the output."""
    output_path = input_path.replace(".mp4", "_h264.mp4")
    subprocess.run([
        "ffmpeg", "-y", "-i", input_path,
        "-vcodec", "libx264",
        "-crf", "23",
        "-preset", "ultrafast",
        "-pix_fmt", "yuv420p",
        output_path
    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return output_path


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
    width = image.shape[1]
    mid   = width // 2

    for line in lines:
        x1, y1, x2, y2 = line[0]
        if x1 == x2:
            continue
        slope = (y2 - y1) / (x2 - x1)
        intercept = y1 - slope * x1
        angle = abs(np.degrees(np.arctan(slope)))
        if angle < 25 or angle > 85:
            continue
        # Positional guard: segment midpoint must be on the correct side
        seg_mid_x = (x1 + x2) / 2
        if slope < 0 and seg_mid_x < mid:        # left line → must be left of centre
            left_fit.append((slope, intercept))
        elif slope > 0 and seg_mid_x > mid:      # right line → must be right of centre
            right_fit.append((slope, intercept))

    left_line  = make_coords(image, np.mean(left_fit,  axis=0)) if left_fit  else None
    right_line = make_coords(image, np.mean(right_fit, axis=0)) if right_fit else None

    # Final sanity check: left line bottom x must be left of right line bottom x
    if left_line is not None and right_line is not None:
        if left_line[0] >= right_line[0]:         # lines have crossed — discard both
            return None, None

    return left_line, right_line

def process_frame(image, canny_low, canny_high, hough_threshold, min_line_length, max_line_gap, fast_mode=False):
    image = cv2.resize(image, (800, 500))
    height, width = image.shape[:2]

    # ── In fast_mode: detect on half-res, draw on full-res ────────────────────
    detect_img = cv2.resize(image, (400, 250)) if fast_mode else image
    dh, dw = detect_img.shape[:2]
    scale  = 2.0 if fast_mode else 1.0

    # ── Night-vision boost: CLAHE on L channel ─────────────────────────────────
    lab = cv2.cvtColor(detect_img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    l = clahe.apply(l)
    enhanced = cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)

    # ── Blur ───────────────────────────────────────────────────────────────────
    gray = cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY)
    if fast_mode:
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
    else:
        blur = cv2.bilateralFilter(gray, 9, 75, 75)

    # ── Canny edges ────────────────────────────────────────────────────────────
    edges = cv2.Canny(blur, canny_low, canny_high)

    # ── Trapezoid ROI ──────────────────────────────────────────────────────────
    mask = np.zeros_like(edges)
    polygon = np.array([[
        (int(dw * 0.05), dh),
        (int(dw * 0.95), dh),
        (int(dw * 0.58), int(dh * 0.58)),
        (int(dw * 0.42), int(dh * 0.58)),
    ]], np.int32)
    cv2.fillPoly(mask, polygon, 255)
    cropped_edges = cv2.bitwise_and(edges, mask)

    # ── Hough transform ────────────────────────────────────────────────────────
    lines = cv2.HoughLinesP(
        cropped_edges, 1, np.pi / 180,
        threshold=max(10, hough_threshold // (2 if fast_mode else 1)),
        minLineLength=max(5,  min_line_length // (2 if fast_mode else 1)),
        maxLineGap=max_line_gap
    )

    # ── Scale lines back to full resolution if needed ──────────────────────────
    if fast_mode and lines is not None:
        lines = (lines * scale).astype(np.int32)

    # ── Average into 2 solid lane lines + smooth ───────────────────────────────
    left_line, right_line = average_lines(image, lines)
    lane_smoother.add(left_line, right_line)
    left_line, right_line = lane_smoother.get()

    # ── Draw precise lines only — no polygon fill ──────────────────────────────
    line_image = np.zeros_like(image)
    line_count = 0

    for lane in [left_line, right_line]:
        if lane is not None:
            x1, y1, x2, y2 = lane
            # Outer glow for visibility
            cv2.line(line_image, (x1, y1), (x2, y2), (0, 180, 80), 14)
            # Sharp bright centre line
            cv2.line(line_image, (x1, y1), (x2, y2), (0, 255, 136), 5)
            line_count += 1

    # edges display: upscale back for the UI
    display_edges = cv2.resize(cropped_edges, (width, height)) if fast_mode else cropped_edges
    combo = cv2.addWeighted(image, 0.85, line_image, 0.9, 0)
    return combo, display_edges, line_count


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


# ─── Sample Image Loader ───────────────────────────────────────────────────────
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
        🎬 Upload a dashcam or road video — green lane lines are drawn precisely on the
        detected lane markings, then download the result!
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

        lane_smoother.__init__()

        cap          = cv2.VideoCapture(tmp_input.name)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps          = cap.get(cv2.CAP_PROP_FPS) or 24
        duration     = round(total_frames / fps, 1)

        # Output file (mp4v — will be re-encoded to H.264 after processing)
        out_path = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False).name
        fourcc   = cv2.VideoWriter_fourcc(*"mp4v")
        out      = cv2.VideoWriter(out_path, fourcc, fps, (800, 500))

        # ── UI layout ──────────────────────────────────────────────────────────
        st.markdown('<div class="section-header">🎬 Live Lane Detection</div>', unsafe_allow_html=True)

        vid_col, stat_col = st.columns([3, 1])

        with vid_col:
            live_frame = st.empty()

        with stat_col:
            prog_bar     = st.progress(0)
            status_box   = st.empty()
            stat_frames  = st.empty()
            stat_fps_box = st.empty()
            stat_lines   = st.empty()

        # ── Process & stream frames live ───────────────────────────────────────
        TARGET_FPS      = 10
        frame_skip      = max(1, int(round(fps / TARGET_FPS)))
        frame_idx       = 0
        total_lines_all = 0
        last_result     = None

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % frame_skip == 0:
                result, _, line_count = process_frame(
                    frame, canny_low, canny_high, hough_threshold,
                    min_line_length, max_line_gap, fast_mode=True
                )
                total_lines_all += line_count
                last_result = result
            else:
                result = last_result if last_result is not None else cv2.resize(frame, (800, 500))

            out.write(result)

            if frame_idx % frame_skip == 0:
                progress = int((frame_idx / max(total_frames, 1)) * 100)
                eta      = round((total_frames - frame_idx) / fps)

                live_frame.image(
                    bgr_to_rgb(result),
                    use_container_width=True,
                    caption=f"Frame {frame_idx} / {total_frames}"
                )

                prog_bar.progress(progress)
                status_box.markdown(f"""
                <div style="font-family:'Rajdhani',sans-serif;color:#00ff88;
                            font-size:0.8rem;letter-spacing:2px;margin-bottom:0.5rem">
                    {'✅ DONE' if progress==100 else f'⚡ {progress}% — ETA {eta}s'}
                </div>""", unsafe_allow_html=True)
                stat_frames.markdown(f"""<div class="stat-box" style="margin-bottom:0.5rem">
                    <div class="stat-number" style="font-size:1.2rem">{frame_idx}</div>
                    <div class="stat-label">Frame</div>
                </div>""", unsafe_allow_html=True)
                stat_fps_box.markdown(f"""<div class="stat-box" style="margin-bottom:0.5rem">
                    <div class="stat-number" style="font-size:1.2rem">{round(fps)}</div>
                    <div class="stat-label">FPS</div>
                </div>""", unsafe_allow_html=True)
                stat_lines.markdown(f"""<div class="stat-box">
                    <div class="stat-number" style="font-size:1.2rem">{total_lines_all}</div>
                    <div class="stat-label">Lines Found</div>
                </div>""", unsafe_allow_html=True)

            frame_idx += 1

        cap.release()
        out.release()

        # ── Re-encode to H.264 so the browser can play it ─────────────────────
        with st.spinner("⚙️ Encoding video for browser playback..."):
            browser_path = reencode_for_browser(out_path)
            try:
                os.unlink(out_path)
            except Exception:
                pass

        # ── Final stats row ────────────────────────────────────────────────────
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
                <div class="stat-number">{total_lines_all}</div>
                <div class="stat-label">Lines Detected</div>
            </div>""", unsafe_allow_html=True)

        # ── Playback + download ────────────────────────────────────────────────
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="section-header">▶ Full Processed Video — Play & Download</div>', unsafe_allow_html=True)
        st.video(browser_path)

        with open(browser_path, "rb") as f:
            st.download_button(
                label="⬇ DOWNLOAD PROCESSED VIDEO",
                data=f,
                file_name="lane_detection_output.mp4",
                mime="video/mp4"
            )

        # Cleanup temp files
        try:
            os.unlink(tmp_input.name)
            os.unlink(browser_path)
        except Exception:
            pass

    else:
        st.markdown("""
        <div class="mode-card" style="text-align:center; padding:3rem;">
            <div style="font-size:3rem">🎬</div>
            <div style="font-family:'Orbitron',monospace; color:#333355; font-size:0.9rem; letter-spacing:3px; margin-top:1rem">
                UPLOAD A ROAD VIDEO TO BEGIN
            </div>
        </div>
        """, unsafe_allow_html=True)
