import io
import re
import base64
import streamlit as st
import google.generativeai as genai
import pandas as pd
import os
from reportlab.platypus import (
    SimpleDocTemplate, Spacer, Table, TableStyle, Paragraph, Image as RLImage
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors

# ---------------------------
# Gemini setup
# ---------------------------

api_key = st.secrets.get("GEMINI_API_KEY", os.getenv("GEMINI_API_KEY"))

if not api_key:
    st.error("Missing GEMINI_API_KEY. Add it in Streamlit Secrets.")
    st.stop()

genai.configure(api_key=api_key)
model = genai.GenerativeModel("gemini-2.5-flash")


# ---------------------------
# Data load
# ---------------------------

st.set_page_config(
    page_title="Blink Analysis - Eye Health Check",
    page_icon="👁️",
    layout="wide"
)

@st.cache_data
def load_data():
    try:
        df = pd.read_csv("countries.csv")
        
        expected = {"Country", "City", "Currency_Code", "Number"}
        missing = expected - set(df.columns)
        if missing:
            st.error(f"countries.csv is missing columns: {missing}")
            return pd.DataFrame(columns=["Country", "City", "Currency_Code", "Number"])

        df["Country"] = df["Country"].astype(str)
        df["City"] = df["City"].astype(str)
        df["Currency_Code"] = df["Currency_Code"].astype(str)
        df["Number"] = pd.to_numeric(df["Number"], errors="coerce")

        return df

    except FileNotFoundError:
        st.error("Error: 'countries.csv' file not found. Please check the path.")
        return pd.DataFrame(columns=["Country", "City", "Currency_Code", "Number"])

    except Exception as e:
        st.error(f"Failed to load countries.csv: {e}")
        return pd.DataFrame(columns=["Country", "City", "Currency_Code", "Number"])

df = load_data()

def get_countries():
    if not df.empty:
        return sorted(df["Country"].dropna().unique().tolist())
    return []

def get_cities(country: str):
    if not df.empty and country:
        return sorted(df[df["Country"] == country]["City"].dropna().unique().tolist())
    return []

def get_numbers_from_file():
    if df.empty or "Number" not in df.columns:
        return []
    nums = df["Number"].dropna().unique().tolist()
    nums = sorted({int(x) for x in nums if pd.notna(x)})
    return nums

# ---------------------------
# PDF generation
# ---------------------------

def generate_pdf_from_text_and_image(text_content: str, image_bytes: bytes | None = None):
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=letter,
        rightMargin=72,
        leftMargin=72,
        topMargin=72,
        bottomMargin=18
    )
    
    styles = getSampleStyleSheet()
    story = []

    title_style = ParagraphStyle(
        "CustomTitle",
        parent=styles["Heading1"],
        fontSize=16,
        textColor=colors.HexColor("#1a1a1a"),
        spaceAfter=14,
        leading=20
    )
    normal_style = ParagraphStyle(
        "CustomNormal",
        parent=styles["Normal"],
        fontSize=10,
        spaceAfter=6,
        leading=14
    )

    story.append(Paragraph("Eye Photo + Gemini Notes", title_style))
    story.append(Spacer(1, 10))

    if image_bytes:
        img_buf = io.BytesIO(image_bytes)
        rl_img = RLImage(img_buf)
        rl_img._restrictSize(440, 280)
        story.append(rl_img)
        story.append(Spacer(1, 14))

    lines = text_content.split("\n")
    i = 0
    while i < len(lines):
        stripped = lines[i].strip()

        if not stripped:
            story.append(Spacer(1, 8))
            i += 1
            continue

        if "|" in stripped and i + 1 < len(lines) and "|" in lines[i + 1]:
            table_data = []
            while i < len(lines) and "|" in lines[i].strip():
                row = lines[i].strip()

                if re.match(r"^[\|\s\-:]+$", row):
                    i += 1
                    continue

                cells = [cell.strip() for cell in row.split("|") if cell.strip() != ""]
                if cells:
                    table_data.append(cells)
                i += 1

            if table_data:
                t = Table(table_data, hAlign="CENTER")
                t.setStyle(TableStyle([
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#3498db")),
                    ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
                    ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                    ("FONTSIZE", (0, 0), (-1, 0), 10),
                    ("BOTTOMPADDING", (0, 0), (-1, 0), 10),
                    ("TOPPADDING", (0, 0), (-1, 0), 10),
                    ("BACKGROUND", (0, 1), (-1, -1), colors.whitesmoke),
                    ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                    ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
                    ("FONTSIZE", (0, 1), (-1, -1), 9),
                ]))
                story.append(t)
                story.append(Spacer(1, 12))
            continue

        story.append(Paragraph(stripped, normal_style))
        i += 1

    doc.build(story)
    buffer.seek(0)
    return buffer.getvalue()

# ---------------------------
# Webcam Component with Live Frame Preview
# ---------------------------

def webcam_with_hidden_upload():
    """
    Captures frames and creates a Blob, then programmatically uploads via hidden file input
    NOW WITH LIVE FRAME PREVIEW!
    """
    
    html_code = """
<!DOCTYPE html>
<html>
<head>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/jszip/3.10.1/jszip.min.js"></script>
</head>
<body>
    <div style="text-align: center;">
        <video id="video" width="640" height="480" autoplay style="border: 2px solid #3498db; border-radius: 8px;"></video>
        <br><br>
        
        <!-- NEW: Live preview canvas -->
        <canvas id="previewCanvas" width="320" height="240" style="border: 2px solid #27ae60; border-radius: 8px; display: none;"></canvas>
        <br><br>
        
        <button id="startBtn" style="padding: 10px 20px; font-size: 16px; background-color: #3498db; color: white; border: none; border-radius: 5px; cursor: pointer; margin: 5px;">
            Start Camera
        </button>
        <button id="captureBtn" style="padding: 10px 20px; font-size: 16px; background-color: #27ae60; color: white; border: none; border-radius: 5px; cursor: pointer; margin: 5px;" disabled>
            Capture & Upload 120 Frames
        </button>
        <canvas id="canvas" style="display: none;"></canvas>
        <p id="status" style="margin-top: 10px; font-size: 14px; color: #555;"></p>
        <p id="progress" style="margin-top: 5px; font-size: 14px; font-weight: bold; color: #3498db;"></p>
    </div>

    <script>
        const video = document.getElementById('video');
        const canvas = document.getElementById('canvas');
        const previewCanvas = document.getElementById('previewCanvas');
        const startBtn = document.getElementById('startBtn');
        const captureBtn = document.getElementById('captureBtn');
        const status = document.getElementById('status');
        const progress = document.getElementById('progress');
        const ctx = canvas.getContext('2d');
        const previewCtx = previewCanvas.getContext('2d');

        let stream = null;

        startBtn.onclick = async () => {
            try {
                status.textContent = 'Requesting camera access...';
                stream = await navigator.mediaDevices.getUserMedia({
                    video: { width: 640, height: 480 }
                });
                video.srcObject = stream;
                status.textContent = '✅ Camera active! Ready to capture.';
                captureBtn.disabled = false;
                startBtn.disabled = true;
            } catch (err) {
                status.textContent = '❌ Error: ' + err.message;
                console.error('Camera error:', err);
            }
        };

        captureBtn.onclick = async () => {
            if (!stream) {
                status.textContent = 'Please start the camera first!';
                return;
            }

            captureBtn.disabled = true;
            status.textContent = '📸 Capturing frames... Look at camera and blink normally.';
            
            // Show preview canvas
            previewCanvas.style.display = 'inline-block';

            canvas.width = video.videoWidth;
            canvas.height = video.videoHeight;

            const capturedFrames = [];

            // Capture 120 frames
            for (let i = 0; i < 120; i++) {
                ctx.drawImage(video, 0, 0);

                // NEW: Show current frame in preview (every 3rd frame to avoid lag)
                if (i % 3 === 0) {
                    previewCtx.drawImage(video, 0, 0, 320, 240);
                }

                // Convert to blob
                const blob = await new Promise(resolve => {
                    canvas.toBlob(resolve, 'image/jpeg', 0.85);
                });

                capturedFrames.push(blob);
                progress.textContent = `Captured ${i + 1}/120 frames`;
                await new Promise(resolve => setTimeout(resolve, 30));
            }

            status.textContent = '📦 Creating ZIP file...';
            progress.textContent = '';

            // Create ZIP
            const zip = new JSZip();
            for (let i = 0; i < capturedFrames.length; i++) {
                zip.file(`frame_${String(i).padStart(3, '0')}.jpg`, capturedFrames[i]);
            }

            const zipBlob = await zip.generateAsync({type: 'blob'});

            status.textContent = '📤 Uploading to Streamlit...';

            // Find Streamlit's file uploader in parent document
            const fileUploader = window.parent.document.querySelector('input[type="file"][accept=".zip"]');

            if (fileUploader) {
                // Create a File object from the blob
                const file = new File([zipBlob], 'captured_frames.zip', { type: 'application/zip' });

                // Create DataTransfer to set files
                const dataTransfer = new DataTransfer();
                dataTransfer.items.add(file);
                fileUploader.files = dataTransfer.files;

                // Trigger change event
                fileUploader.dispatchEvent(new Event('change', { bubbles: true }));

                status.textContent = '✅ Frames uploaded successfully!';
                progress.textContent = 'You can now analyze the frames below.';
                
                // Hide preview canvas after completion
                previewCanvas.style.display = 'none';
            } else {
                status.textContent = '❌ Could not find file uploader. Please refresh and try again.';
            }

            captureBtn.disabled = false;
        };
    </script>
</body>
</html>
"""
    
    st.components.v1.html(html_code, height=780)  # Increased height for preview canvas

# ---------------------------
# Main App
# ---------------------------

if st.button("← Back to Home"):
    st.switch_page("sci_fair.py")

st.image(
    "blink_logo.png",
    use_column_width=False,
    width=180
)

st.title("📸 Blink Analysis")
st.markdown("### AI-Powered Eye Health Assessment")

st.subheader("Step 1: Capture 120 frames")

# Initialize session state
if 'captured_frames' not in st.session_state:
    st.session_state.captured_frames = None

# Render webcam component
webcam_with_hidden_upload()

# Hidden file uploader (will be auto-filled by JavaScript)
uploaded_zip = st.file_uploader("", type=['zip'], key="auto_upload", label_visibility="collapsed")

# Process uploaded ZIP
if uploaded_zip is not None:
    import zipfile
    try:
        with zipfile.ZipFile(uploaded_zip, 'r') as zip_ref:
            frame_files = sorted([f for f in zip_ref.namelist() if f.endswith('.jpg')])
            
            if len(frame_files) < 1:
                st.error("No JPG files found in the ZIP!")
            else:
                frames_bytes = []
                for frame_file in frame_files:
                    with zip_ref.open(frame_file) as f:
                        frames_bytes.append(f.read())

                st.session_state.captured_frames = frames_bytes
                st.success(f"✅ Loaded {len(frames_bytes)} frames!")

                # Show first frame
                st.image(frames_bytes[0], caption=f"First frame (total: {len(frames_bytes)} frames)", use_column_width=True)

    except Exception as e:
        st.error(f"Error reading ZIP file: {e}")

st.write("---")

st.subheader("Step 2: Where are you from?")
patient_country = st.selectbox("Country:", get_countries(), key="h_country")
patient_city = st.selectbox("City:", get_cities(patient_country), key="h_city")

st.subheader("Step 3: Your Age")
numbers = get_numbers_from_file()
if not numbers:
    st.error("No numbers found in the 'Number' column in countries.csv.")
    st.stop()

age_num = st.selectbox("Age", numbers, key="an")

st.write("---")

if st.button("Step 4: 📊 Analyze Frames with AI", key="analyze_btn"):
    if st.session_state.captured_frames is None or len(st.session_state.captured_frames) == 0:
        st.error("⚠️ Please capture frames first using the button above!")
    else:
        frames = st.session_state.captured_frames
        
        try:
            st.image(frames[0], caption="Analyzing this frame and others...", use_column_width=True)
        except Exception as e:
            st.warning(f"Could not display preview image: {e}")

        prompt = f"""
You are given {len(frames)} sequential eye images (frames) from a webcam.
Task: Check for possible blinking problems or abnormal blinking patterns.

- You cannot diagnose.
- Give careful observations and safe advice only.
- Keep it short and focused.
- List urgent red flags that require an eye doctor.

Patient context:
- Country: {patient_country}
- City: {patient_city}
- Age: {age_num}
"""
        
        # Prepare content for Gemini
        contents = [prompt]
        for frame_bytes in frames:
            contents.append({"mime_type": "image/jpeg", "data": frame_bytes})
        
        with st.spinner(f"Analyzing {len(frames)} frames with Gemini AI..."):
            response = model.generate_content(contents)
        
        st.subheader("Analysis Results:")
        st.write(response.text)
        
        # Generate PDF
        pdf_content = generate_pdf_from_text_and_image(response.text, frames[0])
        
        if pdf_content:
            st.subheader("Step 5: Download your Report")
            st.download_button(
                label="Download PDF Report ⬇️",
                data=pdf_content,
                file_name="eye_health_recommendations.pdf",
                mime="application/pdf"
            )
