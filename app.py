import streamlit as st
import cv2
import numpy as np
import tempfile

from PIL import Image
from transformers import pipeline


# =========================================================
# PAGE SETUP
# =========================================================

st.set_page_config(
    page_title="The Deepfake Detective Kit",
    page_icon="🔍",
    layout="wide"
)


# =========================================================
# WEBSITE DESIGN
# =========================================================

st.markdown(
    """
    <style>

    .stApp {
        background-color: #070b14;
        color: white;
    }

    .main-title {
        text-align: center;
        font-size: 52px;
        font-weight: 900;
        letter-spacing: 2px;
        color: #ffffff;
        margin-bottom: 5px;
    }

    .subtitle {
        text-align: center;
        font-size: 20px;
        color: #70d6ff;
        letter-spacing: 3px;
    }

    .tagline {
        text-align: center;
        color: #a7b0c0;
        margin-top: 10px;
        margin-bottom: 40px;
    }

    .result-box {
        padding: 25px;
        border: 1px solid #24344d;
        border-radius: 15px;
        background-color: #0c1422;
        margin-top: 20px;
    }

    </style>
    """,
    unsafe_allow_html=True
)


st.markdown(
    """
    <div class="main-title">
    THE DEEPFAKE DETECTIVE KIT
    </div>

    <div class="subtitle">
    AI + HUMAN MEDIA VERIFICATION
    </div>

    <div class="tagline">
    DETECT • INSPECT • VERIFY
    </div>
    """,
    unsafe_allow_html=True
)


# =========================================================
# LOAD AI
# =========================================================

MODEL_NAME = "prithivMLmods/deepfake-detector-model-v1"


@st.cache_resource
def load_detector():

    return pipeline(
        task="image-classification",
        model=MODEL_NAME
    )


with st.spinner("Loading AI detection model..."):

    detector = load_detector()


# =========================================================
# AI FUNCTIONS
# =========================================================

def analyze_frame(frame):

    rgb_frame = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )

    image = Image.fromarray(rgb_frame)

    predictions = detector(image)

    fake_score = 0.0
    real_score = 0.0

    for prediction in predictions:

        label = prediction["label"].lower()
        score = float(prediction["score"])

        if "fake" in label:

            fake_score = score

        elif "real" in label:

            real_score = score

    return fake_score, real_score


def analyze_video(video_path):

    video = cv2.VideoCapture(video_path)

    total_frames = int(
        video.get(cv2.CAP_PROP_FRAME_COUNT)
    )

    if total_frames <= 0:

        video.release()

        return None

    frame_positions = np.linspace(
        0,
        total_frames - 1,
        8,
        dtype=int
    )

    fake_scores = []
    real_scores = []

    for frame_position in frame_positions:

        video.set(
            cv2.CAP_PROP_POS_FRAMES,
            int(frame_position)
        )

        success, frame = video.read()

        if not success:

            continue

        fake_score, real_score = analyze_frame(
            frame
        )

        fake_scores.append(fake_score)
        real_scores.append(real_score)

    video.release()

    if len(fake_scores) == 0:

        return None

    fake_probability = (
        float(np.mean(fake_scores)) * 100
    )

    real_probability = (
        float(np.mean(real_scores)) * 100
    )

    return (
        fake_probability,
        real_probability,
        len(fake_scores)
    )


# =========================================================
# SESSION MEMORY
# =========================================================

if "fake_score" not in st.session_state:

    st.session_state.fake_score = 0.0


if "analysis_complete" not in st.session_state:

    st.session_state.analysis_complete = False


# =========================================================
# STEP 1
# =========================================================

st.header("01 — VIDEO EVIDENCE")

st.write(
    "Upload a real or suspected deepfake sample clip."
)


uploaded_video = st.file_uploader(
    "Upload video evidence",
    type=["mp4", "mov", "avi"]
)


if uploaded_video is not None:

    st.video(uploaded_video)


    if st.button(
        "SCAN VIDEO",
        type="primary",
        use_container_width=True
    ):

        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".mp4"
        ) as temporary_file:

            temporary_file.write(
                uploaded_video.read()
            )

            video_path = temporary_file.name


        with st.spinner(
            "Sampling frames and running AI detection..."
        ):

            results = analyze_video(video_path)


        if results is None:

            st.error(
                "The video could not be analyzed."
            )


        else:

            fake_probability = results[0]
            real_probability = results[1]
            frames_analyzed = results[2]

            st.session_state.fake_score = (
                fake_probability
            )

            st.session_state.analysis_complete = True


            if fake_probability >= 70:

                assessment = (
                    "HIGH VISUAL SUSPICION"
                )

                st.error(
                    "🔴 HIGH VISUAL SUSPICION"
                )


            elif fake_probability >= 40:

                assessment = (
                    "HUMAN REVIEW REQUIRED"
                )

                st.warning(
                    "🟡 HUMAN REVIEW REQUIRED"
                )


            else:

                assessment = (
                    "LOW VISUAL SUSPICION"
                )

                st.success(
                    "🟢 LOW VISUAL SUSPICION"
                )


            col1, col2, col3 = st.columns(3)


            with col1:

                st.metric(
                    "AI Fake Score",
                    f"{fake_probability:.2f}%"
                )


            with col2:

                st.metric(
                    "AI Real Score",
                    f"{real_probability:.2f}%"
                )


            with col3:

                st.metric(
                    "Frames Analyzed",
                    frames_analyzed
                )


            st.markdown(
                '<div class="result-box">',
                unsafe_allow_html=True
            )


            st.subheader(
                "AI VISUAL ANALYSIS REPORT"
            )


            st.write(
                f"""
                **Assessment:** {assessment}

                **Method:** The toolkit sampled
                {frames_analyzed} frames from
                different points in the uploaded
                video.

                Each frame was analyzed using a
                pretrained deepfake image
                classification model.

                The sampled frame scores were
                averaged to create the AI visual
                score.
                """
            )


            st.markdown(
                "</div>",
                unsafe_allow_html=True
            )


# =========================================================
# STEP 2
# =========================================================

st.divider()

st.header(
    "02 — HUMAN SPOT-THE-FAKE CHECKLIST"
)

st.write(
    """
    Watch the clip carefully and select every
    suspicious sign you observe.
    """
)


col1, col2 = st.columns(2)


with col1:

    blinking = st.checkbox(
        "Unnatural blinking"
    )

    lighting = st.checkbox(
        "Inconsistent lighting or shadows"
    )

    lip_sync = st.checkbox(
        "Lip movement does not match audio"
    )

    face_flicker = st.checkbox(
        "Face edges flicker or distort"
    )


with col2:

    skin_texture = st.checkbox(
        "Skin texture changes unnaturally"
    )

    background_warp = st.checkbox(
        "Background warps near the face"
    )

    audio_artifacts = st.checkbox(
        "Robotic or unusual audio artifacts"
    )

    unnatural_expression = st.checkbox(
        "Facial expressions appear unnatural"
    )


# =========================================================
# STEP 3
# =========================================================

st.divider()

st.header(
    "03 — FINAL KIT ASSESSMENT"
)


if st.button(
    "CALCULATE FINAL ASSESSMENT",
    type="primary",
    use_container_width=True
):

    if not st.session_state.analysis_complete:

        st.warning(
            "Scan a video before calculating "
            "the final assessment."
        )


    else:

        human_checks = [

            blinking,
            lighting,
            lip_sync,
            face_flicker,
            skin_texture,
            background_warp,
            audio_artifacts,
            unnatural_expression

        ]


        suspicious_signs = sum(
            human_checks
        )


        human_score = (
            suspicious_signs / 8
        ) * 100


        ai_score = (
            st.session_state.fake_score
        )


        final_score = (
            ai_score * 0.70
        ) + (
            human_score * 0.30
        )


        col1, col2, col3 = st.columns(3)


        with col1:

            st.metric(
                "AI Visual Score",
                f"{ai_score:.2f}%"
            )


        with col2:

            st.metric(
                "Human Suspicion Score",
                f"{human_score:.2f}%"
            )


        with col3:

            st.metric(
                "Final Kit Score",
                f"{final_score:.2f}%"
            )


        if final_score >= 70:

            st.error(
                "🔴 HIGH SUSPICION"
            )

            final_assessment = (
                "HIGH SUSPICION"
            )


        elif final_score >= 40:

            st.warning(
                "🟡 NEEDS FURTHER REVIEW"
            )

            final_assessment = (
                "NEEDS FURTHER REVIEW"
            )


        else:

            st.success(
                "🟢 LOW SUSPICION"
            )

            final_assessment = (
                "LOW SUSPICION"
            )


        st.subheader(
            "FINAL VERIFICATION REPORT"
        )


        st.write(
            f"""
            **AI Visual Score:**
            {ai_score:.2f}%

            **Human Suspicion Score:**
            {human_score:.2f}%

            **Suspicious Human Signs:**
            {suspicious_signs} / 8

            **Final Kit Score:**
            {final_score:.2f}%

            **Final Assessment:**
            {final_assessment}

            **Prototype scoring method:**
            70% AI visual detection +
            30% structured human review.
            """
        )


# =========================================================
# PROJECT INFORMATION
# =========================================================

st.divider()

st.header(
    "HOW THE DETECTIVE KIT WORKS"
)


st.write(
    """
    **1 — VIDEO SAMPLING**

    The toolkit selects frames from different
    points in the uploaded video.

    **2 — AI VISUAL DETECTION**

    A pretrained deepfake image classifier
    analyzes each sampled frame.

    **3 — HUMAN VERIFICATION**

    The investigator checks blinking, lighting,
    lip-sync, visual artifacts and audio
    artifacts.

    **4 — COMBINED ASSESSMENT**

    The educational prototype combines the AI
    visual score with a structured human
    suspicion score.
    """
)


st.warning(
    """
    EDUCATIONAL PROTOTYPE

    A high suspicion score does not prove that a
    video is a deepfake. Professional forensic
    verification may still be required.
    """
)


st.caption(
    "The Deepfake Detective Kit • AI + Human Media Verification"
)
