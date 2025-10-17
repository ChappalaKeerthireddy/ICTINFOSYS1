import streamlit as st
import pandas as pd
import numpy as np
import spacy
from spacy import displacy
from textblob import TextBlob
from deepface import DeepFace
from rembg import remove
from PIL import Image
import io
import speech_recognition as sr
import tempfile
import librosa
import librosa.display
import matplotlib.pyplot as plt

# -----------------------------
# Streamlit Page Configuration
# -----------------------------
st.set_page_config(page_title="Unstructured Data Analytics", layout="wide")
st.title("🧠 Unstructured Data Analytics Suite")

st.sidebar.title("Select Data Type")
data_type = st.sidebar.radio(
    "Choose what you want to analyze:",
    ["Text Analysis", "Image Analysis", "Audio Analysis"]
)

# =======================================================
# 🧠 1️⃣ TEXT ANALYSIS SECTION
# =======================================================
if data_type == "Text Analysis":
    st.header("💬 Text Analysis")
    st.markdown("Perform NER, Sentiment Analysis, and Keyword Extraction on text.")

    text_input = st.text_area("✏️ Enter or paste your text below:", height=200)

    if st.button("🚀 Run Text Analysis"):
        if text_input.strip():
            nlp = spacy.load("en_core_web_sm")
            doc = nlp(text_input)

            # Named Entity Recognition (NER)
            html = displacy.render(doc, style="ent", jupyter=False)
            st.subheader("🧩 Named Entities")
            st.markdown(html, unsafe_allow_html=True)

            # Entity Table
            entities = [(ent.text, ent.label_) for ent in doc.ents]
            if entities:
                st.table(pd.DataFrame(entities, columns=["Entity", "Label"]))
            else:
                st.info("No named entities found.")

            # Sentiment Analysis
            blob = TextBlob(text_input)
            polarity = blob.sentiment.polarity
            subjectivity = blob.sentiment.subjectivity

            if polarity > 0:
                sentiment = "😊 Positive"
            elif polarity < 0:
                sentiment = "😠 Negative"
            else:
                sentiment = "😐 Neutral"

            st.subheader("📊 Sentiment Analysis")
            col1, col2 = st.columns(2)
            col1.metric("Sentiment", sentiment)
            col2.metric("Polarity", round(polarity, 2))
            st.write(f"**Subjectivity:** {round(subjectivity, 2)}")

            # Keyword Extraction
            keywords = [token.text for token in doc if token.is_alpha and not token.is_stop]
            keywords_df = pd.DataFrame(pd.Series(keywords).value_counts().head(10)).reset_index()
            keywords_df.columns = ["Keyword", "Frequency"]

            st.subheader("🔑 Top Keywords")
            st.table(keywords_df)
        else:
            st.warning("Please enter some text for analysis.")

# =======================================================
# 🖼️ 2️⃣ IMAGE ANALYSIS SECTION
# =======================================================
elif data_type == "Image Analysis":
    st.header("🖼️ Image Analysis")
    st.markdown("Perform **Face Analysis** and **Background Removal** on images.")

    uploaded_image = st.file_uploader("📸 Upload an image file", type=["jpg", "jpeg", "png"])

    if uploaded_image is not None:
        image = Image.open(uploaded_image)
        st.image(image, caption="Uploaded Image", use_container_width=True)

        # Choose image task
        image_task = st.radio(
            "Select Image Operation:",
            ["Face Emotion & Attribute Detection", "Background Removal"]
        )

        # Face Analysis
        if image_task == "Face Emotion & Attribute Detection":
            if st.button("🔍 Analyze Face"):
                with st.spinner("Analyzing face..."):
                    try:
                        analysis = DeepFace.analyze(
                            img_path=np.array(image),
                            actions=['emotion', 'age', 'gender', 'race'],
                            enforce_detection=False
                        )
                        result = analysis[0] if isinstance(analysis, list) else analysis
                        st.success("✅ Face Analysis Complete!")

                        col1, col2 = st.columns(2)
                        col1.metric("Dominant Emotion", result['dominant_emotion'])
                        col1.metric("Estimated Age", result['age'])
                        col2.metric("Gender", result['dominant_gender'])
                        col2.metric("Race", result['dominant_race'])

                        st.json(result)
                    except Exception as e:
                        st.error(f"Error during analysis: {e}")

        # Background Removal
        elif image_task == "Background Removal":
            if st.button("🧼 Remove Background"):
                with st.spinner("Processing image..."):
                    try:
                        input_bytes = io.BytesIO()
                        image.save(input_bytes, format="PNG")
                        output_bytes = remove(input_bytes.getvalue())
                        output_image = Image.open(io.BytesIO(output_bytes))

                        st.image(output_image, caption="Background Removed", use_container_width=True)
                        st.download_button(
                            label="💾 Download Processed Image",
                            data=output_bytes,
                            file_name="bg_removed.png",
                            mime="image/png"
                        )
                    except Exception as e:
                        st.error(f"Error: {e}")
    else:
        st.info("Please upload an image to continue.")

# =======================================================
# 🎵 3️⃣ AUDIO ANALYSIS SECTION
# =======================================================
elif data_type == "Audio Analysis":
    st.header("🎧 Audio Analysis")
    st.markdown("Perform **Speech-to-Text**, **Sentiment**, and **Waveform Analysis** on audio files.")

    uploaded_audio = st.file_uploader("🎤 Upload an audio file", type=["wav", "mp3", "m4a"])

    if uploaded_audio is not None:
        # Save temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio:
            temp_audio.write(uploaded_audio.read())
            audio_path = temp_audio.name

        # Display waveform
        try:
            y, sr_rate = librosa.load(audio_path)
            fig, ax = plt.subplots(figsize=(8, 3))
            librosa.display.waveshow(y, sr=sr_rate, ax=ax)
            st.pyplot(fig)
        except Exception as e:
            st.warning(f"Could not display waveform: {e}")

        if st.button("🗣️ Transcribe & Analyze Audio"):
            recognizer = sr.Recognizer()
            with sr.AudioFile(audio_path) as source:
                audio = recognizer.record(source)
                try:
                    text = recognizer.recognize_google(audio)
                    st.subheader("📝 Transcription:")
                    st.write(text)

                    # Text Sentiment from Transcribed Speech
                    blob = TextBlob(text)
                    polarity = blob.sentiment.polarity
                    sentiment = "😊 Positive" if polarity > 0 else "😠 Negative" if polarity < 0 else "😐 Neutral"

                    st.subheader("💬 Sentiment from Audio Speech")
                    st.metric("Sentiment", sentiment)
                    st.metric("Polarity Score", round(polarity, 2))

                except Exception as e:
                    st.error(f"Speech recognition failed: {e}")
    else:
        st.info("Please upload an audio file (wav, mp3, m4a).")
