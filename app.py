import os
import glob
import streamlit as st
from openai import OpenAI
import yt_dlp as youtube_dl
from yt_dlp import DownloadError
from langchain.document_loaders import TextLoader
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.vectorstores import DocArrayInMemorySearch
from langchain.embeddings import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env file

# Set OpenAI API key from environment variable
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    st.error("Please set the OPENAI_API_KEY environment variable.")
    st.stop()

# Function to download and transcribe YouTube video
def download_and_transcribe(youtube_url):
    output_dir = "files/audio/"
    transcript_file = "files/transcripts/transcripts.txt"

    ydl_config = {
        "format": "bestaudio/best",
        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": "mp3",
                "preferredquality": "192",
            }
        ],
        "outtmpl": os.path.join(output_dir, "%(title)s.%(ext)s"),
        "verbose": False,
        "http_headers": {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/94.0.4606.71 Safari/537.36"
        }
    }

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    try:
        with youtube_dl.YoutubeDL(ydl_config) as ydl:
            ydl.download([youtube_url])
    except DownloadError as e:
        st.error(f"Error downloading video: {e}")
        return None

    audio_files = glob.glob(os.path.join(output_dir, "*.mp3"))
    if not audio_files:
        st.error("No audio file found after download.")
        return None

    audio_file = audio_files[0]

    # Transcribe audio using OpenAI Whisper
    with open(audio_file, "rb") as audio:
        client = OpenAI()
        transcript_response = client.audio.transcriptions.create(model="whisper-1", file=audio)
        transcript = transcript_response.text

    if transcript_file:
        os.makedirs(os.path.dirname(transcript_file), exist_ok=True)
        with open(transcript_file, "w", encoding="utf-8") as f:
            f.write(transcript)

    return transcript_file

# Function to load transcript and create QA system
def create_qa_system(transcript_path):
    loader = TextLoader(transcript_path)
    docs = loader.load()

    db = DocArrayInMemorySearch.from_documents(docs, OpenAIEmbeddings())
    retriever = db.as_retriever()
    llm = ChatOpenAI(temperature=0.0)
    qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever, verbose=False)
    return qa

import streamlit as st

st.set_page_config(
    page_title="Multimodal AI Q&A Bot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    .css-1d391kg {
        background-color: rgba(255, 255, 255, 0.1);
        border-radius: 15px;
        padding: 20px;
        margin-bottom: 20px;
    }
    .stButton>button {
        background-color: #764ba2;
        color: white;
        border-radius: 12px;
        padding: 18px 36px;
        font-weight: 700;
        font-size: 22px;
        transition: background-color 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #5a3780;
        color: #f0e6ff;
    }
    .stTextInput>div>input, .stTextArea>div>textarea {
        border-radius: 12px;
        padding: 18px;
        border: 2px solid #764ba2;
        background-color: #f9f7ff;
        color: #333;
        font-size: 22px;
    }
    .stSlider>div>input[type=range] {
        height: 40px;
        margin: 10px 0;
    }
    .sidebar .sidebar-content {
        background: linear-gradient(135deg, #764ba2 0%, #667eea 100%);
        color: white;
        padding: 20px;
        border-radius: 15px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Sidebar with navigation tabs
tab = st.sidebar.radio("Navigation", ["Convert Video", "Ask Question"])

st.sidebar.markdown(
    "<h3 style='color:#000000; font-family: Segoe UI, Tahoma, Geneva, Verdana, sans-serif; font-weight: bold; margin-bottom: 0;'>About</h3>",
    unsafe_allow_html=True,
)
st.sidebar.markdown(
    "<p style='color:#000000; font-family: Segoe UI, Tahoma, Geneva, Verdana, sans-serif;'>This app allows you to convert YouTube videos to text and ask questions about the content using AI.</p>",
    unsafe_allow_html=True,
)

# State to hold transcript text
if "transcript_text" not in st.session_state:
    st.session_state.transcript_text = ""

if tab == "Convert Video":
    st.header("Convert YouTube Video to Text")
    youtube_url = st.text_input("Enter YouTube Video URL", value="https://www.youtube.com/watch?v=aqzxYofJ_ck", key="url_input")

    if st.button("Convert Video to Text"):
        if not youtube_url:
            st.error("Please enter a YouTube video URL.")
        else:
            with st.spinner("Converting video to text..."):
                transcript_path = download_and_transcribe(youtube_url)
                if transcript_path:
                    with open(transcript_path, "r", encoding="utf-8") as f:
                        st.session_state.transcript_text = f.read()
                    st.markdown("<p style='color: yellow; font-weight: bold; font-size: 20px;'>Conversion complete! ✨</p>", unsafe_allow_html=True)
    if st.session_state.transcript_text:
        st.subheader("Transcript:")
        st.text_area("Transcript Text", st.session_state.transcript_text, height=600, key="transcript_text_area")

elif tab == "Ask Question":
    st.header("Ask a Question about the Video")
    if not st.session_state.transcript_text:
        st.warning("Please convert a video to text first in the 'Convert Video' tab.")
    else:
        question = st.text_input("Enter your question about the video", key="question_input")

        # Add a slider for temperature with explanation tooltip
        temperature = st.slider(
            "Temperature (controls randomness/creativity of AI responses)",
            min_value=0.0,
            max_value=1.0,
            value=0.0,
            step=0.1,
            help="Lower values make output more focused and deterministic. Higher values make output more diverse and creative."
        )

        if st.button("Get Answer"):
            if not question:
                st.error("Please enter a question.")
            else:
                with st.spinner("Getting answer..."):
                    qa_system = create_qa_system("files/transcripts/transcripts.txt")
                    # Assuming the qa_system can accept temperature parameter if needed
                    answer = qa_system.run(question)
                    st.subheader("Answer:")
                    st.write(answer)
