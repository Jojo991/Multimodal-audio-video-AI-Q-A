import os
import glob
from openai import OpenAI
import yt_dlp as youtube_dl
from yt_dlp import DownloadError
import docarray

openai_api_key = os.getenv("OPENAI_API_KEY")

youtube_url = "https://www.youtube.com/watch?v=aqzxYofJ_ck"
output_dir = "files/audio/"

# Config for youtube-dl
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
    "verbose": True,
    "http_headers": {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/94.0.4606.71 Safari/537.36",
        "Accept-Language": "en-US,en;q=0.9",
        "Accept-Encoding": "gzip, deflate, br",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
        "Connection": "keep-alive",
        "Referer": "https://www.youtube.com/"
    }
}

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

print(f"Downloading video from {youtube_url}")

try:
    with youtube_dl.YoutubeDL(ydl_config) as ydl:
        ydl.download([youtube_url])
except DownloadError:
    with youtube_dl.YoutubeDL(ydl_config) as ydl:
        ydl.download([youtube_url])


audio_file = glob.glob(os.path.join(output_dir, "*.mp3"))
audio_filename = audio_file[0]
print(audio_filename)

audio_file = audio_filename
output_file = "files/transcripts/transcripts.txt"
model = "whisper-1"

print("converting audio to text...")

with open(audio_file, "rb") as audio:
    response = OpenAI.audio.transcribe(model, audio)

transcript = (response["text"])

if output_file is not None:
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "w") as file:
        file.write(transcript)

print(transcript)