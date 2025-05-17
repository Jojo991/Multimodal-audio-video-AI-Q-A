import os
import glob

output_dir = "files/audio/"

audio_file = glob.glob(os.path.join(output_dir, "*.mp3"))
