import sounddevice as sd
import wavio
from faster_whisper import WhisperModel
from diffusers import StableDiffusionPipeline
import torch
import os
import time

# 1️⃣ Setup Faster Whisper
model = WhisperModel("base")

# 2️⃣ Setup Stable Diffusion Pipeline
pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float16
)
pipe.to("cuda" if torch.cuda.is_available() else "cpu")

# 3️⃣ Folders
OUTPUT_FOLDER = "static/images/"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# 4️⃣ Record audio
def record_audio(filename, duration=5, fs=44100):
    print(f"Recording for {duration} sec...")
    recording = sd.rec(int(duration * fs), samplerate=fs, channels=1)
    sd.wait()
    wavio.write(filename, recording, fs, sampwidth=2)
    print(f"Saved audio: {filename}")

# 5️⃣ Transcribe audio
def transcribe_audio(filename):
    segments, _ = model.transcribe(filename)
    text = " ".join([seg.text for seg in segments])
    print(f"Transcribed: {text}")
    return text

# 6️⃣ Generate image
def generate_image(prompt, output_name):
    print(f"Generating image for: {prompt}")
    image = pipe(prompt).images[0]
    image.save(output_name)
    print(f"Saved image: {output_name}")

# 7️⃣ Loop
if __name__ == "__main__":
    count = 1
    while True:
        audio_file = "chunk.wav"
        record_audio(audio_file, duration=5)

        text = transcribe_audio(audio_file)

        output_image = os.path.join(OUTPUT_FOLDER, f"image_{count}.png")
        generate_image(text, output_image)

        count += 1
        time.sleep(1)
