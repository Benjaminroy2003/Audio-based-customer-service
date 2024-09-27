import pyaudio
import wave
from transformers import pipeline
import numpy as np

FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
CHUNK = 1024
RECORD_SECONDS = 2  # This ensures 2 seconds of audio at a time

def record_audio():
    # Initialize pyaudio object
    audio = pyaudio.PyAudio()
    
    # Open stream to record audio
    stream = audio.open(format=FORMAT,
                        channels=CHANNELS,
                        rate=RATE,
                        input=True,
                        frames_per_buffer=CHUNK)

    print("Recording...")
    
    # Initialize whisper ASR pipeline once
    pipe = pipeline("automatic-speech-recognition", model="openai/whisper-base.en", device="cpu")

    try:
        while True:
            frames = []
            # Capture 2 seconds of audio
            for _ in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
                data = stream.read(CHUNK, exception_on_overflow=False)
                frames.append(data)
            
            # Save to temporary wav file
            wf = wave.open("temp.wav", "wb")
            wf.setnchannels(CHANNELS)
            wf.setsampwidth(audio.get_sample_size(FORMAT))
            wf.setframerate(RATE)
            wf.writeframes(b''.join(frames))
            wf.close()
            
            # Transcribe audio file using whisper
            print("Transcribing...")
            transcription = pipe("temp.wav")
            print("Transcription:", transcription["text"].strip())

    except KeyboardInterrupt:
        # Gracefully stop recording on keyboard interrupt
        print("\nStopped recording.")
    
    except Exception as e:
        print(f"ERROR: {e}")
    
    finally:
        # Ensure the stream and pyaudio objects are cleaned up
        stream.stop_stream()
        stream.close()
        audio.terminate()

if __name__ == "__main__":
    record_audio()
