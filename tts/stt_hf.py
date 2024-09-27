import pyaudio
import wave
import multiprocessing
import time
from tts.vad import VoiceActivityDetection
from transformers import pipeline

FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
CHUNK = 1024 * 5

def record_audio(frames,stop_event):
    audio = pyaudio.PyAudio()
    stream = audio.open(format=FORMAT,
                        channels=CHANNELS,
                        rate=RATE,
                        input=True,
                        frames_per_buffer=CHUNK)

    print("Recording...")
    silence_seconds = 2
    started = False
    vad = VoiceActivityDetection()  # Instantiate VAD

    while not stop_event.is_set():
        try:
            print("Reading stream...")
            data = stream.read(CHUNK)
            frames.append(data)
            contain_speech = vad.contains_speech(frames[int(-(RATE / CHUNK) * silence_seconds):])
            
            if contain_speech and not started:
                print("Listening to speech...")
                started = True

            if started and not contain_speech:
                break  # Stop recording if speech has stopped

        except Exception as e:
            print(f"Error reading stream: {e}")
            break

    print("Done recording.")
    stream.stop_stream()
    stream.close()
    audio.terminate()
    stop_event.set() # signal transcription is stop 

def transcribe(frames, pipe,stop_event):
    with open('transcriptions.txt',"w") as file:
        while not stop_event.is_set():
            if len(frames) > 0:
                wf = wave.open("temp.wav", "wb")
                wf.setnchannels(CHANNELS)
                wf.setsampwidth(pyaudio.PyAudio().get_sample_size(FORMAT))
                wf.setframerate(RATE)
                wf.writeframes(b''.join(frames))
                wf.close()

                # Perform transcription
                transcription = pipe("temp.wav")
                print("Transcription:", transcription["text"].strip())
                file.write(transcription["text"].strip())

            else:
                time.sleep(0.1)

def ear():
    print("Starting ear...")

    pipe = pipeline("automatic-speech-recognition", model="openai/whisper-base.en", device="cpu")

    # Use a manager to share the 'frames' list between processes
    manager = multiprocessing.Manager()
    frames = manager.list()  # Shared list for frames
    stop_event = multiprocessing.Event()

    # Create a process for recording audio
    print("Starting audio and transcription processes...")
    audio_process = multiprocessing.Process(target=record_audio, args=(frames,stop_event))
    sst_process = multiprocessing.Process(target=transcribe, args=(frames, pipe,stop_event))

    audio_process.start()
    sst_process.start()

    audio_process.join()
    sst_process.join()

    print("All procesing are stopped")

if __name__ == "__main__":
    ear()
