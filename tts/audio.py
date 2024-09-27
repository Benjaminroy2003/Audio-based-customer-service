import pyaudio
import wave
import queue
from tts.vad import VoiceActivityDetection

FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
CHUNK = 1024

def record_audio():
    print("hi")
    audio_queue = queue.Queue()
    audio = pyaudio.PyAudio()

    stream = audio.open(format=FORMAT,
                        channels=CHANNELS,
                        rate=RATE,
                        input=True,
                        frames_per_buffer=CHUNK)

    print("Recording...")
    frames = []
    final_frames = []
    silence_seconds = 2
    started = False
    
    vad = VoiceActivityDetection()  # Instantiate VAD
    while True:
        try:
            data = stream.read(CHUNK)
            frames.append(data)
            audio_queue.put(data)
            contain_speech = vad.contains_speech(frames[int(-(RATE / CHUNK) * silence_seconds):])
            if contain_speech:
                final_frames.append(frames[int(-(RATE / CHUNK) * silence_seconds):])
            if not started and contain_speech:
                started = True
                print("Listening to speech...")
            if started and not contain_speech:
                break

        except Exception as e:
            print(f"Error reading stream: {e}")
            break

    print("Done recording.")

    # Save the recorded frames to a WAV file
    # if final_frames:
    #     with wave.open('output.wav', 'wb') as wf:
    #         wf.setnchannels(CHANNELS)
    #         wf.setsampwidth(audio.get_sample_size(FORMAT))
    #         wf.setframerate(RATE)
    #         wf.writeframes(b''.join(frames))

    stream.stop_stream()
    stream.close()
    audio.terminate()

if __name__ == "__main__":
    record_audio()
