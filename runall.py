from audio import record_audio
from stt_hf import stt
import threading
import queue

from transformers import pipeline

pipe = pipeline(
    "automatic-speech-recognition",model="openai/whisper-base.en",device="cpu",
)

def ear():
    print("in ear")
    audio_queue = queue.Queue()
    transcription_queue = queue.Queue()
    # create a thread
    audio_thread = threading.Thread(target = record_audio)
    sst_thread =threading.Thread(target = stt.transcribe, args=[pipe,audio_queue])
    audio_thread.start()
    sst_thread.start()

    audio_thread.join()
    sst_thread.join()

if __name__=="__main__":
    ear()
