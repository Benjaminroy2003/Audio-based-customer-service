from transformers import pipeline

class stt:
    def __init__(self, model_id, device):
        self.pipe = pipeline(
            "automatic-speech-recognition", model=model_id, device=device
        )
        self.device = device
    
    def transcribe(pipe, audio_queue):
        from torch import no_grad
        with no_grad():
            transcription = pipe(audio_queue)
            print(transcription["text"].strip())
        return transcription["text"].strip()