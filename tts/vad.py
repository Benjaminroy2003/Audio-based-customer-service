import torch
import numpy as np

class VoiceActivityDetection:
    sampling_rate = 16000
    def __init__(self,sampling_rate = 16000):
        self.model , utils =torch.hub.load(
             repo_or_dir="snakers4/silero-vad", model="silero_vad", force_reload=False
        )
        (
            self.get_speech_timestamps,
            self.save_audio,
            self.read_audio,
            self.VADIterator,
            self.collect_chunks,
        ) = utils

    def contains_speech(self,audio):
        frames =np.frombuffer(b"".join(audio) , dtype = np.int16)
        frames =frames/(1 << 15)
        audio = torch.tensor(frames.astype(np.float32))
        speech_timestamps =self.get_speech_timestamps(
            audio,self.model,sampling_rate = self.sampling_rate
        )
        return len(speech_timestamps) > 0