from piano_transcription_inference import PianoTranscription, sample_rate

transcriptor = PianoTranscription(
    device="cuda", checkpoint_path="./model.pth"
)

import torch
import librosa
import os
from tqdm import tqdm

encodec_sr = 32000

splits = torch.load("/home/jongmin/userdata/MAESTRO/maestro-v3.0.0/maestro-v3.0.0_split.pt")
val_split = splits["validation"]
test_split = splits["test"]

# for filename in tqdm(val_split):
#     encodec_filename = os.path.join("/home/jongmin/userdata/MAESTRO/maestro-v3.0.0/", filename).replace(".midi", "_encodec_decoded.pth")
#     pr_filename = os.path.join("/home/jongmin/userdata/MAESTRO/maestro-v3.0.0/", filename).replace(".midi", "_pianoroll.pkl")
#     loaded = torch.load(encodec_filename)
#     print(encodec_filename)
#     pr_loaded = torch.load(pr_filename)
#     if loaded["len"] != pr_loaded["len"]:
#         loaded.pop((loaded["len"]-2)*30)
#     for i in range(pr_loaded["len"]-1):
#         loaded_tmp = librosa.resample(loaded[i*30].numpy().squeeze(), encodec_sr, sample_rate)
#         out = transcriptor.transcribe(loaded_tmp, encodec_filename.replace("_encodec_decoded.pth", f"_bytedance_{i*30}.mid"))
        
#     loaded_tmp = librosa.resample(loaded[-1].numpy().squeeze(), encodec_sr, sample_rate)
#     out = transcriptor.transcribe(loaded_tmp, encodec_filename.replace("_encodec_decoded.pth", "_bytedance_-1.mid"))

for filename in tqdm(test_split):
    encodec_filename = os.path.join(filename).replace(".midi", "_encodec_decoded.pth")
    pr_filename = os.path.join(filename).replace(".midi", "_pianoroll.pkl")
    loaded = torch.load(encodec_filename)
    print(encodec_filename)
    pr_loaded = torch.load(pr_filename)
    if loaded["len"] != pr_loaded["len"]:
        loaded.pop((loaded["len"]-2)*30)
    for i in range(pr_loaded["len"]-1):
        loaded_tmp = librosa.resample(loaded[i*30].numpy().squeeze(), encodec_sr, sample_rate)
        out = transcriptor.transcribe(loaded_tmp, encodec_filename.replace("_encodec_decoded.pth", f"_bytedance_{i*30}.mid"))
        
    loaded_tmp = librosa.resample(loaded[-1].numpy().squeeze(), encodec_sr, sample_rate)
    out = transcriptor.transcribe(loaded_tmp, encodec_filename.replace("_encodec_decoded.pth", "_bytedance_-1.mid"))
