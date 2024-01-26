from audiocraft.models import CompressionModel
import torchaudio
import torch
from collections import OrderedDict
from tqdm import tqdm
import os

model = CompressionModel.get_pretrained('facebook/encodec_32khz', device='cuda')

for dirpath, dirnames, filenames in os.walk("/home/jongmin/userdata/MAESTRO/maestro-v3.0.0/"):
    for filename in tqdm([f for f in filenames if f.endswith("_encodec.pt")]):
        print("Decoding: ", os.path.join(dirpath, filename))
        loaded = torch.load(str(os.path.join(dirpath, filename)))

        wav_dict = {}
        wav_dict['len']=loaded['len']

        for i in range(loaded['len']-1):
            a = model.decode(loaded[i*30].to(torch.int64).cuda())
            wav_dict[i*30]=a.cpu().detach()

        a = model.decode(loaded[-1].to(torch.int64).cuda())
        wav_dict[-1]=a.cpu().detach()

        torch.save(wav_dict, str(os.path.join(dirpath, filename)).rsplit('.', 1)[0]+'_decoded.pth')