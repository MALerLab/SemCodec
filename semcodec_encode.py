from audiocraft.solvers import SemCompressionSolver
import torchaudio
import torch
from collections import OrderedDict
from tqdm import tqdm
import os

model = SemCompressionSolver.model_from_checkpoint("/home/sake/userdata/semcodec_checkpoint_660.th", "cuda")

for dirpath, dirnames, filenames in os.walk("/home/sake/userdata/MAESTRO/maestro-v3.0.0_32khz_mono/"):
    for filename in tqdm([f for f in filenames if f.endswith("_32khz_mono.wav")]):
        print("Tokeninzing: ", os.path.join(dirpath, filename))
        x, sr = torchaudio.load(str(os.path.join(dirpath, filename)))
        # x = torchaudio.functional.resample(x.cuda(), orig_freq=sr, new_freq=32000)
        # x = x.mean(0, keepdim=True)

        token_dict = {}
        token_dict['len']=int(x.shape[-1]/(32000*30))+1

        for i in range(int(x.shape[-1]/(32000*30))):
            a, b = model.encode(x[...,i*32000*30:(i+1)*32000*30].unsqueeze(0).cuda())
            token_dict[i*30]=a.cpu().detach().to(torch.int16)

        a, b = model.encode(x[...,-32000*30:].unsqueeze(0).cuda())
        token_dict[-1]=a.cpu().detach().to(torch.int16)

        torch.save(token_dict, str(os.path.join(dirpath, filename)).rsplit('.', 1)[0]+'_semcodec.pt')