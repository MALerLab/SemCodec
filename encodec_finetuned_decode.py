from audiocraft.solvers import CompressionSolver
import torchaudio
import torch
from collections import OrderedDict
from tqdm import tqdm
import os

model = CompressionSolver.model_from_checkpoint("/home/sake/userdata/encodec_finetuned_checkpoint_350.th", "cpu")

for dirpath, dirnames, filenames in os.walk("/home/sake/userdata/encodec_gen_0308/encodec_generation/conditional/finetuned"):
    for filename in tqdm([f for f in filenames if f.endswith(".pt")]):
        print("Decoding: ", os.path.join(dirpath, filename))
        x = torch.load(str(os.path.join(dirpath, filename)), "cpu")
        x = x[:,1:-1,:]
        x = x.permute(0,2,1)

        a = model.decode(x.cpu())

        torchaudio.save(str(os.path.join(dirpath, filename)).rsplit('.', 1)[0]+'.wav', a.squeeze(0).cpu(), 32000)