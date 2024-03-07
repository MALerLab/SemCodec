from audiocraft.solvers import SemCompressionSolver
import torchaudio
import torch
from collections import OrderedDict
from tqdm import tqdm
import os

model = SemCompressionSolver.model_from_checkpoint("/home/sake/userdata/semcodec_checkpoint_660.th", "cuda")

for dirpath, dirnames, filenames in os.walk("/home/sake/userdata/semcodec_generated/"):
    for filename in tqdm([f for f in filenames if f.endswith(".pt")]):
        print("Decoding: ", os.path.join(dirpath, filename))
        x = torch.load(str(os.path.join(dirpath, filename)))
        x = x[:,1:-1,:]
        x = x.permute(0,2,1)

        a = model.decode(x.cuda())

        torchaudio.save(str(os.path.join(dirpath, filename)).rsplit('.', 1)[0]+'.wav', a.squeeze(0).cpu(), 32000)