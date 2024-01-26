import torch
from tqdm import tqdm

class EnCodecTokenMIDIDataset(torch.utils.data.Dataset):
    def __init__(self, paths):
        self.paths = sorted(paths)
        self.encodec_tensors, self.pr_tensors = self._load_pairs(self.paths)

    def _load_pairs(self, paths):
        encodec_tensors = []
        pr_tensors = []
        for path in tqdm(paths):
            path = '/home/jongmin/userdata/MAESTRO/maestro-v3.0.0' + '/' + path
            t_encodec_tensors, len_et =  self._load_encodec(path.replace('.midi', '_encodec.pt'))
            t_pr_tensors, len_pt = self._load_piano_roll(path.replace('.midi', '_pianoroll.pkl'))
            if len_et != len_pt:
                print("Length mismatch! Popping -2 index")
                print(path)
                print(len_et, len_pt)
                t_encodec_tensors.pop(-2)
            encodec_tensors.extend(t_encodec_tensors)
            pr_tensors.extend(t_pr_tensors)
        assert len(encodec_tensors) == len(pr_tensors)
        return encodec_tensors, pr_tensors
            

    def _load_encodec(self, path):
        loaded = torch.load(path)
        encodec_tensors = []
        for i in range(loaded['len']-1):
            encodec_tensors.append(loaded[i*30].squeeze(0))
        encodec_tensors.append(loaded[-1].squeeze(0))
        return encodec_tensors, loaded['len']

    def _load_piano_roll(self, path):
        loaded = torch.load(path)
        pr_tensors = []
        for i in range(loaded['len']-1):
            pr_tensors.append(loaded[i*30].to(torch.float32))
        pr_tensors.append(loaded[-1].to(torch.float32))
        return pr_tensors, loaded['len']

    def __len__(self):
        return len(self.encodec_tensors)

    def __getitem__(self, idx):
        encodec_tensor = self.encodec_tensors[idx]
        pr_tensor = self.pr_tensors[idx]
        return encodec_tensor, pr_tensor