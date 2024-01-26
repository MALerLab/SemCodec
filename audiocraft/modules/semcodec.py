import typing as tp

import torch
import numpy as np
import torch.nn as nn

from transformers.models.encodec.modeling_encodec import EncodecResidualVectorQuantizer
from transformers.models.encodec.configuration_encodec import EncodecConfig
    
class SemCodecMidiDecoder(nn.Module):
  def __init__(self, in_channels=128, hidden_size=256, out_channels=88, kernel_size=3, stride=1, padding=1):
      super().__init__()
      self.out_channels = out_channels
      # self.emb = SummationEmbedder(vocab_size=[2048, 2048, 2048, 2048], input_keys = 4, dim=in_channels)
      self.layers = nn.Sequential(
            nn.Conv1d(in_channels=in_channels, out_channels=hidden_size, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.BatchNorm1d(hidden_size),
            nn.ELU(),
            nn.Dropout(0.5),
            nn.Conv1d(in_channels=hidden_size, out_channels=hidden_size, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.BatchNorm1d(hidden_size),
            nn.ELU(),
            nn.Dropout(0.5),
            nn.Conv1d(in_channels=hidden_size, out_channels=hidden_size, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.BatchNorm1d(hidden_size),
            nn.ELU(),
            nn.Dropout(0.5),
        )
      self.rnn = nn.GRU(input_size=hidden_size, hidden_size=hidden_size//2, num_layers=2, batch_first=True, bidirectional=True, dropout=0.2)
      self.proj = nn.Linear(hidden_size, out_channels*2)
      self.act = nn.Sigmoid()
  
  def forward(self, x):
      # x = self.emb(x.permute(0,2,1))
      x = self.layers(x)
      x = self.rnn(x.permute(0,2,1))[0]
      x = self.proj(x)
      x = self.act(x.permute(0,2,1))
      x = torch.stack([x[:,:88,:],x[:,88:,:]], dim=1)
      return x

class SemCodecOnlyMidi(nn.Module):
  def __init__(self, in_channels=128, hidden_size=256, out_channels=88, kernel_size=3, stride=1, padding=1):
      super().__init__()
      self.frame_rate = 50
      self.out_channels = out_channels
      self.quantizer = EncodecResidualVectorQuantizer(config = EncodecConfig.from_pretrained('facebook/encodec_32khz'))
      self.quantizer.load_state_dict(torch.load('/home/jongmin/userdata/audiocraft/encodec_32khz_quantizer.pt'))
      self.decoder = SemCodecMidiDecoder(in_channels=in_channels, hidden_size=hidden_size, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding)

  def forward(self, x):
      x = self.quantizer.decode(x.transpose(0,1))
      x = self.decoder(x)
      return x
  
class RVQMultiEmbedding(nn.Module):
  def __init__(self, vocab_size=[2048, 2048, 2048, 2048], input_keys = 4, dim=256):
    super().__init__()
    '''
    vocab_size: dict of vocab size for each embedding layer
    is_attention: bool, whether this embedding is made by attention mechanism or not
    input_keys: list of input keys
    '''
    self.vocab_size = vocab_size
    self.d_model = dim
    self.input_keys = input_keys
    self.layers = []
    self._make_emb_layers()

  def _make_emb_layers(self):
    vocab_sizes = [self.vocab_size[key] for key in range(self.input_keys)]
    self.embedding_sizes = [self.d_model for _ in range(self.input_keys)]
    for vocab_size, embedding_size in zip(vocab_sizes, self.embedding_sizes):
      if embedding_size != 0:
        self.layers.append(nn.Embedding(vocab_size, embedding_size))
    self.layers = nn.ModuleList(self.layers)

  def forward(self, x):
    '''
    in case when need to apply different embedding for each input, for example, in case of flattened nb
    x: B x T
    '''
    embeddings = torch.zeros(x.shape[0], x.shape[1], self.d_model).to(x.device)
    emb_list = [module(x[:, (idx+1)%4::4]) for idx, module in enumerate(self.layers)]
    for idx, emb in enumerate(emb_list):
      embeddings[:, (idx+1)%4::4] = emb
    # emb_list = [module(x[:, idx::4]) for idx, module in enumerate(self.layers)]
    # for idx, emb in enumerate(emb_list):
    #   embeddings[:, idx::4] = emb
    return embeddings
  
  def get_emb_by_key(self, key:str, token:torch.Tensor):
    '''
    key: key of musical info
    token: B x T (idx of musical info)
    '''
    layer_idx = self.input_keys.index(key)
    return self.layers[layer_idx](token)

class SummationEmbedder(RVQMultiEmbedding):
  def __init__(self, vocab_size, input_keys, dim):
    super().__init__(vocab_size, input_keys, dim)

  def forward(self, x):
    '''
    x: B x T x num_musical_info(4)
    '''
    if isinstance(x, torch.Tensor): # input embedder
      emb_list = [module(x[..., i]) for i, module in enumerate(self.layers)]
    elif isinstance(x, list): # double-sequential decoder embedder
      emb_list = x
    stacked_emb = torch.stack(emb_list, dim=2) # B x T x num_musical_info(7, in case cp) x emb_size
    # sum
    output = torch.sum(stacked_emb, dim=2) # B x T x emb_size
    return output