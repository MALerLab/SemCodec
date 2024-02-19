import torch
from tqdm import tqdm

from .audio_dataset import AudioDataset, AudioMeta
import pretty_midi
import mido
import numpy as np

#For No-finetuning Test
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

# class MIDIAudioDataset(AudioDataset):
#     """AudioDataset that always returns metadata as SegmentWithAttributes along with the audio waveform.

#     See `audiocraft.data.audio_dataset.AudioDataset` for initialization arguments.
#     """
#     def __init__(self, meta: tp.List[AudioMeta], **kwargs):
#         self.return_info = True
#         super().__init__(clusterify_all_meta(meta), **kwargs)

#     def __getitem__(self, index: int) -> tp.Union[torch.Tensor, tp.Tuple[torch.Tensor, SegmentWithAttributes]]:
#         if not self.return_info:
#             wav = super().__getitem__(index)
#             assert isinstance(wav, torch.Tensor)
#             return wav
#         wav, meta = super().__getitem__(index)
#         return wav, AudioInfo(**meta.to_dict())

# For Finetuning Test
class MIDIAudioDataset(AudioDataset):
    def __init__(self, meta, **kwargs):
        super().__init__(meta, **kwargs)
        self.return_info = True

    def load_midi(self, midi_file_path):
        """
        Load a MIDI file and return its piano roll representation.
        """
        midi_data = pretty_midi.PrettyMIDI(midi_file_path)
        # Extract piano roll, shape [88, time_steps], then expand to [2, 88, time_steps]
        piano_roll = midi_data.get_piano_roll(fs=50)
        onset_roll = midi_data.get_onset_strengths(fs=50)
        piano_roll = np.stack([onset_roll, piano_roll], axis=0)
        return piano_roll

    def __getitem__(self, index):
        # Fetch the audio segment using parent class method
        audio_segment, segment_info = super().__getitem__(index, quantize_seek_time= True)
        
        # Calculate the corresponding MIDI segment time
        start_time = segment_info.seek_time
        end_time = start_time + self.segment_duration  # 30 seconds segment
        # print(start_time*50, end_time*50)
        # Construct Piano Roll file path (this may vary based on your file structure)
        pr_file_path = f"{segment_info.meta.path.replace('_32khz_mono.wav', '_full_pianoroll.pkl')}"
        
        # Load the Piano Roll file and extract the segment
        # print(start_time, end_time)
        full_pr = torch.load(pr_file_path)
        piano_roll_segment = full_pr[..., int(np.round(start_time))*50:int(np.round(end_time))*50]
        if piano_roll_segment.shape[-1] < 1500:
            print("Piano Roll Segment too short! Popping")
            print(piano_roll_segment.shape)
            return self.__getitem__(index+1)
        # print(piano_roll_segment.shape)
        return audio_segment, piano_roll_segment
    
    def collater(self, samples):
        # print(samples)
        audio_segments = []
        piano_roll_segments = []
        for audio, pr in samples:
            audio_segments.append(audio)
            piano_roll_segments.append(pr)
        audio_segments = torch.stack(audio_segments)
        piano_roll_segments = torch.stack(piano_roll_segments)
        return audio_segments, piano_roll_segments

def parse_midi_slice(path, start_time, end_time):
    """Open midi file and return np.array of (onset, offset, note) rows within start_time and end_time."""
    midi = mido.MidiFile(path)

    # Calculate the absolute time in seconds for each MIDI event
    absolute_time = 0
    events = []
    for track in midi.tracks:
        for message in track:
            absolute_time += mido.tick2second(message.time, midi.ticks_per_beat, mido.bpm2tempo(120)) # Default tempo is 120 BPM

            if message.type == 'control_change' and message.control == 64 and (message.value >= 64):
                # Sustain pedal event
                events.append(dict(time=absolute_time, type='sustain', note=None, velocity=message.value))

            if message.type in ['note_on', 'note_off']:
                # Note event
                velocity = message.velocity if message.type == 'note_on' else 0
                events.append(dict(time=absolute_time, type='note', note=message.note, velocity=velocity))

    # Filter out events that are outside the specified time interval
    events = [e for e in events if start_time <= e['time'] <= end_time]

    # Process note on and off events to create note tuples
    notes = []
    for event in events:
        if event['type'] == 'note' and event['velocity'] > 0:  # Note on
            onset_time = event['time']
            note = event['note']

            # Find corresponding note off event
            note_off_events = [e for e in events if e['type'] == 'note' and e['note'] == note and e['velocity'] == 0]
            offset_time = next((e['time'] for e in note_off_events if e['time'] > onset_time), end_time)

            # Add note tuple (onset, offset, note)
            notes.append((onset_time-start_time, offset_time-start_time, note))

    return np.array(notes)

def create_piano_roll(midi_notes, fs=50, note_range=(21, 109)):
    """
    Create a piano roll from the given MIDI notes.
    
    Parameters:
    midi_notes (np.array): An array of MIDI notes with the format [onset, offset, note, velocity].
    fs (int): The sampling frequency of the piano roll.
    note_range (tuple): The range of notes to include in the piano roll.
    rf (int): The receptive field of EnCodec token.
    sr (int): The sampling rate of the EnCodec model.
    
    Returns:
    np.array: A piano roll of the MIDI notes.
    """
    # Create an empty piano roll
    piano_roll = np.zeros((2, note_range[1] - note_range[0], fs*30))

    # Iterate over all notes
    for note in midi_notes:
        # Get the start and end indices for the note
        start = int(np.floor((note[0]) * fs))
        end = int(np.floor((note[1]) * fs))
    
        # Add the note to the piano roll
        piano_roll[1, int(note[2]) - note_range[0], start:end+1] = 1
        piano_roll[0, int(note[2]) - note_range[0], start] = 1

    return piano_roll
