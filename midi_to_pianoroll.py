import torch
from tqdm import tqdm
import os

import mido
import numpy as np


def parse_midi(path):
    """open midi file and return np.array of (onset, offset, note, velocity) rows"""
    midi = mido.MidiFile(path)

    time = 0
    sustain = False
    events = []
    for message in midi:
        time += message.time

        if message.type == 'control_change' and message.control == 64 and (message.value >= 64) != sustain:
            # sustain pedal state has just changed
            sustain = message.value >= 64
            event_type = 'sustain_on' if sustain else 'sustain_off'
            event = dict(index=len(events), time=time, type=event_type, note=None, velocity=0)
            events.append(event)

        if 'note' in message.type:
            # MIDI offsets can be either 'note_off' events or 'note_on' with zero velocity
            velocity = message.velocity if message.type == 'note_on' else 0
            event = dict(index=len(events), time=time, type='note', note=message.note, velocity=velocity, sustain=sustain)
            events.append(event)

    notes = []
    for i, onset in enumerate(events):
        if onset['velocity'] == 0:
            continue

        # find the next note_off message
        offset = next(n for n in events[i + 1:] if n['note'] == onset['note'] or n is events[-1])

        if offset['sustain'] and offset is not events[-1]:
            # if the sustain pedal is active at offset, find when the sustain ends
            offset = next(n for n in events[offset['index'] + 1:]
                          if n['type'] == 'sustain_off' or n['note'] == onset['note'] or n is events[-1])

        note = (onset['time'], offset['time'], onset['note'])#, onset['velocity'])
        notes.append(note)

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
    piano_roll = np.zeros((2, note_range[1] - note_range[0], int(np.ceil(midi_notes[-1, 1] * fs))))

    # Iterate over all notes
    for note in midi_notes:
        # Get the start and end indices for the note
        start = int(np.floor((note[0]) * fs))
        end = int(np.floor((note[1]) * fs))
    
        # Add the note to the piano roll
        piano_roll[1, int(note[2]) - note_range[0], start:end+1] = 1
        piano_roll[0, int(note[2]) - note_range[0], start] = 1

    return piano_roll

def slice_piano_roll_dict(piano_roll):
    piano_roll_segments = {}
    piano_roll_segments['len']=int(piano_roll.shape[-1]//1500)+1
    for i in range(piano_roll_segments['len']-1):
        piano_roll_segments[i*30]=(torch.Tensor(piano_roll[...,i*1500:(i+1)*1500]).to(torch.int8))
    piano_roll_segments[-1]=(torch.Tensor(piano_roll[...,-1500:]).to(torch.int8))
    return piano_roll_segments

# for dirpath, dirnames, filenames in os.walk("/home/jongmin/userdata/MAESTRO/maestro-v3.0.0/"):
#     for filename in tqdm([f for f in filenames if f.endswith(".midi")]):
#         print("Piano rolling: ", os.path.join(dirpath, filename))
#         parsed_midi = parse_midi(str(os.path.join(dirpath, filename)))
#         piano_roll = create_piano_roll(parsed_midi)
#         piano_rolls_dict = slice_piano_roll_dict(piano_roll)
        
#         torch.save(piano_rolls_dict, str(os.path.join(dirpath, filename)).rsplit('.', 1)[0]+'_pianoroll.pkl')