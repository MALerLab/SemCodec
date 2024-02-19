import torch
from tqdm import tqdm
import os

import mido
import numpy as np

import torchaudio

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
    # piano_roll = np.zeros((2, note_range[1] - note_range[0], fs*30))
    piano_roll = np.zeros((2, note_range[1] - note_range[0], fs*midi_notes[-1,1]))

    # Iterate over all notes
    for note in midi_notes:
        # Get the start and end indices for the note
        start = int(np.floor((note[0]) * fs))
        end = int(np.floor((note[1]) * fs))
    
        # Add the note to the piano roll
        piano_roll[1, int(note[2]) - note_range[0], start:end+1] = 1
        piano_roll[0, int(note[2]) - note_range[0], start] = 1

    return piano_roll

def create_full_piano_roll(midi_notes, duration, fs=50, note_range=(21, 109)):
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
    # piano_roll = np.zeros((2, note_range[1] - note_range[0], fs*30))
    piano_roll = np.zeros((2, note_range[1] - note_range[0], int(np.ceil(fs*duration))))

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

# Example usage
# notes = parse_midi('path_to_midi_file.mid', start_time=10, end_time=40)

# for dirpath, dirnames, filenames in os.walk("/home/jongmin/userdata/MAESTRO/maestro-v3.0.0/"):
#     for filename in tqdm([f for f in filenames if f.endswith(".midi")]):
#         print("Piano rolling: ", os.path.join(dirpath, filename))

#         y, sr = torchaudio.load(str(os.path.join(dirpath, filename)).rsplit('.', 1)[0]+'_32khz_mono.wav')
#         duration = y.shape[-1]/sr

#         parsed_midi = parse_midi(str(os.path.join(dirpath, filename)))
#         piano_roll = create_full_piano_roll(parsed_midi, duration)
#         # piano_roll = create_piano_roll(parsed_midi)
#         # piano_rolls_dict = slice_piano_roll_dict(piano_roll)
#         print(piano_roll.shape)
#         torch.save(piano_roll, str(os.path.join(dirpath, filename)).rsplit('.', 1)[0]+'_full_pianoroll.pkl')
