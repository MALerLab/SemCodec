import note_eval
import os
import pretty_midi
import torch
import wandb
import numpy as np

def create_piano_roll(midi_notes, fs=100, note_range=(21, 109)):
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
        start = int(np.round((note[0]) * fs))
        end = int(np.round((note[1]) * fs))
    
        # Add the note to the piano roll
        if start == piano_roll.shape[-1]:
            continue

        piano_roll[1, int(note[2]) - note_range[0], start:end] = 1
        piano_roll[0, int(note[2]) - note_range[0], start] = 1

    return piano_roll

def slice_midi(midi_data, start_time, end_time):

    # Create a new PrettyMIDI object for the sliced data
    sliced_midi_data = pretty_midi.PrettyMIDI()

    # Iterate over all instruments in the original MIDI data
    for instrument in midi_data.instruments:
        # Create a new instrument object for the sliced data
        sliced_instrument = pretty_midi.Instrument(program=instrument.program, is_drum=instrument.is_drum, name=instrument.name)

        # Slice the notes
        for note in instrument.notes:
            # If the note is within the slice times
            if start_time <= note.start < end_time:
                # Copy the note and adjust the start and end times
                new_note = pretty_midi.Note(
                    velocity=note.velocity,
                    pitch=note.pitch,
                    start=max(note.start, start_time) - start_time,
                    end=min(note.end, end_time) - start_time
                )
                sliced_instrument.notes.append(new_note)

        # Slice the pitch bends
        for pitch_bend in instrument.pitch_bends:
            if start_time <= pitch_bend.time < end_time:
                # Copy the pitch bend and adjust the time
                new_pitch_bend = pretty_midi.PitchBend(
                    pitch=pitch_bend.pitch,
                    time=pitch_bend.time - start_time
                )
                sliced_instrument.pitch_bends.append(new_pitch_bend)

        # Slice the control changes
        for control_change in instrument.control_changes:
            if start_time <= control_change.time < end_time:
                # Copy the control change and adjust the time
                new_control_change = pretty_midi.ControlChange(
                    number=control_change.number,
                    value=control_change.value,
                    time=control_change.time - start_time
                )
                sliced_instrument.control_changes.append(new_control_change)

        # Add the sliced instrument to the new MIDI data
        sliced_midi_data.instruments.append(sliced_instrument)

    return sliced_midi_data

run = wandb.init(
        project="EnCodec_Decoded_Audio_ByteDance_Transcription_mir_eval",
        config = {
        }
    )

root = "/home/jongmin/userdata/MAESTRO/maestro-v3.0.0/"
splits = torch.load("/home/jongmin/userdata/MAESTRO/maestro-v3.0.0/maestro-v3.0.0_split.pt")
val_split = splits["validation"]

metrics = []
avg_metric = {}
count = 0

for filename in val_split:
    label_path = os.path.join(root, filename)
    print(label_path)
    label = pretty_midi.PrettyMIDI(label_path)
    length_sec = label.instruments[0].notes[-1].end
    for i in range(int(length_sec//30)):
        count += 1
        label_midi = slice_midi(label, i*30, (i+1)*30)
        transcribed_midi = pretty_midi.PrettyMIDI(label_path.replace(".midi", "_bytedance_{}.mid".format(i*30)))

        label_array = []
        for note in label_midi.instruments[0].notes:
            label_array.append([note.start, note.end, note.pitch, note.velocity])
        label_array = np.array(label_array)
        label_pr = create_piano_roll(label_array, fs=100)

        transcribed_array = []
        for note in transcribed_midi.instruments[0].notes:
            transcribed_array.append([note.start, note.end, note.pitch, note.velocity])
        transcribed_array = np.array(transcribed_array)
        transcribed_pr = create_piano_roll(transcribed_array, fs=100)

        metric = note_eval.evaluate(torch.Tensor(label_pr), torch.Tensor(transcribed_pr))
        for key in metric.keys():
            metric[key] = float(metric[key][0])
            avg_metric["avg/" + key] = avg_metric.get("avg/" + key, 0) + metric[key]
        metric["filename"] = filename
        metric["idx"] = i
        print(metric)
        wandb.log(dict(metric))
        metrics.append(metric)
    count += 1
    label_midi = slice_midi(label, length_sec-30, length_sec)
    transcribed_midi = pretty_midi.PrettyMIDI(label_path.replace(".midi", "_bytedance_{}.mid".format(-1)))

    label_array = []
    for note in label_midi.instruments[0].notes:
        label_array.append([note.start, note.end, note.pitch, note.velocity])
    label_array = np.array(label_array)
    label_pr = create_piano_roll(label_array, fs=100)

    transcribed_array = []
    for note in transcribed_midi.instruments[0].notes:
        transcribed_array.append([note.start, note.end, note.pitch, note.velocity])
    transcribed_array = np.array(transcribed_array)
    transcribed_pr = create_piano_roll(transcribed_array, fs=100)

    metric = note_eval.evaluate(torch.Tensor(label_pr), torch.Tensor(transcribed_pr))
    for key in metric.keys():
        metric[key] = float(metric[key][0])
        avg_metric["avg/" + key] = avg_metric.get("avg/" + key, 0) + metric[key]
    metric["filename"] = filename
    metric["idx"] = -1
    print(metric)
    wandb.log(dict(metric))
    metrics.append(metric)

torch.save(metrics, "bytedance_metrics.pt")

for key in avg_metric.keys():
    avg_metric[key] /= count

wandb.log(dict(avg_metric))
torch.save(avg_metric, "bytedance_avg_metric.pt")