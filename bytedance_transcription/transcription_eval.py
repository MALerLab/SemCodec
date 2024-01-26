from piano_transcription_inference import PianoTranscription, sample_rate

transcriptor = PianoTranscription(
    device="cuda", checkpoint_path="./model.pth"
)

import torch
import librosa
import os
from tqdm import tqdm
import note_eval
import wandb

encodec_sr = 32000

run = wandb.init(
        project="SemCodec-ByteDance_transcription-comparison"
    )

count = 0

note_precision, note_recall, note_f1, note_overlap, note_with_offsets_precision, note_with_offsets_recall, note_with_offsets_f1, note_with_offsets_overlap = 0, 0, 0, 0, 0, 0, 0, 0
frame_f1, frame_precision, frame_recall = 0, 0, 0
frame_accuracy, frame_substitution, frame_miss_error, frame_false_alarm_error = 0, 0, 0, 0
frame_total_error, frame_chroma_precision, frame_chroma_recall, frame_chroma_accuracy = 0, 0, 0, 0
frame_chroma_substitution_error, frame_chroma_miss_error, frame_chroma_false_alarm_error, frame_chroma_total_error = 0, 0, 0, 0

# note_precisions, note_recalls, note_f1s, note_overlaps, note_with_offsets_precisions, note_with_offsets_recalls, note_with_offsets_f1s, note_with_offsets_overlaps = [], [], [], [], [], [], [], []
# frame_f1s, frame_precisions, frame_recalls = [], [], []
# frame_accuracies, frame_substitutions, frame_miss_errors, frame_false_alarm_errors = [], [], [], []
# frame_total_errors, frame_chroma_precisions, frame_chroma_recalls, frame_chroma_accuracies = [], [], [], []
# frame_chroma_substitution_errors, frame_chroma_miss_errors, frame_chroma_false_alarm_errors, frame_chroma_total_errors = [], [], [], []

for dirpath, dirnames, filenames in os.walk("/home/jongmin/userdata/MAESTRO/maestro-v3.0.0/"):
    for filename in tqdm([f for f in filenames if f.endswith("_encodec_decoded.pth")]):
        pr_filename = filename.replace("_encodec_decoded.pth", "_bt_pianoroll.pkl")
        loaded = torch.load(os.path.join(dirpath, filename))
        pr_loaded = torch.load(os.path.join(dirpath, pr_filename))
        if loaded["len"] != pr_loaded["len"]:
            loaded.pop((loaded["len"]-2)*30)
        for i in range(pr_loaded["len"]-1):
            count += 1
            loaded_tmp = librosa.resample(loaded[i*30].numpy().squeeze(), encodec_sr, sample_rate)
            out = transcriptor.transcribe(loaded_tmp, 'transcription.mid')
            metrics = note_eval.evaluate(torch.Tensor([out['output_dict']['onset_output'], out['output_dict']['frame_output']]).permute(0,2,1), pr_loaded[i*30], 0.5, 0.5)
            note_precision += metrics['metric/note/precision'][0]
            note_recall += metrics['metric/note/recall'][0]
            note_f1 += metrics['metric/note/f1'][0]
            note_overlap += metrics['metric/note/overlap'][0]
            note_with_offsets_precision += metrics['metric/note-with-offsets/precision'][0]
            note_with_offsets_recall += metrics['metric/note-with-offsets/recall'][0]
            note_with_offsets_f1 += metrics['metric/note-with-offsets/f1'][0]
            note_with_offsets_overlap += metrics['metric/note-with-offsets/overlap'][0]
            frame_f1 += metrics['metric/frame/f1'][0]
            frame_precision += metrics['metric/frame/precision'][0]
            frame_recall += metrics['metric/frame/recall'][0]
            frame_accuracy += metrics['metric/frame/accuracy'][0]
            frame_substitution += metrics['metric/frame/substitution_error'][0]
            frame_miss_error += metrics['metric/frame/miss_error'][0]
            frame_false_alarm_error += metrics['metric/frame/false_alarm_error'][0]
            frame_total_error += metrics['metric/frame/total_error'][0]
            frame_chroma_precision += metrics['metric/frame/chroma_precision'][0]
            frame_chroma_recall += metrics['metric/frame/chroma_recall'][0]
            frame_chroma_accuracy += metrics['metric/frame/chroma_accuracy'][0]
            frame_chroma_substitution_error += metrics['metric/frame/chroma_substitution_error'][0]
            frame_chroma_miss_error += metrics['metric/frame/chroma_miss_error'][0]
            frame_chroma_false_alarm_error += metrics['metric/frame/chroma_false_alarm_error'][0]
            frame_chroma_total_error += metrics['metric/frame/chroma_total_error'][0]
            # note_precisions.append(metrics['metric/note/precision'][0])
            # note_recalls.append(metrics['metric/note/recall'][0])
            # note_f1s.append(metrics['metric/note/f1'][0])
            # note_overlaps.append(metrics['metric/note/overlap'][0])
            # note_with_offsets_precisions.append(metrics['metric/note-with-offsets/precision'][0])
            # note_with_offsets_recalls.append(metrics['metric/note-with-offsets/recall'][0])
            # note_with_offsets_f1s.append(metrics['metric/note-with-offsets/f1'][0])
            # note_with_offsets_overlaps.append(metrics['metric/note-with-offsets/overlap'][0])
            # frame_f1s.append(metrics['metric/frame/f1'][0])
            # frame_precisions.append(metrics['metric/frame/precision'][0])
            # frame_recalls.append(metrics['metric/frame/recall'][0])
            # frame_accuracies.append(metrics['metric/frame/accuracy'][0])
            # frame_substitutions.append(metrics['metric/frame/substitution_error'][0])
            # frame_miss_errors.append(metrics['metric/frame/miss_error'][0])
            # frame_false_alarm_errors.append(metrics['metric/frame/false_alarm_error'][0])
            # frame_total_errors.append(metrics['metric/frame/total_error'][0])
            # frame_chroma_precisions.append(metrics['metric/frame/chroma_precision'][0])
            # frame_chroma_recalls.append(metrics['metric/frame/chroma_recall'][0])
            # frame_chroma_accuracies.append(metrics['metric/frame/chroma_accuracy'][0])
            # frame_chroma_substitution_errors.append(metrics['metric/frame/chroma_substitution_error'][0])
            # frame_chroma_miss_errors.append(metrics['metric/frame/chroma_miss_error'][0])
            # frame_chroma_false_alarm_errors.append(metrics['metric/frame/chroma_false_alarm_error'][0])
            # frame_chroma_total_errors.append(metrics['metric/frame/chroma_total_error'][0])

            wandb.log({"filename": os.path.join(dirpath, filename), "chunk_idx": i*30, "note_precision": metrics['metric/note/precision'][0], "note_recall": metrics['metric/note/recall'][0], "note_f1": metrics['metric/note/f1'][0], "note_overlap": metrics['metric/note/overlap'][0], "note_with_offsets_precision": metrics['metric/note-with-offsets/precision'][0], "note_with_offsets_recall": metrics['metric/note-with-offsets/recall'][0], "note_with_offsets_f1": metrics['metric/note-with-offsets/f1'][0], "note_with_offsets_overlap": metrics['metric/note-with-offsets/overlap'][0], "frame_f1": metrics['metric/frame/f1'][0], "frame_precision": metrics['metric/frame/precision'][0], "frame_recall": metrics['metric/frame/recall'][0], "frame_accuracy": metrics['metric/frame/accuracy'][0], "frame_substitution": metrics['metric/frame/substitution_error'][0], "frame_miss_error": metrics['metric/frame/miss_error'][0], "frame_false_alarm_error": metrics['metric/frame/false_alarm_error'][0], "frame_total_error": metrics['metric/frame/total_error'][0], "frame_chroma_precision": metrics['metric/frame/chroma_precision'][0], "frame_chroma_recall": metrics['metric/frame/chroma_recall'][0], "frame_chroma_accuracy": metrics['metric/frame/chroma_accuracy'][0], "frame_chroma_substitution_error": metrics['metric/frame/chroma_substitution_error'][0], "frame_chroma_miss_error": metrics['metric/frame/chroma_miss_error'][0], "frame_chroma_false_alarm_error": metrics['metric/frame/chroma_false_alarm_error'][0], "frame_chroma_total_error": metrics['metric/frame/chroma_total_error'][0]})
        count += 1
        loaded_tmp = librosa.resample(loaded[-1].numpy().squeeze(), encodec_sr, sample_rate)
        out = transcriptor.transcribe(loaded_tmp, 'transcription.mid')
        metrics = note_eval.evaluate(torch.Tensor([out['output_dict']['onset_output'], out['output_dict']['frame_output']]).permute(0,2,1), pr_loaded[i*30], 0.5, 0.5)
        note_precision += metrics['metric/note/precision'][0]
        note_recall += metrics['metric/note/recall'][0]
        note_f1 += metrics['metric/note/f1'][0]
        note_overlap += metrics['metric/note/overlap'][0]
        note_with_offsets_precision += metrics['metric/note-with-offsets/precision'][0]
        note_with_offsets_recall += metrics['metric/note-with-offsets/recall'][0]
        note_with_offsets_f1 += metrics['metric/note-with-offsets/f1'][0]
        note_with_offsets_overlap += metrics['metric/note-with-offsets/overlap'][0]
        frame_f1 += metrics['metric/frame/f1'][0]
        frame_precision += metrics['metric/frame/precision'][0]
        frame_recall += metrics['metric/frame/recall'][0]
        frame_accuracy += metrics['metric/frame/accuracy'][0]
        frame_substitution += metrics['metric/frame/substitution_error'][0]
        frame_miss_error += metrics['metric/frame/miss_error'][0]
        frame_false_alarm_error += metrics['metric/frame/false_alarm_error'][0]
        frame_total_error += metrics['metric/frame/total_error'][0]
        frame_chroma_precision += metrics['metric/frame/chroma_precision'][0]
        frame_chroma_recall += metrics['metric/frame/chroma_recall'][0]
        frame_chroma_accuracy += metrics['metric/frame/chroma_accuracy'][0]
        frame_chroma_substitution_error += metrics['metric/frame/chroma_substitution_error'][0]
        frame_chroma_miss_error += metrics['metric/frame/chroma_miss_error'][0]
        frame_chroma_false_alarm_error += metrics['metric/frame/chroma_false_alarm_error'][0]
        frame_chroma_total_error += metrics['metric/frame/chroma_total_error'][0]
        # note_precisions.append(metrics['metric/note/precision'][0])
        # note_recalls.append(metrics['metric/note/recall'][0])
        # note_f1s.append(metrics['metric/note/f1'][0])
        # note_overlaps.append(metrics['metric/note/overlap'][0])
        # note_with_offsets_precisions.append(metrics['metric/note-with-offsets/precision'][0])
        # note_with_offsets_recalls.append(metrics['metric/note-with-offsets/recall'][0])
        # note_with_offsets_f1s.append(metrics['metric/note-with-offsets/f1'][0])
        # note_with_offsets_overlaps.append(metrics['metric/note-with-offsets/overlap'][0])
        # frame_f1s.append(metrics['metric/frame/f1'][0])
        # frame_precisions.append(metrics['metric/frame/precision'][0])
        # frame_recalls.append(metrics['metric/frame/recall'][0])
        # frame_accuracies.append(metrics['metric/frame/accuracy'][0])
        # frame_substitutions.append(metrics['metric/frame/substitution_error'][0])
        # frame_miss_errors.append(metrics['metric/frame/miss_error'][0])
        # frame_false_alarm_errors.append(metrics['metric/frame/false_alarm_error'][0])
        # frame_total_errors.append(metrics['metric/frame/total_error'][0])
        # frame_chroma_precisions.append(metrics['metric/frame/chroma_precision'][0])
        # frame_chroma_recalls.append(metrics['metric/frame/chroma_recall'][0])
        # frame_chroma_accuracies.append(metrics['metric/frame/chroma_accuracy'][0])
        # frame_chroma_substitution_errors.append(metrics['metric/frame/chroma_substitution_error'][0])
        # frame_chroma_miss_errors.append(metrics['metric/frame/chroma_miss_error'][0])
        # frame_chroma_false_alarm_errors.append(metrics['metric/frame/chroma_false_alarm_error'][0])
        # frame_chroma_total_errors.append(metrics['metric/frame/chroma_total_error'][0])
        
        wandb.log({"filename": os.path.join(dirpath, filename), "chunk_idx": -1,"note_precision": metrics['metric/note/precision'][0], "note_recall": metrics['metric/note/recall'][0], "note_f1": metrics['metric/note/f1'][0], "note_overlap": metrics['metric/note/overlap'][0], "note_with_offsets_precision": metrics['metric/note-with-offsets/precision'][0], "note_with_offsets_recall": metrics['metric/note-with-offsets/recall'][0], "note_with_offsets_f1": metrics['metric/note-with-offsets/f1'][0], "note_with_offsets_overlap": metrics['metric/note-with-offsets/overlap'][0], "frame_f1": metrics['metric/frame/f1'][0], "frame_precision": metrics['metric/frame/precision'][0], "frame_recall": metrics['metric/frame/recall'][0], "frame_accuracy": metrics['metric/frame/accuracy'][0], "frame_substitution": metrics['metric/frame/substitution_error'][0], "frame_miss_error": metrics['metric/frame/miss_error'][0], "frame_false_alarm_error": metrics['metric/frame/false_alarm_error'][0], "frame_total_error": metrics['metric/frame/total_error'][0], "frame_chroma_precision": metrics['metric/frame/chroma_precision'][0], "frame_chroma_recall": metrics['metric/frame/chroma_recall'][0], "frame_chroma_accuracy": metrics['metric/frame/chroma_accuracy'][0], "frame_chroma_substitution_error": metrics['metric/frame/chroma_substitution_error'][0], "frame_chroma_miss_error": metrics['metric/frame/chroma_miss_error'][0], "frame_chroma_false_alarm_error": metrics['metric/frame/chroma_false_alarm_error'][0], "frame_chroma_total_error": metrics['metric/frame/chroma_total_error'][0]})
note_precision /= count
note_recall /= count
note_f1 /= count
note_overlap /= count
note_with_offsets_precision /= count
note_with_offsets_recall /= count
note_with_offsets_f1 /= count
note_with_offsets_overlap /= count
frame_f1 /= count
frame_precision /= count
frame_recall /= count
frame_accuracy /= count
frame_substitution /= count
frame_miss_error /= count
frame_false_alarm_error /= count
frame_total_error /= count
frame_chroma_precision /= count
frame_chroma_recall /= count
frame_chroma_accuracy /= count
frame_chroma_substitution_error /= count
frame_chroma_miss_error /= count
frame_chroma_false_alarm_error /= count
frame_chroma_total_error /= count
wandb.log({"avg/note_precision": note_precision, "avg/note_recall": note_recall, "avg/note_f1": note_f1, "avg/note_overlap": note_overlap, "avg/note_with_offsets_precision": note_with_offsets_precision, "avg/note_with_offsets_recall": note_with_offsets_recall, "avg/note_with_offsets_f1": note_with_offsets_f1, "avg/note_with_offsets_overlap": note_with_offsets_overlap, "avg/frame_f1": frame_f1, "avg/frame_precision": frame_precision, "avg/frame_recall": frame_recall, "avg/frame_accuracy": frame_accuracy, "avg/frame_substitution": frame_substitution, "avg/frame_miss_error": frame_miss_error, "avg/frame_false_alarm_error": frame_false_alarm_error, "avg/frame_total_error": frame_total_error, "avg/frame_chroma_precision": frame_chroma_precision, "avg/frame_chroma_recall": frame_chroma_recall, "avg/frame_chroma_accuracy": frame_chroma_accuracy, "avg/frame_chroma_substitution_error": frame_chroma_substitution_error, "avg/frame_chroma_miss_error": frame_chroma_miss_error, "avg/frame_chroma_false_alarm_error": frame_chroma_false_alarm_error, "avg/frame_chroma_total_error": frame_chroma_total_error})
