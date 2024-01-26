import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from audiocraft.data.midi_dataset import EnCodecTokenMIDIDataset
from audiocraft.modules.semcodec import SemCodecMidiDecoder, SemCodecOnlyMidi
from audiocraft.losses.note_eval import evaluate, extract_notes
import pretty_midi

import wandb
import matplotlib.pyplot as plt
import muspy

class SemCodecMidiDecoderTrainer:
    def __init__(self, model, optimizer, scheduler, criterion = torch.nn.BCELoss(), device = "cuda"):
        self.model = model.to(device)
        self.optimizer = optimizer
        # self.scheduler = scheduler
        self.criterion = criterion
        self.device = device
        self.iteration = 0

    def train(self, train_loader, epoch, onset_threshold, frame_threshold):
        self.model.train()
        for batch_idx, (encodec_tensor, pr_tensor) in tqdm(enumerate(train_loader)):
            encodec_tensor = encodec_tensor.to(self.device)
            pr_tensor = pr_tensor.to(self.device)
            self.optimizer.zero_grad()
            output = self.model(encodec_tensor)
            prediction = output.round()
            acc = (prediction == pr_tensor).sum().item() / (prediction.shape[0] * prediction.shape[1] * prediction.shape[2] * prediction.shape[3])
            if prediction.sum().item() == 0 or pr_tensor.sum().item() == 0:
                pass
            else: 
                frame_precision = (prediction * pr_tensor).sum().item() / prediction.sum().item()
                frame_recall = (prediction * pr_tensor).sum().item() / pr_tensor.sum().item()
                if frame_precision + frame_recall == 0:
                    frame_f1 = 0
                else:
                    frame_f1 = 2 * frame_precision * frame_recall / (frame_precision + frame_recall)
                wandb.log({"frame_f1": frame_f1, "frame_precision": frame_precision, "frame_recall": frame_recall}, step=self.iteration)
            loss = self.criterion(output, pr_tensor)
            loss.backward()
            self.optimizer.step()

            #mir_eval
            if self.iteration % 2000 == 0 and batch_idx == 0 and self.iteration != 0:
                note_precision, note_recall, note_f1, note_overlap, note_with_offsets_precision, note_with_offsets_recall, note_with_offsets_f1, note_with_offsets_overlap = 0, 0, 0, 0, 0, 0, 0, 0
                frame_f1, frame_precision, frame_recall = 0, 0, 0
                frame_accuracy, frame_substitution, frame_miss_error, frame_false_alarm_error = 0, 0, 0, 0
                frame_total_error, frame_chroma_precision, frame_chroma_recall, frame_chroma_accuracy = 0, 0, 0, 0
                frame_chroma_substitution_error, frame_chroma_miss_error, frame_chroma_false_alarm_error, frame_chroma_total_error = 0, 0, 0, 0
                
                for pred, label in zip(output, pr_tensor):
                    metrics = evaluate(pred, label, onset_threshold, frame_threshold)
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
                note_precision /= output.shape[0]
                note_recall /= output.shape[0]
                note_f1 /= output.shape[0]
                note_overlap /= output.shape[0]
                note_with_offsets_precision /= output.shape[0]
                note_with_offsets_recall /= output.shape[0]
                note_with_offsets_f1 /= output.shape[0]
                note_with_offsets_overlap /= output.shape[0]
                frame_f1 /= output.shape[0]
                frame_precision /= output.shape[0]
                frame_recall /= output.shape[0]
                frame_accuracy /= output.shape[0]
                frame_substitution /= output.shape[0]
                frame_miss_error /= output.shape[0]
                frame_false_alarm_error /= output.shape[0]
                frame_total_error /= output.shape[0]
                frame_chroma_precision /= output.shape[0]
                frame_chroma_recall /= output.shape[0]
                frame_chroma_accuracy /= output.shape[0]
                frame_chroma_substitution_error /= output.shape[0]
                frame_chroma_miss_error /= output.shape[0]
                frame_chroma_false_alarm_error /= output.shape[0]
                frame_chroma_total_error /= output.shape[0]
                print({'metric/note/precision': note_precision, "metric/note/recall": note_recall, "metric/note/f1": note_f1, "metric/note/overlap": note_overlap, "metric/note-with-offsets/precision": note_with_offsets_precision, "metric/note-with-offsets/recall": note_with_offsets_recall, "metric/note-with-offsets/f1": note_with_offsets_f1, "metric/note-with-offsets/overlap": note_with_offsets_overlap})
                wandb.log({"epoch": epoch, 'metric/note/precision': note_precision, "metric/note/recall": note_recall, "metric/note/f1": note_f1, "metric/note/overlap": note_overlap, "metric/note-with-offsets/precision": note_with_offsets_precision, "metric/note-with-offsets/recall": note_with_offsets_recall, "metric/note-with-offsets/f1": note_with_offsets_f1, "metric/note-with-offsets/overlap": note_with_offsets_overlap, 'metric/frame/f1': frame_f1, 'metric/frame/precision': frame_precision, 'metric/frame/recall': frame_recall, 'metric/frame/accuracy': frame_accuracy, 'metric/frame/substitution_error': frame_substitution, 'metric/frame/miss_error': frame_miss_error, 'metric/frame/false_alarm_error': frame_false_alarm_error, 'metric/frame/total_error': frame_total_error, 'metric/frame/chroma_precision': frame_chroma_precision, 'metric/frame/chroma_recall': frame_chroma_recall, 'metric/frame/chroma_accuracy': frame_chroma_accuracy, 'metric/frame/chroma_substitution_error': frame_chroma_substitution_error, 'metric/frame/chroma_miss_error': frame_chroma_miss_error, 'metric/frame/chroma_false_alarm_error': frame_chroma_false_alarm_error, 'metric/frame/chroma_total_error': frame_chroma_total_error}, step=self.iteration)    
            if batch_idx == 0:
                print('\n')
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.5f}'.format(epoch, batch_idx * len(encodec_tensor), len(train_loader.dataset), 100. * batch_idx / len(train_loader), loss.item()))

            wandb.log({"loss": loss.item(), "epoch": epoch, "batch_idx": batch_idx, "accuracy": acc}, step=self.iteration)
            self.iteration += 1
        # self.scheduler.step()
                    
    def validate(self, val_loader, epoch, onset_threshold, frame_threshold):
        self.model.eval()
        val_loss = 0

        with torch.no_grad():
            for batch_idx, (encodec_tensor, pr_tensor) in tqdm(enumerate(val_loader)):
                encodec_tensor = encodec_tensor.to(self.device)
                pr_tensor = pr_tensor.to(self.device)
                output = self.model(encodec_tensor)
                prediction = output.round()
                acc = (prediction == pr_tensor).sum().item() / (prediction.shape[0] * prediction.shape[1] * prediction.shape[2] * prediction.shape[3])
                if prediction.sum().item() == 0 or pr_tensor.sum().item() == 0:
                    pass
                else: 
                    val_frame_precision = (prediction * pr_tensor).sum().item() / prediction.sum().item()
                    val_frame_recall = (prediction * pr_tensor).sum().item() / pr_tensor.sum().item()
                    if val_frame_precision + val_frame_recall == 0:
                        val_frame_f1 = 0
                    else:
                        val_frame_f1 = 2 * val_frame_precision * val_frame_recall / (val_frame_precision + val_frame_recall)
                    wandb.log({"val_frame_f1": val_frame_f1, "val_frame_precision": val_frame_precision, "val_frame_recall": val_frame_recall}, step=self.iteration)
                val_loss += self.criterion(output, pr_tensor).item()
                
                if self.iteration % 2000 == 0 and epoch != 0 and batch_idx==0:
                    note_precision, note_recall, note_f1, note_overlap, note_with_offsets_precision, note_with_offsets_recall, note_with_offsets_f1, note_with_offsets_overlap = 0, 0, 0, 0, 0, 0, 0, 0
                    frame_f1, frame_precision, frame_recall = 0, 0, 0
                    frame_accuracy, frame_substitution, frame_miss_error, frame_false_alarm_error = 0, 0, 0, 0
                    frame_total_error, frame_chroma_precision, frame_chroma_recall, frame_chroma_accuracy = 0, 0, 0, 0
                    frame_chroma_substitution_error, frame_chroma_miss_error, frame_chroma_false_alarm_error, frame_chroma_total_error = 0, 0, 0, 0
                    
                    for pred, label in zip(output, pr_tensor):
                        metrics = evaluate(pred, label, onset_threshold, frame_threshold)
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

                    note_precision /= output.shape[0]
                    note_recall /= output.shape[0]
                    note_f1 /= output.shape[0]
                    note_overlap /= output.shape[0]
                    note_with_offsets_precision /= output.shape[0]
                    note_with_offsets_recall /= output.shape[0]
                    note_with_offsets_f1 /= output.shape[0]
                    note_with_offsets_overlap /= output.shape[0]
                    frame_f1 /= output.shape[0]
                    frame_precision /= output.shape[0]
                    frame_recall /= output.shape[0]
                    frame_accuracy /= output.shape[0]
                    frame_substitution /= output.shape[0]
                    frame_miss_error /= output.shape[0]
                    frame_false_alarm_error /= output.shape[0]
                    frame_total_error /= output.shape[0]
                    frame_chroma_precision /= output.shape[0]
                    frame_chroma_recall /= output.shape[0]
                    frame_chroma_accuracy /= output.shape[0]
                    frame_chroma_substitution_error /= output.shape[0]
                    frame_chroma_miss_error /= output.shape[0]
                    frame_chroma_false_alarm_error /= output.shape[0]
                    frame_chroma_total_error /= output.shape[0]
                    wandb.log({'val_metric/note/precision': note_precision, "val_metric/note/recall": note_recall, "val_metric/note/f1": note_f1, "val_metric/note/overlap": note_overlap, "val_metric/note-with-offsets/precision": note_with_offsets_precision, "val_metric/note-with-offsets/recall": note_with_offsets_recall, "val_metric/note-with-offsets/f1": note_with_offsets_f1, "val_metric/note-with-offsets/overlap": note_with_offsets_overlap, "val_metric/frame/f1": frame_f1, "val_metric/frame/precision": frame_precision, "val_metric/frame/recall": frame_recall, "val_metric/frame/accuracy": frame_accuracy, "val_metric/frame/substitution_error": frame_substitution, "val_metric/frame/miss_error": frame_miss_error, "val_metric/frame/false_alarm_error": frame_false_alarm_error, "val_metric/frame/total_error": frame_total_error, "val_metric/frame/chroma_precision": frame_chroma_precision, "val_metric/frame/chroma_recall": frame_chroma_recall, "val_metric/frame/chroma_accuracy": frame_chroma_accuracy, "val_metric/frame/chroma_substitution_error": frame_chroma_substitution_error, "val_metric/frame/chroma_miss_error": frame_chroma_miss_error, "val_metric/frame/chroma_false_alarm_error": frame_chroma_false_alarm_error, "val_metric/frame/chroma_total_error": frame_chroma_total_error}, step=self.iteration)
        val_loss /= len(val_loader.dataset) / val_loader.batch_size
        print('\nValidation set: Average loss: {:.5f}\n'.format(val_loss))
        wandb.log({"val_loss": val_loss, "epoch": epoch, "val_accuracy": acc}, step=self.iteration)
        return val_loss
    
    def test(self, test_loader, onset_threshold, frame_threshold):
        self.model.eval()
        test_loss = 0
        note_precision, note_recall, note_f1, note_overlap, note_with_offsets_precision, note_with_offsets_recall, note_with_offsets_f1, note_with_offsets_overlap = 0, 0, 0, 0, 0, 0, 0, 0
        frame_f1, frame_precision, frame_recall = 0, 0, 0
        frame_accuracy, frame_substitution, frame_miss_error, frame_false_alarm_error = 0, 0, 0, 0
        frame_total_error, frame_chroma_precision, frame_chroma_recall, frame_chroma_accuracy = 0, 0, 0, 0
        frame_chroma_substitution_error, frame_chroma_miss_error, frame_chroma_false_alarm_error, frame_chroma_total_error = 0, 0, 0, 0
            
        with torch.no_grad():
            for encodec_tensor, pr_tensor in test_loader:
                encodec_tensor = encodec_tensor.to(self.device)
                pr_tensor = pr_tensor.to(self.device)
                output = self.model(encodec_tensor)
                prediction = output.round()
                acc = (prediction == pr_tensor).sum().item() / (prediction.shape[0] * prediction.shape[1] * prediction.shape[2] * prediction.shape[3])
                
                for pred, label in zip(output, pr_tensor):
                    metrics = evaluate(pred, label, onset_threshold, frame_threshold)
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
                test_loss += self.criterion(output, pr_tensor).item()

            note_precision /= len(test_loader.dataset)
            note_recall /= len(test_loader.dataset)
            note_f1 /= len(test_loader.dataset)
            note_overlap /= len(test_loader.dataset)
            note_with_offsets_precision /= len(test_loader.dataset)
            note_with_offsets_recall /= len(test_loader.dataset)
            note_with_offsets_f1 /= len(test_loader.dataset)
            note_with_offsets_overlap /= len(test_loader.dataset)
            frame_f1 /= len(test_loader.dataset)
            frame_precision /= len(test_loader.dataset)
            frame_recall /= len(test_loader.dataset)
            frame_accuracy /= len(test_loader.dataset)
            frame_substitution /= len(test_loader.dataset)
            frame_miss_error /= len(test_loader.dataset)
            frame_false_alarm_error /= len(test_loader.dataset)
            frame_total_error /= len(test_loader.dataset)
            frame_chroma_precision /= len(test_loader.dataset)
            frame_chroma_recall /= len(test_loader.dataset)
            frame_chroma_accuracy /= len(test_loader.dataset)
            frame_chroma_substitution_error /= len(test_loader.dataset)
            frame_chroma_miss_error /= len(test_loader.dataset)
            frame_chroma_false_alarm_error /= len(test_loader.dataset)
            frame_chroma_total_error /= len(test_loader.dataset)

        test_loss /= len(test_loader.dataset) / test_loader.batch_size
        print('\nTest set: Average loss: {:.5f}\n'.format(test_loss))
        wandb.log({"test_loss": test_loss, "test_accuracy": acc, 'test_metric/note/precision': note_precision, "test_metric/note/recall": note_recall, "test_metric/note/f1": note_f1, "test_metric/note/overlap": note_overlap, "test_metric/note-with-offsets/precision": note_with_offsets_precision, "test_metric/note-with-offsets/recall": note_with_offsets_recall, "test_metric/note-with-offsets/f1": note_with_offsets_f1, "test_metric/note-with-offsets/overlap": note_with_offsets_overlap, 'test_metric/frame/f1': frame_f1, 'test_metric/frame/precision': frame_precision, 'test_metric/frame/recall': frame_recall, 'test_metric/frame/accuracy': frame_accuracy, 'test_metric/frame/substitution_error': frame_substitution, 'test_metric/frame/miss_error': frame_miss_error, 'test_metric/frame/false_alarm_error': frame_false_alarm_error, 'test_metric/frame/total_error': frame_total_error, 'test_metric/frame/chroma_precision': frame_chroma_precision, 'test_metric/frame/chroma_recall': frame_chroma_recall, 'test_metric/frame/chroma_accuracy': frame_chroma_accuracy, 'test_metric/frame/chroma_substitution_error': frame_chroma_substitution_error, 'test_metric/frame/chroma_miss_error': frame_chroma_miss_error, 'test_metric/frame/chroma_false_alarm_error': frame_chroma_false_alarm_error, 'test_metric/frame/chroma_total_error': frame_chroma_total_error}, step=self.iteration)
        
        return test_loss

    def val_inference(self, validset, idx, save_target, epoch):
        self.model.eval()
        with torch.no_grad():
            encodec_tensor, pr_tensor = validset[idx]
            encodec_tensor = encodec_tensor.to("cuda")
            pr_tensor = pr_tensor.to("cuda")
            output = self.model(encodec_tensor.unsqueeze(0))
            prediction = output.round()
            pred_notes = extract_notes(prediction[0,0].T, prediction[0,1].T, None)
            pred_midi = pretty_midi.PrettyMIDI()
            pred_midi.instruments.append(pretty_midi.Instrument(0))
            for i, pitch in enumerate(pred_notes[0]):
                pred_midi.instruments[0].notes.append(pretty_midi.Note(
                    velocity=100,
                    pitch=pitch,
                    start=pred_notes[1][i,0]/50,
                    end=pred_notes[1][i,1]/50
                ))
            pred_wav = muspy.from_pretty_midi(pred_midi).synthesize()
            wandb.log({f"val_inference/{idx}/wav": wandb.Audio(pred_wav.T[0], caption=f"pred_wav_{epoch}", sample_rate=44100)}, step=self.iteration)
            plt.figure(figsize = (25,10))
            pred_onset_plt = plt.imshow(prediction[0,0].cpu().numpy(), aspect='auto', origin='lower', interpolation='none')
            wandb.log({f"val_inference/{idx}/onset": wandb.Image(pred_onset_plt, caption=f"pred_onset_{epoch}")}, step=self.iteration)
            plt.close('all')
            plt.figure(figsize = (25,10))
            pred_sustain_plt = plt.imshow(prediction[0,1].cpu().numpy(), aspect='auto', origin='lower', interpolation='none')
            wandb.log({f"val_inference/{idx}/sustain": wandb.Image(pred_sustain_plt, caption=f"pred_sustain_{epoch}")}, step=self.iteration)    
            plt.close('all')    

            if save_target:
                target_notes = extract_notes(pr_tensor[0].T, pr_tensor[1].T, None)
                target_midi = pretty_midi.PrettyMIDI()
                target_midi.instruments.append(pretty_midi.Instrument(0))
                for i, pitch in enumerate(target_notes[0]):
                    target_midi.instruments[0].notes.append(pretty_midi.Note(
                        velocity=100,
                        pitch=pitch,
                        start=target_notes[1][i,0]/50,
                        end=target_notes[1][i,1]/50
                    ))
                target_wav = muspy.from_pretty_midi(target_midi).synthesize()
                wandb.log({f"val_inference/{idx}/target/wav": wandb.Audio(target_wav.T[0], caption=f"target_wav_{epoch}", sample_rate=44100)}, step=self.iteration)
                plt.figure(figsize = (25,10))
                target_onset_plt = plt.imshow(pr_tensor[0].cpu().numpy(), aspect='auto', origin='lower', interpolation='none')
                wandb.log({f"val_inference/{idx}/target/onset": wandb.Image(target_onset_plt, caption=f"target_onset_{epoch}")}, step=self.iteration)
                plt.close('all')
                plt.figure(figsize = (25,10))
                target_sustain_plt = plt.imshow(pr_tensor[1].cpu().numpy(), aspect='auto', origin='lower', interpolation='none')
                wandb.log({f"val_inference/{idx}/target/sustain": wandb.Image(target_sustain_plt, caption=f"target_sustain_{epoch}")}, step=self.iteration)
                plt.close('all')
                
def main():
    epochs = 100000
    lr = 1e-4
    onset_threshold = 0.3
    frame_threshold = 0.3

    loaded = torch.load('/home/jongmin/userdata/MAESTRO/maestro-v3.0.0/maestro-v3.0.0_split.pt')
    train_list = loaded["train"]
    validation_list = loaded["validation"]
    test_list = loaded["test"]

    path = '/home/jongmin/userdata/MAESTRO/maestro-v3.0.0'
    trainset = EnCodecTokenMIDIDataset(train_list)
    validset = EnCodecTokenMIDIDataset(validation_list)
    testset = EnCodecTokenMIDIDataset(test_list)

    train_dataloader = DataLoader(trainset, batch_size=64, shuffle=True)
    valid_dataloader = DataLoader(validset, batch_size=64, shuffle=True)
    test_dataloader = DataLoader(testset, batch_size=64, shuffle=True)

    model = SemCodecOnlyMidi().to("cuda")
    
    # Freezing quantizer
    for param in model.quantizer.parameters():
        param.requires_grad = False
        
    # model.load_state_dict(torch.load('/home/jongmin/userdata/audiocraft/semcodec_decoder_chocolate-voice-22_2500.pt'))

    run = wandb.init(
        project="SemCodecMidiDecoder_3CNN_GRU_EncodecResidualVectorQuantizer",
        config = {
            "epochs": epochs,
            "lr": lr,
            "onset_threshold": onset_threshold,
            "frame_threshold": frame_threshold,
        },
        # id = "mqmjg2n9",
        # resume = "must"
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95), eps=1e-08, weight_decay=0.01)

    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.1, last_epoch=-1)
    save_target = True
    trainer = SemCodecMidiDecoderTrainer(model, optimizer, None)
    print(run.name)
    for i in tqdm(range(epochs)):
        trainer.train(train_dataloader, i, onset_threshold, frame_threshold)
        if i % 50 == 0 and i != 0:
            trainer.validate(valid_dataloader, i, onset_threshold, frame_threshold)
        if i % 100 == 0 and i != 0:
            trainer.val_inference(validset, 0, save_target, i)
            trainer.val_inference(validset, 1, save_target, i)
            trainer.val_inference(validset, 2, save_target, i)
            save_target = False
            # trainer.test(test_dataloader, onset_threshold, frame_threshold)
            torch.save([model.state_dict(), optimizer.state_dict()], f'/home/jongmin/userdata/audiocraft/semcodec_decoder_{run.name}_{i}.pt')
    trainer.test(test_dataloader)

if __name__ == '__main__':
    main()
