from piano_transcription_inference import PianoTranscription

transcriptor = PianoTranscription(
    device="cuda", checkpoint_path="./model.pth"
)