"""
Precompute Whisper encoder features for semantic learning.

Usage:
  python tools/precompute_whisper.py --filepaths_and_text filelist.txt --output_dir data/whisper
  
Expected filelist format:
  /path/to/audio.wav|speaker_id
"""
import os
import sys
import argparse
import torch
import torchaudio
from pathlib import Path
import numpy as np
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import whisper


def get_whisper_features(wav, model, device, sr=16000):
    """Extract Whisper encoder features from waveform."""
    # Whisper expects 16kHz audio
    if sr != 16000:
        import librosa
        wav = librosa.resample(wav.numpy(), orig_sr=sr, target_sr=16000)
        wav = torch.from_numpy(wav).float()
    
    # Normalize to [-1, 1]
    wav = wav / (torch.abs(wav).max() + 1e-8)
    
    with torch.no_grad():
        mel = whisper.log_mel_spectrogram(wav.to(device))  # [80, T]
        # Add batch dimension
        mel = mel.unsqueeze(0)  # [1, 80, T]
        # Get encoder output
        features = model.encoder(mel)  # [1, T', 384] or [1, T', 768] depending on model size
    
    return features.squeeze(0)  # [T', encoder_dim]


def main():
    parser = argparse.ArgumentParser(
        description='Precompute Whisper encoder features'
    )
    parser.add_argument('--filepaths_and_text', type=str, required=True,
                        help='Path to file with audio paths and speaker IDs')
    parser.add_argument('--output_dir', type=str, default='data/whisper',
                        help='Output directory for features')
    parser.add_argument('--whisper_model', type=str, default='tiny',
                        choices=['tiny', 'base', 'small', 'medium', 'large'],
                        help='Whisper model size')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device to use for extraction')
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load Whisper model
    print(f"Loading Whisper {args.whisper_model} model...")
    model = whisper.load_model(args.whisper_model)
    model.eval()
    model.to(args.device)
    encoder_dim = model.encoder.ln_post.weight.shape[0]
    print(f"Whisper encoder dimension: {encoder_dim}")
    
    # Load file list
    with open(args.filepaths_and_text, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # Extract features
    print(f"Processing {len(lines)} audio files...")
    
    for i, line in enumerate(tqdm(lines, desc='Extracting Whisper features')):
        parts = line.strip().split('|')
        if len(parts) < 2:
            print(f"Warning: invalid line: {line}")
            continue
        
        audio_path = parts[0]
        speaker_id = parts[1]
        
        try:
            # Load audio at 16kHz
            wav, sr = torchaudio.load(audio_path)
            if wav.shape[0] > 1:
                wav = wav.mean(0)  # Mono
            else:
                wav = wav.squeeze(0)
            
            # Extract features
            features = get_whisper_features(wav, model, args.device, sr)
            
            # Save features
            basename = Path(audio_path).stem
            output_path = output_dir / f'{basename}.pt'
            torch.save(features.cpu(), output_path)
            
        except Exception as e:
            print(f"Error processing {audio_path}: {e}")
            continue
    
    print(f"✓ Saved Whisper features to {output_dir}")


if __name__ == '__main__':
    main()
