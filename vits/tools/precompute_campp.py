"""
Precompute CAM++ speaker embeddings for VITS training.

Usage:
  python tools/precompute_campp.py --filepaths_and_text filelist.txt --output_dir data/campp
  
Expected filelist format:
  /path/to/audio.wav|speaker_id
  /path/to/audio2.wav|speaker_id2
"""
import os
import sys
import argparse
import torch
import torch.nn.functional as F
from pathlib import Path
import numpy as np
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from third_party.seedvc.campplus import CAMPPlus
from utils import load_wav_to_torch
from mel_processing import spectrogram_torch


def get_mel_from_wav(wav, filter_length, hop_length, win_length, sampling_rate):
    """Convert waveform to mel spectrogram."""
    mel = spectrogram_torch(
        wav.unsqueeze(0),  # Add batch dim
        filter_length=filter_length,
        hop_length=hop_length,
        win_length=win_length,
        center=False
    )
    return mel.squeeze(0).transpose(0, 1)  # [T, mel_bins]


def extract_campplus_embedding(mel, model, device):
    """Extract CAM++ embedding from mel spectrogram."""
    with torch.no_grad():
        mel = mel.unsqueeze(0).to(device)  # [1, T, mel_bins]
        mel_len = torch.tensor([mel.shape[1]], device=device)
        embedding = model(mel, x_lens=mel_len)  # [1, embedding_size]
    return embedding.squeeze(0)  # [embedding_size]


def main():
    parser = argparse.ArgumentParser(
        description='Precompute CAM++ speaker embeddings for VITS'
    )
    parser.add_argument('--filepaths_and_text', type=str, required=True,
                        help='Path to file with audio paths and speaker IDs')
    parser.add_argument('--output_dir', type=str, default='data/campp',
                        help='Output directory for embeddings')
    parser.add_argument('--campplus_ckpt', type=str,
                        default='../seed-vc/campplus_cn_common.bin',
                        help='Path to CAM++ checkpoint')
    parser.add_argument('--hparams', type=str, default='configs/default.json',
                        help='Path to hparams config')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device to use for extraction')
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    spk_emb_dir = output_dir / 'spk2emb'
    spk_emb_dir.mkdir(parents=True, exist_ok=True)
    
    # Load hyperparameters
    from utils import get_hparams
    hparams = get_hparams(args.hparams)
    
    # Load CAM++ model
    print("Loading CAM++ model...")
    campplus_model = CAMPPlus(feat_dim=80, embedding_size=192)
    campplus_ckpt = torch.load(args.campplus_ckpt, map_location='cpu')
    campplus_model.load_state_dict(campplus_ckpt)
    campplus_model.eval()
    campplus_model.to(args.device)
    print(f"CAM++ model loaded on {args.device}")
    
    # Load file list
    with open(args.filepaths_and_text, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # Extract embeddings
    speaker_embeddings = {}
    print(f"Processing {len(lines)} audio files...")
    
    for line in tqdm(lines, desc='Extracting embeddings'):
        parts = line.strip().split('|')
        if len(parts) < 2:
            print(f"Warning: invalid line: {line}")
            continue
        
        audio_path = parts[0]
        speaker_id = parts[1]
        
        try:
            # Load audio
            wav, sr = load_wav_to_torch(audio_path)
            if sr != hparams.sampling_rate:
                # Resample if needed (would need librosa)
                import librosa
                wav = librosa.resample(wav.numpy(), orig_sr=sr, target_sr=hparams.sampling_rate)
                wav = torch.from_numpy(wav).float()
            
            # Extract mel spectrogram
            mel = get_mel_from_wav(
                wav, hparams.filter_length, hparams.hop_length,
                hparams.win_length, hparams.sampling_rate
            )
            
            # Extract CAM++ embedding
            embedding = extract_campplus_embedding(mel, campplus_model, args.device)
            
            # Store embedding
            if speaker_id not in speaker_embeddings:
                speaker_embeddings[speaker_id] = []
            speaker_embeddings[speaker_id].append(embedding.cpu().numpy())
            
        except Exception as e:
            print(f"Error processing {audio_path}: {e}")
            continue
    
    # Average embeddings per speaker
    print("Averaging embeddings per speaker...")
    spk_embeddings_dict = {}
    for spk_id, embeddings in speaker_embeddings.items():
        avg_emb = np.mean(embeddings, axis=0)
        spk_embeddings_dict[spk_id] = torch.from_numpy(avg_emb).float()
    
    # Save as pt file
    output_path = output_dir / 'spk2emb.pt'
    torch.save(spk_embeddings_dict, output_path)
    print(f"Saved speaker embeddings to {output_path}")
    print(f"Total speakers: {len(spk_embeddings_dict)}")


if __name__ == '__main__':
    main()
