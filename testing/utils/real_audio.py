"""
real audio module, uses Wav2Vec2 ASR model
"""

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import os
import tempfile

import torch
import torchaudio
from torchaudio.pipelines import WAV2VEC2_ASR_BASE_960H

# for cleaner data pass-through
@dataclass
class RealAudioInputs:
    log_probs: torch.Tensor  # (B, T, V)
    input_lengths: torch.Tensor  # (B,)
    tokens: list[str]  # vocab
    blank_idx: int
    sample_rate: int
    audio_path: str

# a classic audio sample
def get_sample_audio_url() -> str:
    return "https://www.voiptroubleshooter.com/open_speech/american/OSR_us_000_0010_8k.wav"

def download_sample_audio(cache_dir) -> Path:
    
    if cache_dir is None:
        cache_dir = Path(tempfile.gettempdir()) / "ctc_decoder_test_audio"
    
    cache_dir.mkdir(parents=True, exist_ok=True)
    sample_path = cache_dir / "sample_audio.wav"
    
    if sample_path.exists():
        return sample_path
    
    # get the sample
    url = get_sample_audio_url()
    
    try:
        import urllib.request
        urllib.request.urlretrieve(url, str(sample_path))
    except Exception as e:

        # synthetic waveform alternative
        print(f"error on sample download: {e}, generating synthetic audio")

        sample_rate = 16000
        duration = 3.0  #s 
        t = torch.linspace(0, duration, int(sample_rate * duration))
        
        # speech like waveform
        waveform = (
            0.3 * torch.sin(2 * 3.14159 * 200 * t) +  # fundamental
            0.2 * torch.sin(2 * 3.14159 * 400 * t) +  # harmonic
            0.1 * torch.sin(2 * 3.14159 * 600 * t) +  # harmonic
            0.05 * torch.randn_like(t)  # noise
        )

        envelope = torch.ones_like(t)
        envelope[:int(0.1 * sample_rate)] = torch.linspace(0, 1, int(0.1 * sample_rate))
        envelope[-int(0.1 * sample_rate):] = torch.linspace(1, 0, int(0.1 * sample_rate))
        waveform = waveform * envelope
        waveform = waveform.unsqueeze(0) 
        torchaudio.save(str(sample_path), waveform, sample_rate)
    
    return sample_path


def load_wav2vec2_model(device: torch.device = None):
    if device is None:
        device = torch.device("cpu")
    
    bundle = WAV2VEC2_ASR_BASE_960H
    model = bundle.get_model().to(device)
    model.eval()
    
    return model, bundle

# get real vocab
def get_wav2vec2_tokens() -> list[str]:
    bundle = WAV2VEC2_ASR_BASE_960H
    labels = bundle.get_labels()
    return list(labels)


def load_audio(
    audio_path: str,
    target_sample_rate: int = 16000,
) -> tuple[torch.Tensor, int]:

    waveform, sample_rate = torchaudio.load(audio_path)
    
    # resample audio if necessary
    if sample_rate != target_sample_rate:
        resampler = torchaudio.transforms.Resample(
            orig_freq=sample_rate,
            new_freq=target_sample_rate,
        )
        waveform = resampler(waveform)
        sample_rate = target_sample_rate
    
    # stereo audio
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    
    return waveform, sample_rate


def process_audio_with_wav2vec2(
    audio_path: str,
    device: torch.device = None,
    batch_size: int = 1,
) -> RealAudioInputs:
    
    if device is None:
        device = torch.device("cpu")

    model, bundle = load_wav2vec2_model(device)
    sample_rate = bundle.sample_rate
    
    waveform, _ = load_audio(audio_path, target_sample_rate=sample_rate)
    waveform = waveform.to(device)
    
    with torch.no_grad():
        emissions, lengths = model(waveform)
        # emissions shape: (1, T, V)
        log_probs = torch.log_softmax(emissions, dim=-1)
    
    if batch_size > 1:
        log_probs = log_probs.repeat(batch_size, 1, 1)
        lengths = lengths.repeat(batch_size)
    
    tokens = get_wav2vec2_tokens()
    blank_idx = 0
    
    return RealAudioInputs(
        log_probs=log_probs,
        input_lengths=lengths,
        tokens=tokens,
        blank_idx=blank_idx,
        sample_rate=sample_rate,
        audio_path=audio_path,
    )


def generate_real_audio_inputs(
    audio_path,
    device: torch.device = None,
    batch_size: int = 1,
    use_sample: bool = True,
) -> RealAudioInputs:
    # option to use custom audio

    if device is None:
        device = torch.device("cpu")
    
    if audio_path is None:
        if use_sample:
            audio_path = str(download_sample_audio())
        else:
            raise ValueError("No audio_path provided and use_sample=False")
    
    if not Path(audio_path).exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")
    
    return process_audio_with_wav2vec2(
        audio_path=audio_path,
        device=device,
        batch_size=batch_size,
    )


def format_wav2vec2_outputs(
    sequences: torch.Tensor,
    lengths: torch.Tensor,
    tokens: list[str],
    blank_idx: int,
) -> list[str]:
    # formatting to use real vocab
    results = []
    
    for seq, length in zip(sequences, lengths):
        valid_length = int(length)
        if valid_length < 0:
            valid_length = 0
        valid_length = min(valid_length, seq.numel())
        
        trimmed = seq[:valid_length].tolist()
        
        chars = []
        for idx in trimmed:
            if idx == blank_idx or idx < 0 or idx >= len(tokens):
                continue
            token = tokens[idx]
            if token == '|':
                chars.append(' ')
            elif token in ('<s>', '</s>', '<pad>', '<unk>'):
                continue
            else:
                chars.append(token)
        
        text = ''.join(chars).strip()
        
        while '  ' in text:
            text = text.replace('  ', ' ')
        
        results.append(text)
    
    return results


def format_reference_outputs_wav2vec2(
    ref_output: list,
    tokens: list[str],
    blank_idx: int,
    top_k: int,
) -> list[list[str]]:
    # formatting to use real vocab
    formatted = []
    
    for hypotheses in ref_output:
        sample_texts = []
        
        for h in hypotheses[:top_k]:
            seq = getattr(h, "tokens", [])
            
            chars = []
            for idx in seq:
                if idx == blank_idx or idx < 0 or idx >= len(tokens):
                    continue
                token = tokens[idx]
                if token == '|':
                    chars.append(' ')
                elif token in ('<s>', '</s>', '<pad>', '<unk>'):
                    continue
                else:
                    chars.append(token)
            
            text = ''.join(chars).strip()
            while '  ' in text:
                text = text.replace('  ', ' ')
            
            sample_texts.append(text)
        
        formatted.append(sample_texts)
    
    return formatted


def format_candidate_outputs_wav2vec2(
    candidate_sequences: torch.Tensor,
    candidate_lengths: torch.Tensor,
    tokens: list[str],
    blank_idx: int,
    top_k: int,
) -> list[list[str]]:
    # formatting to use real vocab
    formatted = []
    
    for sample_hyps, sample_lengths in zip(candidate_sequences, candidate_lengths):
        sample_texts = []
        
        for seq, length in zip(sample_hyps[:top_k], sample_lengths[:top_k]):
            valid_length = int(length)
            if valid_length < 0:
                valid_length = 0
            valid_length = min(valid_length, seq.numel())
            
            trimmed = seq[:valid_length].tolist()
            
            # real vocab
            chars = []
            for idx in trimmed:
                if idx == blank_idx or idx < 0 or idx >= len(tokens):
                    continue
                token = tokens[idx]
                if token == '|':
                    chars.append(' ')
                elif token in ('<s>', '</s>', '<pad>', '<unk>'):
                    continue
                else:
                    chars.append(token)
            
            text = ''.join(chars).strip()
            while '  ' in text:
                text = text.replace('  ', ' ')
            
            sample_texts.append(text)
        
        formatted.append(sample_texts)
    
    return formatted

