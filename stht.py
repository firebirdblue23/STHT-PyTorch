import torch

STHT_PEAK_MUL = 0.54 # STHT_PEAK = frequency dimension * STHT_PEAK_MUL

def normalize(x: torch.Tensor, min_val: float = -1.0, max_val: float = 1.0):
    """Normalize STHT spectrograms from [-stht_peak, stht_peak] to [min_val, max_val]"""
    dim_f = x.shape[-2]
    stht_peak = dim_f * STHT_PEAK_MUL
    x_clamped = torch.clamp(x, min=-stht_peak, max=stht_peak)
    # Convert from [-stht_peak, stht_peak] to [0, 1]
    x_normalized_to_unit = (x_clamped + stht_peak) / (2 * stht_peak)
    # Scale to desired output range
    return min_val + (x_normalized_to_unit * (max_val - min_val))

def denormalize(x: torch.Tensor, min_val: float = -1.0, max_val: float = 1.0):
    dim_f = x.shape[-2]
    stht_peak = dim_f * STHT_PEAK_MUL
    val_diff = abs(min_val) + abs(max_val)
    x_clamped = torch.clamp(x, min=min_val, max=max_val)
    
    x_normalized_to_unit = (x_clamped + (val_diff / 2)) / val_diff
    return (x_normalized_to_unit * (2 * stht_peak)) - stht_peak

def compress(x: torch.Tensor, factor: float = 1.0):
    assert factor > 0 and factor <= 1.0, "compression factor must be between 0 and 1"
    signs = torch.sign(x)
    return torch.pow(x*signs, factor)*signs

def decompress(x: torch.Tensor, factor: float = 1.0):
    assert factor > 0 and factor <= 1.0, "compression factor must be between 0 and 1"
    signs = torch.sign(x)
    return torch.pow(x*signs, 1.0/factor)*signs

def transform(x: torch.Tensor, n_fft: int, hop_length: int, win_length: int = None, window: torch.Tensor = None) -> torch.Tensor:
    # Validate input dimensions and parameters
    assert x.dim() == 3, f"Input must be 3D tensor (batch, channel, time), got {x.shape}"
    assert hop_length > 0, "hop_length must be positive"
    assert n_fft > 0, "n_fft must be positive"
    
    device = x.device
    win_length = n_fft if win_length is None else win_length
    assert win_length <= n_fft, "win_length cannot exceed n_fft"
    
    # Create window and ensure proper device placement
    if window is None:
        window = torch.hamming_window(win_length, device=device)
    else:
        window = window.to(device=device)

    batch_size, num_channels, time_len = x.shape
    
    # Combine batch and channels for vectorized processing
    x_reshaped = x.view(batch_size * num_channels, time_len)
    
    # Compute STFT with optimized parameters
    stft_result = torch.stft(
        x_reshaped,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=window,
        center=True,
        pad_mode="reflect",
        onesided=False,
        return_complex=True
    )
    
    # Transform complex output to real representation
    spectrogram = stft_result.real - stft_result.imag
    
    # Restore original dimensions with separated batch and channels
    spectrogram = spectrogram.view(batch_size, num_channels, n_fft, -1)
    return spectrogram


def inverse(x: torch.Tensor, n_fft: int, hop_length: int, 
           win_length: int = None, window: torch.Tensor = None, length: int = None) -> torch.Tensor:
    # Validate input dimensions and parameters
    assert x.dim() == 4, f"Input must be 4D tensor (batch, channel, freq, time), got {x.shape}"
    assert x.size(2) == n_fft, "Frequency dimension must match n_fft"
    assert hop_length > 0, "hop_length must be positive"
    assert n_fft > 0, "n_fft must be positive"
    
    device = x.device
    win_length = n_fft if win_length is None else win_length
    assert win_length <= n_fft, "win_length cannot exceed n_fft"
    
    # Create window and ensure proper device placement
    if window is None:
        window = torch.hamming_window(win_length, device=device)
    else:
        window = window.to(device=device)

    batch_size, num_channels, _, num_frames = x.shape
    
    # Combine batch and channels for vectorized processing
    x_reshaped = x.view(batch_size * num_channels, n_fft, num_frames)
    
    # Precompute frequency mirror indices (saves computation in loop)
    mirrored_indices = (n_fft - torch.arange(n_fft, device=device)) % n_fft
    
    # Perform frequency mirroring efficiently using tensor indexing
    h_mirror = x_reshaped[:, mirrored_indices, :]
    
    # Reconstruct complex spectrogram
    real_part = 0.5 * (x_reshaped + h_mirror)
    imag_part = 0.5 * (h_mirror - x_reshaped)
    complex_spec = torch.complex(real_part, imag_part)
    
    # Compute iSTFT for all channels in one call
    audio = torch.istft(
        complex_spec,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=window,
        center=True,
        onesided=False,
        length=length
    )
    
    # Restore original dimensions
    return audio.view(batch_size, num_channels, -1)