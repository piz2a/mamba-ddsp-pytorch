import torch
import librosa
import numpy as np
import os
import warnings

def compute_spectral_centroid(hex_y, win_length, hop_length, sample_rate):
    window = torch.hann_window(
        win_length,
        dtype=hex_y.dtype,
        device=hex_y.device,
    )
    spectrum = torch.stft(
        hex_y,
        n_fft=win_length,
        hop_length=hop_length,
        win_length=win_length,
        center=True,
        return_complex=True,
        window=window,
    ).abs()
    freqs = torch.linspace(
        0.0,
        sample_rate / 2,
        spectrum.shape[1],
        dtype=hex_y.dtype,
        device=hex_y.device,
    ).reshape(1, -1, 1)
    numerator = (spectrum * freqs).sum(dim=1)
    denominator = spectrum.sum(dim=1).clamp_min(1e-8)
    return numerator / denominator

# from @caillonantoine ircarm/ddsp_pytorch, used to verify the pure pytorch implementation below
def compute_loudness_ptddsp(signal, sampling_rate, block_size, n_fft):
    S = librosa.stft(
        signal,
        n_fft=n_fft,
        hop_length=block_size,
        win_length=n_fft,
        center=True,
        pad_mode="constant",
        window= "hann"
    )
    print(S.shape)
    print(S.mean(0))
    S = np.log(abs(S) + 1e-7)
    f = librosa.fft_frequencies(sr=sampling_rate, n_fft=n_fft)
    a_weight = librosa.A_weighting(f)
    S = S + a_weight.reshape(-1, 1)
    S = np.mean(S, 0)
    print(S.shape)
    return S

# pure pytorch implementation of compute_loudness. 
# adapted from @caillonantoine ircarm/ddsp_pytorch
def compute_loudness(signal,sample_rate,hop_length,n_fft):
    S = torch.stft(
        signal,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=n_fft,
        center=True,
        return_complex=True,
        pad_mode="constant",
        window=torch.hann_window(n_fft, dtype=signal.dtype, device=signal.device)
    )
    S = torch.log(torch.abs(S) + 1e-7)
    f = librosa.fft_frequencies(sr=sample_rate, n_fft=n_fft)
    a_weight = librosa.A_weighting(f)
    a_weight = torch.as_tensor(a_weight, dtype=S.dtype, device=S.device)

    S = S + a_weight.reshape(1,-1,1)
    S =torch.mean(S, 1)
    return S

def compute_rms(y, window_size,hop_frames):
    # warn if hop_frames is larger than window_size
    if hop_frames > window_size:
        print("Warning: hop_frames is larger than window_size. This may cause unexpected behavior.")
    rms = []
    for i in range(y.shape[0]):
        frames = y[i].unfold(0, window_size, hop_frames)
        ld = torch.sqrt(torch.mean(frames**2, dim=1))
        ld = ld[None,...]
        rms.append(ld)
    rms = torch.cat(rms, dim=0)
    return rms

def compute_pitch(y, sample_rate,fmin, fmax, hop_frames, device=None,batch_size=None,pad=False, return_probabilities=False, algorithm=None):
    PITCH_EXTRACTION_ALGORITHM = (
        algorithm
        or os.environ.get("DDSP_GUITAR_PITCH_EXTRACTOR")
        or "crepe"
    ).lower()
    pitch = []
    periodicity = []
    probabilities = []

    def use_fallback(exc):
        allow_fallback = os.environ.get("DDSP_GUITAR_ALLOW_PITCH_FALLBACK", "1") != "0"
        if not allow_fallback:
            raise exc
        warnings.warn(
            f"{PITCH_EXTRACTION_ALGORITHM} pitch extraction is unavailable ({exc}). "
            "Falling back to librosa.pyin. Set DDSP_GUITAR_ALLOW_PITCH_FALLBACK=0 "
            "to fail instead.",
            RuntimeWarning,
        )
        return "librosa_pyin"

    if PITCH_EXTRACTION_ALGORITHM=="penn":
        try:
            import penn
            gpu=device
            checkpoint=penn.DEFAULT_CHECKPOINT
            interp_unvoiced_at = None
            hopsize_s=hop_frames/sample_rate

            for i in range(y.shape[0]):
                p,p2 = penn.from_audio(
                    y[i][None,...],
                    sample_rate,
                    hopsize=hopsize_s,
                    fmin=fmin,
                    fmax=fmax,
                    checkpoint=checkpoint,
                    batch_size=batch_size,
                    pad=pad,
                    interp_unvoiced_at=interp_unvoiced_at,
                    gpu=gpu)

                pitch.append(p.cpu())
                periodicity.append(p2.cpu())
        except (ImportError, OSError) as exc:
            PITCH_EXTRACTION_ALGORITHM = use_fallback(exc)

    if PITCH_EXTRACTION_ALGORITHM=="crepe":
        try:
            import torchcrepe
        except (ImportError, OSError) as exc:
            PITCH_EXTRACTION_ALGORITHM = use_fallback(exc)

    if PITCH_EXTRACTION_ALGORITHM=="crepe":
        # Here we'll use a 5 millisecond hop length
        hop_frames= hop_frames
        # Provide a sensible frequency range for your domain (upper limit is 2006 Hz)
        # This would be a reasonable range for speech
    
        # Select a model capacity--one of "tiny" or "full"
        model = 'full'
        # Choose a device to use for inference
        # Pick a batch size that doesn't cause memory errors on your gpu
        # Compute pitch using first gpu
        for i in range(y.shape[0]):
            if return_probabilities:
                p,p2, probs = torchcrepe.predict(
                                        y[i][None,...],
                                        sample_rate,
                                        hop_frames,
                                        fmin,
                                        fmax,
                                        model,
                                        batch_size=batch_size,
                                        device=device,
                                        return_periodicity=True,
                                        return_probabilities=return_probabilities,
                                        pad=pad
                )
                probabilities.append(probs)
            else:
                p,p2= torchcrepe.predict(
                                        y[i][None,...],
                                        sample_rate,
                                        hop_frames,
                                        fmin,
                                        fmax,
                                        model,
                                        batch_size=batch_size,
                                        device=device,
                                        return_periodicity=True,
                                        pad=pad
                )
            pitch.append(p.cpu())
            periodicity.append(p2.cpu())

    elif PITCH_EXTRACTION_ALGORITHM in ["librosa", "librosa_pyin", "pyin"]:
        frame_length = max(2048, int(2 ** np.ceil(np.log2(hop_frames * 4))))
        for i in range(y.shape[0]):
            channel = y[i].detach().cpu().float().numpy()
            p, voiced_flag, voiced_prob = librosa.pyin(
                channel,
                fmin=fmin,
                fmax=fmax,
                sr=sample_rate,
                frame_length=frame_length,
                hop_length=hop_frames,
                center=pad,
                fill_na=0.0,
            )
            if p is None:
                p = np.zeros(0, dtype=np.float32)
                voiced_prob = np.zeros(0, dtype=np.float32)
            p = np.nan_to_num(p, nan=0.0).astype(np.float32)
            voiced_prob = np.nan_to_num(voiced_prob, nan=0.0).astype(np.float32)
            pitch.append(torch.from_numpy(p)[None, ...])
            periodicity.append(torch.from_numpy(voiced_prob)[None, ...])
            if return_probabilities:
                probabilities.append(torch.from_numpy(voiced_prob)[None, ...])
    else:
        raise ValueError(f"Unknown pitch extraction algorithm: {PITCH_EXTRACTION_ALGORITHM}")


    pitch = torch.cat(pitch, dim=0)
    periodicity = torch.cat(periodicity, dim=0)

    if return_probabilities:
        if len(probabilities) > 0 and all(p.ndim == probabilities[0].ndim for p in probabilities):
            probabilities = torch.cat(probabilities, dim=0)
        return pitch, periodicity, probabilities

    return pitch, periodicity

def median_filtering(x, window_size):
    new_x = x.clone()
    x = x.clone()
    # pad x
    x = torch.nn.functional.pad(x, (window_size//2,window_size//2), mode='reflect')
    for i in range(new_x.shape[1]):
        new_x[:,i] = torch.median(x[:,i:i+window_size],dim=1)[0]
    return new_x
    
def compute_pseudo_velocity(midi_activity, audio_loudness):
    assert midi_activity.shape == audio_loudness.shape
    # for every midi pitch, get the max loudness
    midi_loudness = torch.zeros_like(midi_activity)
    note_start_stops = []
    is_active = False
    for i in range(midi_activity.shape[0]):
        if midi_activity[i]:
            if not is_active:
                note_start_stops.append([i])
                is_active = True
        else:
            if is_active:
                note_start_stops[-1].append(i)
                is_active = False

    # Check if the last note is still active at the end
    if is_active:
        note_start_stops[-1].append(midi_activity.shape[0])

    for note_start_stop in note_start_stops:
        midi_loudness[note_start_stop[0]:note_start_stop[1]] = torch.max(audio_loudness[note_start_stop[0]:note_start_stop[1]])
    return midi_loudness
