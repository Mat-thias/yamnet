import os
import math
import tempfile
from typing import Dict, List, Optional, Tuple

import av
import librosa
import numpy as np
import soundfile as sf
import yt_dlp
import matplotlib.pyplot as plt
import torch
from datasets import load_dataset, Audio


def get_audio_waveform(
    id:str,
    start_sec:float, 
    end_sec:float,
    sr:int = 16000,
    save_file:Optional[str] = None
) -> np.ndarray:
    """
    Loads and audio data to its waveform at a given sampling rate.

    Read the audio file is it exist locally or from youtube and cuts the clip to the
    given duratrion. It resamples to sr sampling rate and saves the clip is a path is
    provided.
    """
    if os.path.exists(id):
        audio, native_sr = librosa.load(id, mono=True, sr=None)
    else:
        with tempfile.TemporaryDirectory() as tmp:
            # Download the video's audio track.
            opts = {"format": "bestaudio", "outtmpl": f"{tmp}/audio", "quiet": True, "cookiefile": os.path.expanduser("~/yt_cookies.txt")}
            with yt_dlp.YoutubeDL(opts) as ydl:
                ydl.download([f"https://www.youtube.com/watch?v={id}"])

            # Decode it (YouTube serves webm/m4a, which librosa can't read).
            with av.open(f"{tmp}/audio") as f:
                native_sr = f.streams.audio[0].rate
                audio = np.concatenate([frame.to_ndarray() for frame in f.decode(audio=0)], axis=1)
                audio = librosa.to_mono(audio)
    
    # Mono, cut the segment, resample to `sr`.
    audio = audio[int(start_sec * native_sr):int(end_sec * native_sr)]
    audio_waveform = librosa.resample(audio, orig_sr=native_sr, target_sr=sr)

    # Optionally save_file the clip as a WAV file.
    if save_file:
        dirname = os.path.dirname(save_file)
        if dirname: os.makedirs(dirname, exist_ok=True)
        sf.write(save_file, audio_waveform, sr)
    return audio_waveform


def linear_to_mel_weight_matrix(n_mels:int, n_freqs:int, sample_rate:float, freq_min:float, freq_max:float):
    """
    Port of tf.signal.linear_to_mel_weight_matrix, shape (n_freqs, n_mels).
    Unlike torchaudio, the triangles are straight lines in mel (not Hz), and the DC bin is always zero.
    """
    hz_to_mel = lambda hz: 1127.0 * torch.log(1.0 + hz / 700.0)

    # (n_freqs - 1, 1), skipping the DC bin
    freqs_mel = hz_to_mel(torch.linspace(0.0, sample_rate / 2.0, n_freqs)[1:]).unsqueeze(1)
    # (n_mels + 2,) evenly spaced in mel: each triangle is (lower, center, upper)
    edges_mel = torch.linspace(
        hz_to_mel(torch.tensor(float(freq_min))).item(), hz_to_mel(torch.tensor(float(freq_max))).item(), n_mels + 2
    )
    lower, center, upper = edges_mel[:-2], edges_mel[1:-1], edges_mel[2:]

    rising = (freqs_mel - lower) / (center - lower)
    falling = (upper - freqs_mel) / (upper - center)
    weights = torch.clamp(torch.minimum(rising, falling), min=0.0)
    # add the DC bin back as a row of zeros
    return torch.nn.functional.pad(weights, (0, 0, 1, 0))


def wavefrom_to_mel_spectrogram(
    waveform:torch.Tensor,
    sample_rate:int,
    window_length_sec:float,
    hop_size_sec:float,
    frame_window_length_sec:float,
    frame_hop_size_sec:float,
    n_mels:int,
    freq_min:int,
    freq_max:int,
    n_fft:Optional[int] = None,
    log_offset:float = 1e-3,
    pad:bool = True
) -> Dict[str, torch.Tensor]:
    """Returns the framed waveform, its freqency component, its mel bands and its log mel bands"""
    # all must be divisible by hop_size
    assert frame_window_length_sec % hop_size_sec, \
        f"frame_window_length_sec: {frame_window_length_sec} must be divisible by hop_size_sec: {hop_size_sec}"
    assert frame_hop_size_sec % hop_size_sec, \
        f"frame_hop_size_sec: {frame_hop_size_sec} must be divisible by hop_size_sec: {hop_size_sec}"
    
    window_length = int(sample_rate * window_length_sec)
    hop_size = int(sample_rate * hop_size_sec)
    frame_window_length = int(frame_window_length_sec / hop_size_sec)
    frame_hop_size = int(frame_hop_size_sec / hop_size_sec)

    def pad_waveform(waveform:torch.Tensor):
        """Pads waveform such that no valid point is discarded"""
        waveform_len = waveform.size(-1)
        # padding to make the waveform at least 1 frame
        size_of_single_frame = (frame_window_length-1) * hop_size + window_length
        target_len = max(size_of_single_frame, waveform_len)

        # padding to make sure last frame is not discarded
        frame_hop_size_in_waveform = frame_hop_size * hop_size
        n_hops = math.ceil((target_len - size_of_single_frame) / frame_hop_size_in_waveform)
        target_len = size_of_single_frame + n_hops * frame_hop_size_in_waveform
        return torch.nn.functional.pad(waveform, (0, target_len - waveform_len))

    base = 2
    # defualting the num of frequency to the nearest highest power of 2
    n_fft = n_fft or int(math.pow(base, math.ceil(math.log(window_length, base))))
    n_freqs = n_fft//2 + 1

    hann_window = (0.5 + (-0.5) * torch.cos(
        torch.arange(window_length) / window_length * 2 * torch.pi
    )).to(dtype=torch.float32, device=waveform.device)

    melscale_fbanks = linear_to_mel_weight_matrix(
        n_mels=n_mels, n_freqs=n_freqs, sample_rate=sample_rate,
        freq_min=freq_min, freq_max=freq_max
    ).to(waveform.device)

    if pad: waveform = pad_waveform(waveform)
    # (*, n_windows, window_length)
    framed_waveform = waveform.unfold(-1, window_length, hop_size)
    # (*, n_windows, n_fft)
    # torch.stft is not used as its pads each window to the center, yamnet requires right padding
    waveform_freq = torch.fft.rfft(input=framed_waveform * hann_window, n=n_fft).abs()
    # (*, n_windows, n_bins)
    waveform_mel = torch.matmul(waveform_freq, melscale_fbanks)

    # batching the frames
    # (*, n_windows, x) -> (*, n_windows, frame_window_length)
    def batch_to_frames(waveform:torch.Tensor):
        return waveform.unfold(-2, frame_window_length, frame_hop_size).transpose(-2, -1)

    # return waveform_mel if log else torch.log(waveform_mel + log_offset)
    return {
        "waveform": batch_to_frames(framed_waveform),
        "waveform_freq": batch_to_frames(waveform_freq),
        "waveform_mel": batch_to_frames(waveform_mel),
        "waveform_log_mel": batch_to_frames(torch.log(waveform_mel + log_offset))
    }


def show_spectrogram(spectrogram:torch.Tensor, axis_grid:Tuple[int, int], imshow:bool=True, **kwargs):
    """Plots each patch of a (n_patches, n_frames, n_bins) spectrogram in a grid."""
    assert spectrogram.ndim == 3, \
        f"The spectrogram should be a 3 dim tensor, received {spectrogram.ndim}"
    assert math.prod(axis_grid) == spectrogram.size(0), (
        f"The size of the axis_grid: {math.prod(axis_grid)} has to be equal "
        f"to the number of patches in the spectrogram: {spectrogram.size(0)}"
    )
    spectrogram = spectrogram.detach().cpu()
    # One colour scale for every patch, so they can be compared.
    vmin = spectrogram.min().item()
    vmax = spectrogram.max().item()

    fig, axes = plt.subplots(*axis_grid, squeeze=False, sharex=True, sharey=True)
    for axis, patch in zip(axes.ravel(), spectrogram):
        # (n_frames, n_bins) -> time on x, frequency on y, low frequencies at the bottom.
        if imshow:
            image = axis.imshow(patch.T, aspect="auto", origin="lower", vmin=vmin, vmax=vmax, **kwargs)
        else:
            image = axis.plot()
    fig.supxlabel("Frame")
    fig.supylabel("Frequency bin")
    fig.colorbar(image, ax=axes)
    return fig



class AudioSetDataset(torch.utils.data.Dataset):
    """
    Dataset for HuggingFace AudioSet, returs the waveform and the multi-hot label.
    dataset files can be gotten from
    https://huggingface.co/datasets/agkphysics/AudioSet
    """
    ID_IDX = 0
    START_SEC_IDX = 1
    END_SEC_IDX = 2
    CLASS_IDX = 3

    SKIP_CLASSES = [
        "/m/05zppz", # Male speech, man speaking
        "/m/02zsn", # Female speech, woman speaking
        "/m/04gy_2", # Battle cry
        "/t/dd00003", # Male singing
        "/t/dd00004", # Female singing
        "/t/dd00032" # Funny music
    ]
    def __init__(
        self,
        path:str = "agkphysics/AudioSet",
        name:str = "balanced",
        split:str = "test",
        sr:int = 16000,
        class_mapping:Optional[list[tuple[str, str]]] = None, # list of (audioset_name, class_name)
    ):
        self.sr = sr
        self.audioset_dataset = load_dataset(
            path=path, name=name, split=split,
            trust_remote_code=True
        ).cast_column("audio", Audio(sampling_rate=self.sr))

        self.with_class_mapping = class_mapping is not None
        if self.with_class_mapping:
            self.class_mapping = class_mapping
            self.num_class = len(class_mapping)
            self.class_idx = {audioset_name: i for i, (audioset_name, _) in enumerate(class_mapping)}

            self.valid_indices = list()
            self.classes_indices = list()
            for i, data in enumerate(self.audioset_dataset):
                valid_classes = [
                    c_idx for audioset_name in data["labels"]
                    if (c_idx := self.class_idx.get(audioset_name, None)) is not None
                ]
                if valid_classes:
                    self.valid_indices.append(i)
                    self.classes_indices.append(valid_classes)
                    # data["classes"] = torch.tensor(valid_classes, dtype=torch.long)
                    # valid_row_by_classes.append(data)
        else: self.valid_indices =  list(range(len(self.audioset_dataset)))
        # self.audioset_dataset = list(self.audioset_dataset)

        
    def __len__(self) -> int:
        return len(self.valid_indices)

    def __getitem__(self, idx:int) -> Dict[str, Optional[torch.Tensor]]:
        data = self.audioset_dataset[self.valid_indices[idx]]
        waveform = torch.from_numpy(data["audio"]["array"])
        classes = torch.zeros(self.num_class).scatter_(
            0, torch.tensor(self.classes_indices[idx], dtype=torch.long), 1.0
        ) if self.with_class_mapping else None
        return {
            "waveform" : waveform,
            "classes": classes
        }


    @staticmethod
    def get_collate_fn(sample_duration:float, sr:int):
        """
        Return a collation function which batches the waveform and classes as dict.
        It resizes the waveform to the expected duration at the sample rate by:
            - By trimming the waveform if it is long
            - Center padding the waveform if it short
        """
        expected_len = int(round(sample_duration * sr))
        
        def resize_wave(waveform:torch.Tensor) -> torch.Tensor:
            waveform = waveform[:expected_len]
            pad_size = expected_len - len(waveform)
            left_pad = pad_size // 2
            right_pad = pad_size - left_pad
            return torch.nn.functional.pad(waveform, pad=(left_pad, right_pad))

        def collate_fun(
            batch:List[Dict[str, Optional[torch.Tensor]]]
        ) -> Dict[torch.Tensor, Optional[torch.Tensor]]:
            waveform = torch.stack([
                resize_wave(b["waveform"]) for b in batch
            ])
            classes = torch.stack([b["classes"] for b in batch])\
                if batch[0]["classes"] is not None else None
            return {
                "waveform": waveform,
                "classes": classes
            }
        return collate_fun
    
# class AudioSet(torch.utils.data.Dataset):
#     """
#     Dateset for AudioSet, returs the waveform and the multi-hot label.
#     dataset files can be gotten from
#     https://research.google.com/audioset/download.html
#     """
#     ID_IDX = 0
#     START_SEC_IDX = 1
#     END_SEC_IDX = 2
#     CLASS_IDX = 3

#     SKIP_CLASSES = [
#         "/m/05zppz", # Male speech, man speaking
#         "/m/02zsn", # Female speech, woman speaking
#         "/m/04gy_2", # Battle cry
#         "/t/dd00003", # Male singing
#         "/t/dd00004", # Female singing
#         "/t/dd00032" # Funny music
#     ]
#     def __init__(
#         self,
#         file:str,
#         sr:int = 16000,
#         rows_skip:int = 3,
#         class_mapping:Optional[list] = None,
#         download_dir:Optional[str] = None
#     ):
#         self.sr = sr
#         self.download_dir = self.download(file, sr, rows_skip, download_dir)
#         self.with_class_mapping = class_mapping is not None
#         if self.with_class_mapping:
#             self.class_mapping = class_mapping
#             self.num_class = len(class_mapping)
#             self.class_idx = {audioset_name: i for i, (audioset_name, _) in enumerate(class_mapping)}

#         with open(file) as f:
#             file_rows = tuple(list(csv.reader(f)))[rows_skip:]
#             if self.with_class_mapping:
#                 valid_row_by_classes = list()
#                 for row in file_rows:
#                     classes = self.process_audioset_label_to_classes(row[self.CLASS_IDX:])
#                     valid_classes = [c for c in classes if c is not None]
#                     if valid_classes:
#                         valid_row_by_classes.append(row[:self.CLASS_IDX] + valid_classes)
#                 self.rows = valid_row_by_classes
#             else: self.rows = file_rows

#         self.rows = [row for row in self.rows if os.path.exists(self.get_audio_path(self.download_dir, row[self.ID_IDX]))]

#     def process_audioset_label_to_classes(self, audioset_label:str) -> str:
#         return [
#             self.class_idx.get(
#                 audioset_name.replace('"', "").replace("'", "").strip(), None
#             ) for audioset_name in audioset_label
#         ]

#     def __len__(self) -> int:
#         return len(self.rows)

#     def __getitem__(self, idx):
#         row = self.rows[idx]
#         audio_file = self.get_audio_path(self.download_dir, row[self.ID_IDX])
#         waveform, sr = librosa.load(audio_file, sr=None, mono=True)
#         assert sr == self.sr, \
#             f"audio_file {audio_file} has a sample rate of {sr}Hz, expected {self.sr}Hz."
#         waveform = torch.from_numpy(waveform)

#         if self.with_class_mapping:
#             class_idx = torch.tensor(row[self.CLASS_IDX:], dtype=torch.long)
#             classes = torch.zeros(self.num_class).scatter_(
#                 0, class_idx, 1.0
#             )
#             return waveform, classes
        
#         return waveform
    

#     @staticmethod
#     def get_audio_path(dir:str, file:str) -> str:
#         return os.path.join(dir, f"{file}.wav")
    
#     @classmethod
#     def download(
#         cls,
#         file:str,
#         sr:int = 16000,
#         rows_skip:int = 3,
#         download_dir:Optional[str] = None
#     ) -> str:
#         """
#         Downloads audiodata in a give sampling rate from the audioset dataset.

#         It downloads all the audio data and saves them in the download_dir with the id as the file name.
#         If the audio file with the same name exists in the download_dir, it is skipped.
#         """
#         if download_dir is None:
#             dirname = os.path.dirname(file)
#             name_with_ext = os.path.splitext(os.path.basename(file))[0]
#             download_dir = os.path.join(dirname, name_with_ext)
#         os.makedirs(download_dir, exist_ok=True)

#         with open(file) as f:
#             file_rows = tuple(list(csv.reader(f)))[rows_skip:]
#             for row in file_rows:
#                 id = row[cls.ID_IDX]
#                 start_sec = row[cls.START_SEC_IDX]
#                 end_sec = row[cls.END_SEC_IDX]

#                 audio_file_path = cls.get_audio_path(download_dir, id)
#                 if not os.path.exists(audio_file_path):
#                     try:
#                         get_audio_waveform(
#                             id, float(start_sec), float(end_sec), sr, save_file=audio_file_path
#                         )
#                     except yt_dlp.utils.DownloadError: pass

#         return download_dir
    