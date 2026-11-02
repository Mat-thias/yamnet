"""
The power map detector: the PyTorch YAMNet port, frozen, with a small head on its embeddings.

YAMNet is imported from ../pytorch_dir rather than copied, so the validated port stays the
only implementation. Its modules import each other by bare name (`from dataset import ...`),
so that directory goes on sys.path, and no module here may reuse one of those names.
"""
import os
import sys
from typing import Dict, Tuple

import torch
# loads triton, which segfaults if tensorflow (loaded by YamNet.from_tf_weights) is already
# in the process, as optimizers import it lazily. Loaded first, both work.
import torch._dynamo
from torch import nn

REPO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TF_DIR = os.path.join(REPO_DIR, "tensorflow_dir")
sys.path.append(os.path.join(REPO_DIR, "pytorch_dir"))

from model import YamNet
from params import Params


class PowerMapHead(nn.Module):
    """
    Small classifier on YAMNet's 1024-d patch embeddings.

    One independent output per class like YAMNet's own head, so an alarm sounding over a
    running generator can be both, and negative is every output off.
    """

    def __init__(self, in_features:int = 1024, num_classes:int = 2, hidden:int = 256, dropout:float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden, num_classes)
        )

    def forward(self, embeddings:torch.Tensor) -> torch.Tensor:
        """Returns the logits, sigmoid them for probabilities."""
        return self.net(embeddings)


class PowerMapDetector(nn.Module):
    """
    YAMNet, frozen, and the power map head: waveform in, per-patch logits out.

    Only the head trains. YAMNet runs without gradients and stays in eval mode even when the
    detector is put in train mode, so its batch norms keep the statistics it was trained with.
    """

    def __init__(self, yamnet:YamNet, head:PowerMapHead):
        super().__init__()
        self.yamnet = yamnet.requires_grad_(False)
        self.head = head

    def train(self, mode:bool = True) -> "PowerMapDetector":
        super().train(mode)
        self.yamnet.eval()
        return self

    def forward(self, waveform:torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        (batch, samples) 16kHz mono waveform -> the head's logits and YAMNet's own class scores,
        both (batch, n_patches, n). A 15600 sample waveform is exactly one patch.
        """
        with torch.no_grad():
            output = self.yamnet(waveform)
        batch, n_patches, _ = output["prediction"].size()
        return {
            "logits": self.head(output["embeddings"]).view(batch, n_patches, -1),
            "yamnet_scores": output["prediction"]
        }


def load_detector(file:str, device:str = "cpu") -> Tuple[PowerMapDetector, Dict]:
    """
    Loads a detector saved by train.py with its classes and thresholds. YAMNet's weights are
    in the file too, so unlike YamNet.from_tf_weights this needs no tensorflow.
    """
    saved = torch.load(file, map_location=device)
    head = PowerMapHead(num_classes=len(saved["classes"]), hidden=saved["hidden"], dropout=saved["dropout"])
    detector = PowerMapDetector(YamNet(Params()), head)
    detector.load_state_dict(saved["state_dict"])
    return detector.to(device).eval(), saved
