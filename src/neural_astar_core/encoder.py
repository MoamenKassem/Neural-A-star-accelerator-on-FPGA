"""The CNN that produces the guidance map.

What this network actually does
-------------------------------
It runs **once per problem**, not once per expansion.  Input is a 2-channel
H x W image -- the obstacle map, and the sum of the start and goal one-hot maps.
Output is a single H x W map of values in (0, 1), which A* then uses as the
per-cell step cost.

Cells the optimal path is likely to pass through get a low cost, everything
else gets a high one, so the search is pulled along promising corridors and
away from dead ends.  That is the whole of "neural" in Neural A*: the network
never sees the open list, never ranks nodes, and is never consulted inside the
search loop.

Architecture (fixed by the shipped checkpoint)
----------------------------------------------
Five 3x3 convolutions with channels ``2 -> 32 -> 64 -> 128 -> 256 -> 1``, each
followed by BatchNorm and ReLU except the last, which has BatchNorm only.  A
sigmoid is applied in ``forward``.

Cost, which matters for the FPGA decision:

===================  ===========
Parameters           389,952
MACs per 32x32 map   ~399 million
MACs per 64x64 map   ~1.6 billion
===================  ===========

For scale, a PYNQ-Z2 (Zynq-7020) has 220 DSPs and 140 BRAM-36K.  At 8-bit,
weights alone are ~390 KB against 630 KB of BRAM.  It fits, barely, but
:func:`build_encoder` takes a ``channels`` argument precisely so the model can be retrained
a narrower version if the board turns out to be the constraint.
"""

from __future__ import annotations

import re
from glob import glob
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

# Channel widths of the shipped checkpoint.  These must not change when
# to load `mazes_032_moore_c8`; change them (and retrain) to shrink the model.
CHECKPOINT_CHANNELS = (32, 64, 128, 256)


class GuidanceEncoder(nn.Module):
    """CNN encoder producing a per-cell cost map in (0, 1).

    Layer naming is kept as ``self.model.<i>`` with a plain ``nn.Sequential`` so
    that the upstream checkpoint's ``encoder.model.0.weight`` style keys load
    without renaming.
    """

    def __init__(
        self,
        input_dim: int = 2,
        channels: tuple[int, ...] = CHECKPOINT_CHANNELS,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.channels = tuple(channels)

        widths = [input_dim] + list(channels) + [1]
        blocks: list[nn.Module] = []
        for i in range(len(widths) - 1):
            blocks.append(nn.Conv2d(widths[i], widths[i + 1], 3, 1, 1))
            blocks.append(nn.BatchNorm2d(widths[i + 1]))
            blocks.append(nn.ReLU())
        # Drop the final ReLU: the sigmoid in forward() provides the output
        # nonlinearity, and a ReLU before it would clamp half the range away.
        self.model = nn.Sequential(*blocks[:-1])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Args: ``x`` of shape ``(B, 2, H, W)``. Returns ``(B, 1, H, W)`` in (0, 1)."""
        return torch.sigmoid(self.model(x))

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def macs_per_map(self, height: int, width: int) -> int:
        """Multiply-accumulates for one forward pass, assuming 'same' padding."""
        widths = [self.input_dim] + list(self.channels) + [1]
        return sum(
            widths[i] * widths[i + 1] * 9 * height * width
            for i in range(len(widths) - 1)
        )


def encoder_input(
    obstacle_map: np.ndarray,
    start_map: np.ndarray,
    goal_map: np.ndarray,
) -> torch.Tensor:
    """Assemble the 2-channel network input for a single problem.

    Channel 0 is the obstacle map; channel 1 is ``start + goal``.  Upstream
    calls this format ``"m+"``.  Both start and goal share a channel, which is
    why the network can tell them apart only through context -- an odd design
    choice, but changing it would invalidate the checkpoint.
    """
    stacked = np.stack(
        [
            np.asarray(obstacle_map, dtype=np.float32),
            np.asarray(start_map, dtype=np.float32)
            + np.asarray(goal_map, dtype=np.float32),
        ]
    )
    return torch.from_numpy(stacked).unsqueeze(0)


def load_encoder_from_checkpoint(
    checkpoint_dir: str,
    channels: tuple[int, ...] = CHECKPOINT_CHANNELS,
    device: str = "cpu",
) -> GuidanceEncoder:
    """Load the encoder weights out of a PyTorch Lightning checkpoint.

    Args:
        checkpoint_dir: a directory searched recursively for ``*.ckpt``.  The
            lexicographically last match is used, matching upstream.
        channels: must match the checkpoint's architecture.
        device: where to place the model.

    Two upstream breakages are handled here:

    1. ``torch.load`` defaults to ``weights_only=True`` from torch 2.6 onward,
       which refuses Lightning checkpoints because they pickle the training
       config.  This asks for ``weights_only=False`` explicitly.  That is safe for
       a checkpoint produced in this project and would not be for a downloaded one.
    2. Every key is prefixed ``planner.`` by the Lightning module wrapper, and
       the search module contributes a ``planner.astar.neighbor_filter`` buffer
       that the encoder does not want.  Both are stripped here.
    """
    matches = sorted(glob(f"{checkpoint_dir}/**/*.ckpt", recursive=True))
    if not matches:
        raise FileNotFoundError(f"no .ckpt found under {checkpoint_dir}")
    ckpt_path = matches[-1]

    try:
        raw = torch.load(ckpt_path, map_location=device, weights_only=False)
    except TypeError:
        # torch < 2.0 has no weights_only argument at all.
        raw = torch.load(ckpt_path, map_location=device)

    state_dict = raw["state_dict"] if "state_dict" in raw else raw

    encoder_state = {}
    for key, value in state_dict.items():
        if ".encoder." not in key:
            continue
        encoder_state[re.split(r"\.encoder\.", key, maxsplit=1)[-1]] = value

    if not encoder_state:
        raise RuntimeError(
            f"{ckpt_path} contains no encoder weights; keys look like: "
            f"{list(state_dict)[:5]}"
        )

    model = GuidanceEncoder(input_dim=2, channels=channels)
    missing, unexpected = model.load_state_dict(encoder_state, strict=False)
    if missing or unexpected:
        raise RuntimeError(
            f"checkpoint does not match the model architecture.\n"
            f"  missing:    {sorted(missing)[:6]}\n"
            f"  unexpected: {sorted(unexpected)[:6]}\n"
            f"A changed `channels` requires retraining."
        )

    model.to(device).eval()
    model.checkpoint_path = str(Path(ckpt_path))  # type: ignore[attr-defined]
    return model
