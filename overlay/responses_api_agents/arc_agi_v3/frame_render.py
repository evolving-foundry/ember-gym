# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""ARC-AGI-3 frame rendering.

Upstream frames are 64x64 grids of 4-bit colour indices (integers 0-15), not
images. Visual models need a rendered PNG, so this module maps the indices
through the official 16-colour palette (mirrored from the arcprize
ARC-AGI-3-Agents SDK, ``agents/templates/multimodal.py``) and upscales with
nearest-neighbour resampling to keep the pixel art crisp.
"""

import base64
import io


PALETTE: tuple[tuple[int, int, int], ...] = (
    (0xFF, 0xFF, 0xFF),  # 0  white
    (0xCC, 0xCC, 0xCC),  # 1  off-white
    (0x99, 0x99, 0x99),  # 2  neutral light
    (0x66, 0x66, 0x66),  # 3  neutral
    (0x33, 0x33, 0x33),  # 4  off-black
    (0x00, 0x00, 0x00),  # 5  black
    (0xE5, 0x3A, 0xA3),  # 6  magenta
    (0xFF, 0x7B, 0xCC),  # 7  magenta light
    (0xF9, 0x3C, 0x31),  # 8  red
    (0x1E, 0x93, 0xFF),  # 9  blue
    (0x88, 0xD8, 0xF1),  # 10 blue light
    (0xFF, 0xDC, 0x00),  # 11 yellow
    (0xFF, 0x85, 0x1B),  # 12 orange
    (0x92, 0x12, 0x31),  # 13 maroon
    (0x4F, 0xCC, 0x30),  # 14 green
    (0xA3, 0x56, 0xD6),  # 15 purple
)

_SCALE = 4  # 64 px -> 256 px


def validate_grid(grid: list[list[int]]) -> None:
    """Reject malformed frames early: rectangular, values 0-15."""
    if not grid or not all(isinstance(row, list) for row in grid):
        raise ValueError("frame grid must be a non-empty list of rows")
    width = len(grid[0])
    if any(len(row) != width for row in grid):
        raise ValueError("frame grid rows must all have the same width")
    if any(not isinstance(cell, int) or not 0 <= cell <= 15 for row in grid for cell in row):
        raise ValueError("frame grid cell values must be integers in [0, 15]")


def grid_to_png_data_url(grid: list[list[int]], scale: int = _SCALE) -> str:
    """Render a frame grid to a ``data:image/png;base64,...`` URI for image_url content."""
    validate_grid(grid)

    # PIL is imported lazily so environments without it (prepare.py, data tooling) can still
    # import the module; rendering is only needed inside the agent rollout.
    from PIL import Image

    height, width = len(grid), len(grid[0])
    image = Image.new("RGB", (width, height))
    for y, row in enumerate(grid):
        for x, cell in enumerate(row):
            image.putpixel((x, y), PALETTE[cell])
    image = image.resize((width * scale, height * scale), Image.NEAREST)

    buffer = io.BytesIO()
    image.save(buffer, format="PNG", optimize=True)
    return f"data:image/png;base64,{base64.b64encode(buffer.getvalue()).decode('ascii')}"
