# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Model-agnostic advantage labelling shared by offline value-model pipelines.

RECAP and STEAM produce a per-frame continuous advantage score in very different
ways (RECAP: N-step TD over a regressed value model; STEAM: worst-of-N ensemble
of a progress critic), but they turn that score into a boolean positive/negative
label with the *same* top-fraction quantile rule. This module owns that shared
post-processing — the quantile threshold and the boolean labelling — so both
pipelines (and any future value-model pipeline) keep one definition of it,
independent of how the continuous score was computed.

The only convention that differs is the comparison at the threshold: RECAP uses
``>=`` and STEAM uses ``>``; :func:`apply_boolean_label` exposes that as the
``inclusive`` flag so each caller reproduces its own behaviour verbatim.
"""

import numpy as np


def quantile_threshold(scores, positive_fraction: float) -> float:
    """Threshold whose top ``positive_fraction`` of ``scores`` is positive.

    ``threshold = percentile(scores, (1 - positive_fraction) * 100)`` — e.g.
    ``positive_fraction=0.3`` returns the 70th percentile, so the top 30% of
    scores lie at/above it. This is the exact rule shared by RECAP's
    ``positive_quantile`` and STEAM's ``rollout_quantile`` / ``expert_quantile``.

    Args:
        scores: 1-D array-like of continuous advantage scores.
        positive_fraction: Fraction of top scores to treat as positive, in
            ``(0, 1)``.

    Returns:
        The threshold value as a Python float.
    """
    return float(
        np.percentile(np.asarray(scores), (1.0 - float(positive_fraction)) * 100.0)
    )


def apply_boolean_label(continuous, threshold: float, *, inclusive: bool = True):
    """Boolean positive-advantage label from a continuous score and threshold.

    ``inclusive=True`` → ``continuous >= threshold`` (RECAP convention);
    ``inclusive=False`` → ``continuous > threshold`` (STEAM convention).

    Accepts a numpy array or pandas Series and returns the elementwise boolean
    of the same type. Callers own any data-type-specific override on top of this
    (e.g. forcing every ``sft`` frame True regardless of threshold).
    """
    if inclusive:
        return continuous >= threshold
    return continuous > threshold


__all__ = ["quantile_threshold", "apply_boolean_label"]


def smoothing_window(seconds: float, fps: float) -> int:
    """Odd number of frames covering ``seconds`` at ``fps`` (1 disables)."""
    if seconds <= 0:
        return 1
    frames = max(1, int(round(seconds * fps)))
    return frames if frames % 2 else frames + 1


def smoothed_td_advantages(
    episode_index,
    returns,
    values,
    *,
    lookahead: int,
    window: int,
    return_min: float,
    return_max: float,
):
    """N-step RECAP advantages from values averaged over ``window`` frames.

    Recomputes ``A_t = norm(R_t - R_{t+N}) + V_s(o_{t+N}) - V_s(o_t)`` for
    ``gamma = 1``, where ``V_s`` is a centred moving average of the per-frame
    value prediction inside each episode (edge-padded, never across episodes)
    and ``V_s(o_{t+N}) = 0``, ``R_{t+N} = 0`` past the episode end. With
    ``window = 1`` this equals the unsmoothed RECAP formula exactly.

    Frame-to-frame value noise is much larger than the per-step reward over a
    short lookahead, so smoothing keeps the label from flipping between
    neighbouring frames while preserving the episode-level trend.

    Args:
        episode_index: Episode id per row; rows of one episode must be
            contiguous and in frame order.
        returns: Raw (unnormalised) return ``R_t`` per row.
        values: Value prediction ``V(o_t)`` per row (normalised scale).
        lookahead: N, in frames.
        window: Odd moving-average width in frames (1 disables smoothing).
        return_min: Global return minimum used for normalisation.
        return_max: Global return maximum used for normalisation.

    Returns:
        ``(advantage, value_current, value_next, reward_sum)`` arrays.
    """
    if window < 1 or window % 2 == 0:
        raise ValueError(f"window must be a positive odd integer, got {window}")
    episode_index = np.asarray(episode_index)
    returns = np.asarray(returns, dtype=np.float64)
    values = np.asarray(values, dtype=np.float64)
    span = return_max - return_min
    out = [np.empty(len(values)) for _ in range(4)]
    starts = np.flatnonzero(np.r_[True, episode_index[1:] != episode_index[:-1]])
    for start, end in zip(starts, np.r_[starts[1:], len(values)]):
        v, r = values[start:end], returns[start:end]
        if window > 1:
            half = window // 2
            v = np.convolve(
                np.pad(v, half, mode="edge"), np.ones(window) / window, "valid"
            )
        n = end - start
        nxt = np.arange(n) + lookahead
        past_end = nxt >= n
        nxt = np.minimum(nxt, n - 1)
        v_next = np.where(past_end, 0.0, v[nxt])
        r_next = np.where(past_end, 0.0, r[nxt])
        # Same normalisation as compute_advantages.normalize().
        reward_sum = (r - r_next - return_min) / span - 1.0
        out[0][start:end] = reward_sum + v_next - v
        out[1][start:end], out[2][start:end], out[3][start:end] = v, v_next, reward_sum
    return tuple(out)


def threshold_scores(scores_by_dataset, dataset_types, allowed_types=None):
    """Concatenate the scores the advantage threshold is computed on.

    Args:
        scores_by_dataset: One 1-D score array per dataset.
        dataset_types: The dataset ``type`` (e.g. ``"sft"``, ``"rollout"``) of
            each entry, in the same order.
        allowed_types: Types whose frames set the threshold; ``None`` uses all
            datasets (the original behaviour). Every dataset is still labelled
            with the resulting threshold.

    Returns:
        The concatenated scores.
    """
    if len(scores_by_dataset) != len(dataset_types):
        raise ValueError("One dataset type is required per score array.")
    if allowed_types is None:
        selected = list(scores_by_dataset)
    else:
        allowed = {str(t).lower() for t in allowed_types}
        selected = [
            scores
            for scores, kind in zip(scores_by_dataset, dataset_types)
            if str(kind).lower() in allowed
        ]
        if not selected:
            raise ValueError(f"No dataset of types {sorted(allowed)} to threshold on.")
    return np.concatenate([np.asarray(s, dtype=np.float64) for s in selected])
