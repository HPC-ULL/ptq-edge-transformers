## `KLMinimizerThreshold` crashes with `IndexError` for every even `num_bins`

**Version:** brevitas 0.13.0 (confirmed against `src/brevitas/core/stats/stats_op.py` on the `v0.13.0` tag; the affected loop is unchanged on `master` as of this writing).

### Summary

`KLMinimizerThreshold.forward()` raises `IndexError: index N out of bounds for dimension 0 with size N` for every **even** `num_bins`, and only succeeds for **odd** `num_bins`. The default (`num_bins=1001`) happens to work only because it is odd, not because it was validated against this code path.

### Root cause

```python
hist_edges = torch.linspace(-absmax, absmax, self.num_bins + 1)  # size = num_bins + 1, valid indices 0..num_bins
for i in range(num_quantized_bins // 2, self.num_bins // 2 + 1):
    p_bin_idx_stop = self.num_bins // 2 + i + 1
    thresholds[i - num_quantized_bins // 2] = hist_edges[p_bin_idx_stop]
```

At the loop's last iteration, `i == self.num_bins // 2`, so:

```
p_bin_idx_stop = (num_bins // 2) + (num_bins // 2) + 1
```

- **`num_bins` even:** `num_bins // 2 + num_bins // 2 == num_bins` exactly, so `p_bin_idx_stop == num_bins + 1` — one past the last valid index of `hist_edges` (which has `num_bins + 1` elements, indices `0..num_bins`) → `IndexError`.
- **`num_bins` odd:** floor division truncates, `num_bins // 2 + num_bins // 2 == num_bins - 1`, so `p_bin_idx_stop == num_bins` — the last valid index → no crash.

### Reproduction

Tested on a real captured activation tensor (shape `(128, 197, 192)`, `absmax≈10.44`, from a ViT attention block's QKV projection input), `bit_width=8` (`num_quantized_bins=127`), `smoothing_eps=0.0001`:

| `num_bins` | parity | result |
|---|---|---|
| 256, 384, 512, 750, 1500, 2048, 4096 | even | `IndexError: index (num_bins+1) out of bounds for dimension 0 with size (num_bins+1)` |
| 1001 | odd | succeeds |

### Suggested fix

Clamp `p_bin_idx_stop` to `self.num_bins` (`p_bin_idx_stop = min(self.num_bins // 2 + i + 1, self.num_bins)`), or change the loop's upper bound so the last iteration never produces an out-of-range index regardless of `num_bins`'s parity.

### Impact

Any user passing an even `num_bins` to `KLMinimizerThreshold` (directly, or via a composed quantizer such as the `entropy`/KL-divergence calibration preset) hits this immediately. Since it's parity-dependent rather than value-dependent, it's easy to trip accidentally (e.g. any power-of-two `num_bins`, a common default elsewhere in ML tooling) and easy to miss if only the library's own odd default has ever been exercised.
