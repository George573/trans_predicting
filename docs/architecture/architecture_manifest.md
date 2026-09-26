# Boarding-only forecasting architecture

The only observation stream is bounded hourly boarding history. Calendar features,
route identity and requested forecast lead remain model inputs. Raw transaction
lists, event embeddings, event convolutions, chunking and the second hourly branch
have been removed.

## Default computation graph

All shapes below omit the batch dimension. Convolution paths are independent
Conv1d + GELU branches concatenated over channels, with same padding.
A path tuple is `(output channels, kernel length, dilation)`.

| Stage | Paths | Output shape |
|---|---|---|
| Input | Scaled counts + six calendar features | 7 × 504 |
| Boarding hourly CNN | (12,3,1), (12,5,24), (8,3,168) | 32 × 504 |
| Shared 1 + max pool 2 | (24,3,1), (24,5,2), (16,3,12) | 64 × 252 |
| Shared 2 + max pool 2 | (12,3,1), (12,5,2), (8,3,12) | 32 × 126 |
| Shared 3 + max pool 3 | (6,3,1), (6,5,2), (4,3,6) | 16 × 42 |
| Flatten | | 672 |
| Request conditioning | Route embedding 8 + target calendar 4 + lead 1 | 685 |
| Head | Linear 250, GELU, dropout 0.1, Linear 24 | 24 |
| Output | Softplus × count scale | 24 nonnegative hourly counts |

Default parameter count: **195,868**. The boarding branch, shared stages and head
retain the same geometry and state-dict keys as the previous boarding-only model.
History length, hourly stride, shared depth, max/average temporal pooling, adaptive
pool size, convolution paths and head width remain configurable.

Historical calendar channels are sine/cosine of hour/24, weekday/7 and
(day-of-month−1)/31. Target calendar omits hour. Lead is `(lead−1)/60`, with
integer leads 1–61. Count scale is `max(1, mean fitting-period counts)`.

Contexts are route/cutoff pairs. A context is encoded once and reused for all
eligible requested days. Future labels never enter history. Route 5 retains its
external zero fallback; the neural routes are 1, 7, 11, 12, 17, 25, 26, 28 and 50.

See [implementation contracts](architecture_implementation_plan.md) for storage,
training, checkpoints and migration.
