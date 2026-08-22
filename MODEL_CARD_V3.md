# ChessNet v3 — Model Card

Generated for the native/GPU training restart. Records the selectable model
sizes and their identity so a checkpoint can never be silently loaded into a
differently-shaped body.

## Architecture

ResNet encoder (conv-in + BN + ReLU, then `num_res_blocks` residual blocks of
two 3×3 convs each) with:

- a **spatial policy head** — 1×1 conv over the board grid, one channel per
  action plane (73), flattened in NHWC order so flat logit index equals
  `from_square * 73 + plane` (the AlphaZero 4,672-action map).
- a **value head** — 1×1 conv → 32 channels, ReLU, flatten, linear → filters,
  ReLU, linear → 1, tanh (output in [−1, 1]).

Input: 104 planes (8-position history × 12 piece planes + 8 meta planes), White
orientation, `rank*8+file` order.

## Registered sizes

| architecture_id | res blocks | filters | parameters | FP32 weights |
|---|---:|---:|---:|---:|
| `v2-6x128` | 6 | 128 | 2,170,218 | 8.68 MiB |
| `v3-10x128` | 10 | 128 | 3,352,938 | 13.41 MiB |
| `v3-10x192` | 10 | 192 | 7,241,194 | 28.96 MiB |
| `v3-10x256` | 10 | 256 | 12,604,010 | 50.42 MiB |

Parameter counts are verified by `tests/test_model_v3.py` against the plan.

## Identity & versioning

- `Config.architecture_id` names the body; `resolve_architecture_id(res, filters)`
  maps a shape to its registered id (or `custom-<res>x<filters>` for an
  unregistered shape).
- `infer_state_dict_architecture_id(state_dict)` recovers the id from a
  checkpoint's weight shapes.
- The schema-v3 loader **validates** `architecture_id` before `load_state_dict`
  and rejects any cross-body or unknown-schema load (`IncompatibleCheckpointError`).

## Selection (Task 9)

Default remains **`v2-6x128`** until the native/GPU pipeline is benchmarked
end-to-end. The preferred first larger model is **`v3-10x192`** (7.24M params,
>3× the v2 model's useful capacity) if it stays under the VRAM/throughput gates
on the RTX 2080 Ti; `v3-10x256` is only for when throughput remains acceptable.

## Training (Task 8)

- Mixed precision via `torch.amp.GradScaler` + autocast on CUDA; FP32 master
  weights; scaler state is checkpointed and restored on resume.
- Replay stays in RAM; a pinned prefetch loader stages minibatches for async H2D.
