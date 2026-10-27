# Five-host Deepdish comparison

Inspected through the user's SSH aliases on **2026-10-26**, around 22:02--22:10 UTC
(17:02--17:10 America/Chicago). All five SSH connections succeeded. Raw initial
and follow-up JSON inventories, including active GPU process lists, are in
`results/policy_gradient_part1/remote/`. Loads and GPU availability are snapshots.

| Host | GPU inventory and reported VRAM per card | Observed GPU use (MiB used; compute utilization) | CPU / RAM available | Python and PyTorch | Suitability |
|---|---|---|---|---|---|
| deepdish1 | 4 x GTX 1080, 8192 MiB, CC 6.1 | GPU 0: 8071; 100%. GPUs 1/2/3: 3; 0% | 12 logical cores, i7-6850K; about 119.7 GiB; load 1.31 | System Python 3.10.12 without torch. Shared Python 3.9.12, torch 2.4.1+cu124, sm_60 included | Good low-CPU-load fallback with a Pascal-compatible isolated build; no kernels tested |
| deepdish2 | GTX 1080, 8192 MiB, CC 6.1; IDs 0/1/3 healthy, ID 2 device-handle error | Healthy GPUs: 3; 0%; no reported compute process | 12 logical cores, i7-6850K; about 118.9 GiB; load 9.00 | Python 3.10.12; shared 3.9.12; no torch found | Lower priority: hardware error on GPU 2, higher CPU load; do not use/reset the bad card |
| deepdish3 | 8 x RTX 2080 Ti, 11264 MiB, CC 7.5 | GPU 0: 10810; 40--52%. GPU 2: 8676; 35--44%. IDs 1/3/4/5/6/7: 1; 0% | 48 logical cores, Xeon Silver 4214; about 101.9 GiB; load 64.95 | Python 3.10.12; isolated torch 2.8.0+cu128 being prepared | **Selected: GPU 4**, ample VRAM, mature Turing support and tmux; high CPU load must be measured |
| deepdish4 | 1 x RTX A6000, 49140 MiB, CC 8.6 | GPU 0: 417; 0%, but an active process owns 408 MiB | 32 logical cores, Xeon Silver 4208; about 171.6 GiB; load 2.36 | Python 3.10.12 torch 2.6.0+cu124; shared Python 3.9 torch 2.8.0+cu128; sm_86 present | Excluded while occupied, despite low utilization and excellent hardware |
| deepdish5 | IDs 0/2: RTX 5060 Ti, 16311 MiB; ID 1: RTX PRO 5000 Blackwell, 48935 MiB; CC 12.0 | All 0 MiB; 0%; no reported GPU processes | 48 logical cores, Threadripper 7960X; about 74.3 GiB; load 40.45 | Python 3.13.5, no inspected torch; needs an sm_120-capable build | Strong conditional option, but no tmux/screen, unvalidated Python/Atari/GPU stack, and substantial CPU load |

Drivers: hosts 1--4 report **580.178.04**, CUDA compatibility **13.0**. Host 5
reports **615.71.09**, CUDA UMD **13.4**. Installed `nvcc` is 11.5 on hosts 1/2
and absent from PATH on 3/4/5. A prebuilt wheel uses its bundled CUDA runtime;
the absence of nvcc does not by itself block PyTorch inference/training.

All hosts had no discovered Slurm/PBS commands on PATH. Hosts 1--4 have tmux and
screen; host 5 has neither on PATH or the inspected conventional tmux paths.
The user explicitly confirmed that direct single-GPU use is permitted for this
coursework. No other user's process was stopped or altered.

## Decision and ranking

1. **deepdish3 GPU 4**, UUID `GPU-4657a7f9-c919-cead-c311-8663723dbd61`:
   idle, 10820 MiB reported free, CC 7.5, tmux available, and modern wheel
   architecture support. Use `CUDA_VISIBLE_DEVICES` set to that UUID and address
   it only as `cuda:0`. Real forward/backward kernels and the complete-game,
   checkpoint and separate-evaluation GPU smoke passed with torch 2.8.0+cu128
   and sm_75 support. Full games took 4.1--7.4 seconds; peak tensor allocation
   was about 32.5 MiB. Host load and serial Atari stepping limit throughput.
2. **deepdish1 GPU 3** (or another freshly checked idle card): lower CPU load,
   enough memory for an 81,146-parameter CNN, and mature persistent sessions.
   Pascal needs a deliberately compatible wheel, rather than assuming every
   new CUDA build supports it. It remains a fallback, not a second concurrent job.
3. **deepdish5 GPU 1**, conditional on a validated Blackwell build and an approved
   user-level persistent-session setup. The inspected torch 2.8.0+cu128 build
   on host 4 lists sm_120, but no GPU kernel was run on host 5. It is therefore
   not recommended as a verified ready-to-run choice. Large VRAM does little for
   this small serial Atari pipeline without demonstrated end-to-end throughput.
4. **deepdish2 healthy cards**, with explicit avoidance of GPU 2. The device error
   lowers confidence in unattended operation; no system repair/reset is attempted.
5. **deepdish4 GPU 0 excluded** while another process occupies it. Zero sampled
   compute utilization is not the same as an unoccupied GPU.

The selected device is one candidate, not an allocation against other users.
Recheck the device's process list immediately before launching. The remote
wrapper holds a per-account lock to prevent overlapping Pong GPU jobs and
enforces exactly one visible GPU; it does not reserve an idle GPU for this user.
Measure actual steps/second, seconds/game, peak allocated VRAM and entropy in
the smoke/pilots before choosing a formal budget. Switching hosts, if justified,
must happen sequentially after the prior GPU process exits.
