# Plan for xptycho

Goal: a PyTorch package for PMACE ptychographic reconstruction that is
easy to use, streams data that does not fit in memory, and runs on one
GPU or the GPUs of one node.  It replaces the research code
ptycho_pmace.

The full design, the milestones, and the record of decisions live in
~/claude-notes/xptycho/ on Charlie's Mac (conops.md, api-design.md,
scale-design.md, data-and-verification.md, critiques.md, decisions.md,
status.md).  This file is the short version.

## 1. Repository skeleton — DONE (2026-09-26)

Packaging, documentation, tests, and developer scripts from the xcal
design.

## 2. Preserve the old code

Tag ptycho_pmace v0.0.3 and ptycho_pmace_papers v1.0.0-tci2023 with
release notes saying xptycho owes them nothing.

## 3. Numerical baseline

Capture reference runs of the old code (a tiny problem with every
iterate, the paper's synthetic case, the gold balls, the blind
two-mode case) before any porting.

## 4. Design the API by writing demo 1

The demo script is the design.  Iterate on it with Charlie.

## 5. Interface skeleton

Real docstrings, stub bodies, rendered documentation reviewed as a new
user.

## 6. Implement and verify

Data layer; operators and simulator; PMACE with a known probe matching
the baseline; stop and take stock; batching and streaming; blind and
multi-mode; measured data; multi-GPU on one node; release v0.1.0.
