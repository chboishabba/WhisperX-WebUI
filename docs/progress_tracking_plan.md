# Unified Progress Tracking Plan

## Goal
Implement accurate, end-to-end progress tracking across VAD → ASR → alignment → diarization with batching support and a single 0–100% signal usable by any backend and UI ETA displays.

## Approach
- Track work at the segment level (`completed segments / total segments`) with stage-specific weights.
- Maintain a job-scoped `ProgressTracker` storing total and completed units.
- Pre-scan VAD output to determine segment count, then allocate weighted units per stage.
- Wrap ASR, alignment, and diarization loops (batch-aware) to increment completed units.
- Emit progress events with stage labels and ETA for the WebUI.

## Implementation Steps
1. **ProgressTracker**
   - Create a lightweight class with `add_units`, `complete_units`, and `progress` properties for a single transcription job.

2. **Estimate Workload**
   - After VAD segmentation, count speech segments (`N`).
   - Add weighted units: `N` for ASR, `N` for alignment, and `N` for diarization when enabled.

3. **ASR Integration**
   - **faster-whisper:** hook after each processed batch, incrementing by batch size.
   - **openai/whisper:** increment once per processed segment.

4. **Alignment Integration**
   - Align per segment (or per batch) and increment after each unit to expose progress for this previously silent phase.

5. **Diarization Integration**
   - Increment per processed segment or per diarization window when diarization is enabled.

6. **Emit Progress Updates**
   - Provide a callback/state update or websocket event with fields: `progress` (0–1), `completed`, `total`, and `stage` (`"vad"`, `"asr"`, `"alignment"`, `"diarization"`).

7. **ETA Calculation**
   - Track start time and compute `eta = elapsed * (1 / progress - 1)` when progress > 0.

8. **UI Integration**
   - Drive a single progress bar with optional stage label and ETA. Keep per-stage bars optional/advanced only.

## Minimal MVP
- Use `total_units = N × 2` (ASR + alignment), increment once per segment in each stage, and emit progress after each increment.

## Design Principles
- Track segment-level work units; avoid token/frame counts or parsing tqdm output.
- Keep logic backend-agnostic and batching-friendly with stable progress behavior on long audio files.
