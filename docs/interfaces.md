# WhisperX-WebUI Interface Contract (Intended)

## Intersections
- Provides transcription/diarization services to `tircorder-JOBBIE/`.
- Feeds transcript artifacts into `SensibLaw/` and ITIR evidence pipelines.
- Shares runtime outputs with suite-level processing and review tools.

## Interaction Model
1. Accept audio/media inputs from UI, API, or file paths.
2. Execute selected transcription/translation/diarization pipeline.
3. Persist subtitle/transcript artifacts with timestamps.
4. Return structured outputs and run diagnostics.

## Exchange Channels
### Channel A: Media Ingress
- Input: files, streams, URLs, microphone/system audio.
- Output: normalized processing job definitions.

### Channel B: Pipeline Control Ingress
- Input: model/runtime configuration, language, diarization, translation options.
- Output: run configuration snapshot for reproducibility.

### Channel C: Transcript Artifact Egress
- Output: text, SRT/WebVTT/TXT, diarization labels, alignment metadata.
- Consumer: `tircorder-JOBBIE/` and ingest adapters.

### Channel D: Service/API Egress
- Output: endpoint responses, status payloads, and error diagnostics.
- Consumer: local automation and orchestration scripts.
