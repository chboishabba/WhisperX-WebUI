Repository Guidelines
Project Structure & Module Organization

- The Gradio web UI boots from `app.py` at the repository root and sources feature logic from `modules/`, where subpackages cover transcription (`modules/whisper/`), diarization & VAD (`modules/diarize/`, `modules/audio/`), and shared utilities (`modules/utils/`).
- Backend APIs and task orchestration live in `backend/` with FastAPI routers under `backend/api/`, persistence helpers inside `backend/db/`, and fixtures + integration tests in `backend/tests/`.
- Configuration presets, model bundles, and example notebooks are stored in `configs/`, `models/`, and `notebook/` respectively. Shell helpers (`start-webui.sh`, `Install.sh`, scripts in `scripts/`) sit at the top level.
- UI- and pipeline-focused tests reside in `tests/`, mirroring the package they validate (for example, `tests/test_diarization.py` exercises `modules/diarize/`). Generated assets such as transcripts or diarization CSVs belong under `outputs/` and should not be committed.

Build, Test, and Development Commands

1. Create an isolated environment with `python -m venv venv && source venv/bin/activate` (Windows: `python -m venv venv && venv\Scripts\activate`).
2. Install web UI dependencies via `pip install -r requirements.txt` and add backend extras with `pip install -r backend/requirements-backend.txt` when modifying the FastAPI service. Useful ad-hoc extras include `pip install python-dotenv sqlmodel jiwer openai-whisper` for the backend test suite.
3. Launch the web UI with `bash start-webui.sh` and run the API locally using `uvicorn backend.main:app --reload`.
4. Execute the automated tests using `pytest -q` from the repository root; scope to a single module with commands such as `pytest tests/test_transcription.py::test_pipeline`. When backend services require database state, seed via `backend/scripts/` helpers before running tests.
5. Format code with `black .` and sort imports using `isort .`; enforce lint fixes prior to committing with `ruff check --fix .` if the linter is configured for the module you are touching.

Coding Style & Naming Conventions

- Target Python 3.10+ semantics, keeping modules import-safe for synchronous execution. Use 4-space indentation, snake_case for variables/functions, and PascalCase for classes and Pydantic models.
- Prefer dependency injection (passing models, devices, or configuration objects) over module-level singletons. Constants should be UPPER_SNAKE_CASE.
- Keep Gradio component construction declarative and colocate callback helpers near the UI definition in `app.py`. For reusable utilities, expose a concise `__all__` when the module is meant to be imported elsewhere.
- Rely on Black for formatting to avoid churn; do not mix tabs and spaces. Add type hints to new functions and favour `dataclasses` or Pydantic models for structured data.

Testing Guidelines

- Mirror the source tree when creating new tests (`tests/test_<feature>.py` for UI modules, `backend/tests/test_<feature>.py` for API features). Use descriptive test names like `test_handles_segment_overlap` and leverage pytest fixtures in `tests/fixtures/` (or create new ones nearby) to share setup.
- Mock outbound network calls (Hugging Face, YouTube, DeepL) using `pytest-mock` or `unittest.mock` so the suite remains offline. Capture expected regressions with snapshot files under `tests/templates/` when behaviour changes.
- Run `pytest -q` before opening a PR. If tests rely on optional GPU/accelerator libraries, mark them with `pytest.importorskip` and document the expectation in the test docstring.

Commit & Pull Request Guidelines

- Write imperative, present-tense commit subjects ("Patch torchaudio shim for diarization"). Group related edits into a single commit and explain non-obvious changes in the body.
- Verify CI-critical checks locally (`pytest -q`, `black --check .`, `isort --check-only .`, and any configured `ruff` targets). Call out intentional skips or environment constraints in the PR description.
- Always run the relevant tests before submitting your work and report the results.
- Do not wait more than two minutes for dependencies to install while testing - they should already be installed. Just report this to the user as the requirements are almost 10gb. I don't want to waste energy needlessly. I can run the tests locally.
- PR summaries should highlight behavioural changes, mention new configuration flags, and include before/after screenshots when modifying visible UI components. Link related issues or discussions whenever available.
Core Gradio orchestration lives at the repository root with the primary entry point in app.py and implementation details split under modules/. Audio pre/post-processing helpers (VAD, UVR, diarization, transcription) sit in modules/whisper/ and modules/audio/, while reusable launch scripts live in scripts/ and shell wrappers (start-webui.sh, Install.sh) in the top-level directory. Backend API code, including FastAPI routers and database helpers, resides under backend/ with its own configs/ and tests/. Configuration presets and language packs live in configs/, and automated notebooks are kept in notebook/. Keep end-to-end functional tests under tests/ mirroring the modules they exercise, and store generated artifacts (subtitles, diarization tracks) in outputs/.
Build, Test, and Development Commands

Create an isolated environment (python -m venv venv && source venv/bin/activate) and install dependencies with pip install -r requirements.txt for the web UI or pip install -r backend/requirements-backend.txt for the API service. Run the UI locally using bash start-webui.sh and launch the backend with uvicorn backend.main:app --reload. Execute the automated suite via pytest -q from the repository root; focus on a specific feature with pytest tests/test_transcription.py::test_basic_transcription. Use npm install && npm run build only when working inside frontend assets bundled in modules/ui/ (rare).
Coding Style & Naming Conventions

Target Python 3.10–3.12 semantics, using type hints and dataclasses where appropriate while preserving compatibility with synchronous Gradio callbacks. Follow PEP 8 with 4-space indentation, snake_case for functions/variables, and PascalCase for classes. Keep configuration keys uppercase with underscores and prefer dependency injection (passing model/device handles) to module-level globals. Format code with black and ensure imports stay sorted with isort; avoid committing files that are not auto-formatted. Expose intended public helpers via __all__ when a module is meant for reuse.
Testing Guidelines

Place new regression or smoke tests inside tests/ mirroring the structure of the module under test (e.g., tests/test_diarization.py for modules/whisper/diarize pipelines). Name files test_<feature>.py and use descriptive function names like test_handles_empty_segments. Mock external services (Hugging Face, DeepL, YouTube) using pytest-mock or monkeypatch to keep the suite offline. Run pytest -q and capture any expected skips with pytest.mark.skipif. Update fixtures in tests/test_srt.srt and tests/test_vtt.vtt only when behavior changes are intentional.
Commit & Pull Request Guidelines

Adopt imperative, present-tense commit subjects ("Pin torchaudio to compatible release"). Keep each commit focused and explain non-obvious changes in the body. Before opening a PR, ensure pytest -q passes along with formatting checks (black --check ., isort --check-only .) and document any intentional skips. PR descriptions should summarize behavior changes, mention any new configuration flags, and include before/after screenshots when UI widgets are affected. Reference related issues or discussions when available.
