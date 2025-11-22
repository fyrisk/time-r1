# Omni audio-video temporal evaluation

This helper mirrors the flow described earlier (data loading, model abstraction, metrics) but is tailored to the Omni Reason Ground QA layout.

## Directory layout
- **QA root**: `/data5/fy/omni-reason-ground/qa`
- **Video root**: `/data5/fy/data/mybenchvideo`

For each QA JSON (e.g., `qa/worldsense/AAWgrzYx.json`), the evaluator looks for the matching video at `mybenchvideo/worldsense/AAWgrzYx.mp4` and iterates through every QA entry inside the JSON file. Each item needs a `question` and `time` pair (`["MM:SS", "MM:SS"]`).

## Running the evaluator
```bash
python scripts/run_audio_video_eval.py \
  --qa-root /data5/fy/omni-reason-ground/qa \
  --video-root /data5/fy/data/mybenchvideo \
  --output outputs/omni_eval.jsonl \
  --model-module /path/to/my_model_impl.py
```

The script writes per-sample predictions to `outputs/omni_eval.jsonl` and a metric summary to `outputs/omni_eval.summary.json`.

### Concurrency
`--concurrent` (default: off) enables a thread pool for issuing requests in parallel—useful when calling a remote API. Use `--max-workers` to cap the worker count. Local models can keep the default sequential loop.

### Resume / checkpointing
Results stream to the JSONL file one sample at a time. If you rerun the script, any QA already present in the JSONL (matched by `qa filename + question`) will be skipped, allowing for safe interruption and restart.

## Plugging in your own model
Provide a Python file that defines `build_model() -> TemporalLocalizationModel`. The model must implement `predict_timestamps(video_path: str, question: str, qa_sample: dict | None) -> tuple[str | float, str | float]`, returning start and end timestamps either as seconds or `MM:SS` strings. Example stub:

```python
from omni_eval.model_interface import TemporalLocalizationModel

class MyApiModel(TemporalLocalizationModel):
    def predict_timestamps(self, video_path: str, question: str, qa_sample=None):
        payload = {"video": video_path, "question": question}
        response = call_my_backend(payload)
        return response["start"], response["end"]

def build_model():
    return MyApiModel()
```

The default fallback model (`EchoGroundTruthModel`) simply echoes the ground-truth `time` field for quick smoke tests.

## Sampling a smaller QA subset
Use `scripts/sample_qa_subset.py` to create a filtered copy of the QA tree while preserving subdirectories and filenames. For example, to randomly pull three items from each category prefix `1.1`, `1.2`, and `2.1`:

```bash
python scripts/sample_qa_subset.py \
  --qa-root /data5/fy/omni-reason-ground/qa \
  --output-root /data5/fy/omni-reason-ground/small_qa \
  --target-codes 1.1 1.2 2.1 \
  --k 3
```

Only the selected entries are written to the mirrored location under `--output-root` (e.g., `qa/daily/_-BAFzpKigw_video.json` → `small_qa/daily/_-BAFzpKigw_video.json`). Use `--seed` to make the sampling deterministic.
