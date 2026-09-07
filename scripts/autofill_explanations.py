#!/usr/bin/env python3
"""Standalone, non-interactive counterpart to app.py's "Auto-fill missing
explanations" button (render_ai_autofill_missing), meant to run on a
schedule via GitHub Actions instead of needing someone to have the
Streamlit app open in a browser tab.

Reuses app.py's actual logic (suggest_ai_explanation, apply_explanation_edit,
is_valid_for_practice, the Gemini throttling/retry/error-classification
code, the GitHub Contents API push) rather than reimplementing any of it,
by importing app.py directly with `streamlit` stubbed out. Nothing about
the quiz app's Gemini prompt, rate-limit handling, or GitHub-push
behavior is duplicated here -- if that logic changes in app.py, this
script picks the change up automatically on its next run.

Run from anywhere -- APP_PATH below is resolved relative to this script's
own location on disk, not the current working directory:

    python scripts/autofill_explanations.py

Configuration (environment variables -- see the accompanying GitHub
Actions workflow, .github/workflows/autofill_explanations.yml, for how
these map to repo secrets):

  GEMINI_API_KEY    required. Same key app.py reads via
                    st.secrets["gemini_api_key"].
  GITHUB_TOKEN      required. A token with contents:write on this repo.
                    In the Actions workflow this is the automatically
                    provided secrets.GITHUB_TOKEN -- no new secret needed.
  GITHUB_REPOSITORY required. "owner/repo". GitHub Actions sets this
                    automatically; set it yourself for a local test run,
                    e.g. GITHUB_REPOSITORY=yourname/your-quiz-repo.
  AUTOFILL_MAX_PER_RUN  optional, default 30. Same role as app.py's
                    AI_AUTOFILL_MAX_PER_RUN, just a larger default since
                    this isn't bounded by keeping a browser tab responsive
                    -- tune it down if a run is getting close to Gemini's
                    per-day quota alongside whatever the live app also
                    uses that day.
  AUTOFILL_USER     optional, default "auto (GitHub Actions)". Recorded
                    as the edit-log "user" for every explanation this
                    script saves, so the audit trail (and the app's
                    "Auto AI Explanation Saves" panel) can tell automated
                    fills apart from ones an admin filled by hand in the UI.
"""
import os
import sys
import types
import importlib.util
from pathlib import Path

APP_PATH = Path(__file__).resolve().parent.parent / "app.py"


class _DummyStreamlit(types.ModuleType):
    """Minimal stand-in for the `streamlit` module so app.py can be
    imported and its non-UI functions called directly, outside a real
    `streamlit run` session.

    Every attribute not explicitly defined below resolves to a no-op
    callable via __getattr__ -- enough for the UI calls (st.write,
    st.button, st.expander, ...) that only run inside main()/render_*(),
    none of which this script calls, plus the try/except-guarded
    st.set_page_config() that runs at app.py's module-import time.
    """

    def __init__(self):
        super().__init__("streamlit")
        self.secrets = {
            "gemini_api_key": os.environ.get("GEMINI_API_KEY", ""),
            "github_token": os.environ.get("GITHUB_TOKEN", ""),
            "github_repo": os.environ.get("GITHUB_REPOSITORY", ""),
        }

    @staticmethod
    def cache_data(f):
        # app.py decorates several loaders with a bare @st.cache_data and
        # later calls loader.clear() itself (e.g. inside
        # apply_explanation_edit, after saving). A plain function object
        # accepts an arbitrary attribute, so this satisfies that call
        # without needing a real cache.
        f.clear = lambda: None
        return f

    @staticmethod
    def cache_resource(f):
        f.clear = lambda: None
        return f

    def __getattr__(self, name):
        return lambda *args, **kwargs: None


def _load_app_module():
    sys.modules["streamlit"] = _DummyStreamlit()
    spec = importlib.util.spec_from_file_location("quiz_app", APP_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    max_per_run = int(os.environ.get("AUTOFILL_MAX_PER_RUN", "30"))
    run_user = os.environ.get("AUTOFILL_USER", "auto (GitHub Actions)")

    missing_env = [
        name for name in ("GEMINI_API_KEY", "GITHUB_TOKEN", "GITHUB_REPOSITORY")
        if not os.environ.get(name)
    ]
    if missing_env:
        print(f"Missing required environment variable(s): {', '.join(missing_env)}. "
              "Set them as repo secrets -- see this script's docstring. Exiting.")
        sys.exit(1)

    app = _load_app_module()

    questions, _conflicts, _skipped, source_file_by_id = app.load_questions()
    missing = [
        q for q in questions
        if app.is_valid_for_practice(q) and not (q.get("explanation") or "").strip()
    ]
    if not missing:
        print("No questions are missing an explanation. Nothing to do.")
        return

    batch = missing[:max_per_run]
    print(f"{len(missing)} question(s) missing an explanation; filling up to "
          f"{len(batch)} this run ({app._MIN_SECONDS_BETWEEN_GEMINI_CALLS:.1f}s "
          "between Gemini calls).")

    filled = 0
    failures = []
    stopped_early = False
    for i, q in enumerate(batch):
        result = app.suggest_ai_explanation(q, "generate")
        if "error" in result:
            failures.append((q["question_id"], result["error"]))
            print(f"  [{i + 1}/{len(batch)}] {q['question_id']}: FAILED -- {result['error']}")
            if result.get("stop_batch"):
                stopped_early = True
                break
            continue

        save_result = app.apply_explanation_edit(
            q["question_id"], result["revised_explanation"], run_user, source_file_by_id,
            ai_model=result["model_version"], issues_found=result["issues_found"],
        )
        filled += 1
        if save_result["github_ok"] and save_result["github_log_ok"]:
            sync_note = "synced"
        else:
            data_part = "ok" if save_result["github_ok"] else f"FAILED: {save_result['github_msg']}"
            log_part = "ok" if save_result["github_log_ok"] else f"FAILED: {save_result['github_log_msg']}"
            sync_note = f"data sync {data_part}, log sync {log_part}"
        print(f"  [{i + 1}/{len(batch)}] {q['question_id']}: filled ({sync_note})")

    print(f"\nDone. Filled {filled} of {len(batch)} attempted this run "
          f"({len(missing) - filled} question(s) still missing overall).")
    if stopped_early:
        print(f"Stopped early: {failures[-1][1]}")
    elif failures:
        print(f"{len(failures)} question(s) failed this run and will be retried on the next scheduled run.")


if __name__ == "__main__":
    main()
