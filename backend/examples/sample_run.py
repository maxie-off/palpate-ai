"""Print citations from the synthetic lecture. See ``app.agent.demo``."""

import tempfile
from pathlib import Path

from app.agent.demo import run_sample, write_silent_wav

FIXTURE = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "sample_lecture.pdf"


def main() -> None:
    if not FIXTURE.is_file():
        raise SystemExit(f"Missing fixture: {FIXTURE}")
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        audio = root / "lecture.wav"
        write_silent_wav(audio)
        results = run_sample(
            FIXTURE,
            question="What are milia?",
            audio_path=audio,
            figures_dir=root / "figures",
        )
    for name, result in results.items():
        print(f"== {name} status={result.status}")
        print(f"   trace: {' -> '.join(result.trace)}")
        if result.summary:
            print(f"   summary: {result.summary}")
        for citation in result.citations:
            print(f"   [{citation.modality}] {citation.quote}")
            for anchor in citation.anchors:
                print(f"      {anchor.model_dump()}")


if __name__ == "__main__":
    main()
