import json
import os
import zipfile
from pathlib import Path

ROOT = Path(os.environ.get("OEIS_PROBE_OUTPUT", "tmp-hybrid-probe")).resolve()
ROOT.mkdir(parents=True, exist_ok=True)
archive = Path(
    os.environ.get("OEIS_PROBE_ARCHIVE", "upload/bootstrap-step1-hybrid-control-001-weights.zip")
).resolve()
history = []
with zipfile.ZipFile(archive) as z:
    names = set(z.namelist())
    for name in sorted(names):
        if (
            not name.startswith("roles/train-")
            or "-attempt-" not in name
            or not name.endswith("/result.json")
        ):
            continue
        result = json.loads(z.read(name))
        item = {
            key: result.get(key)
            for key in ("completed_updates", "validation_loss", "validation_denominator", "device")
        }
        item["source"] = name
        diagnostic = str(Path(name).parent / "diagnostic.json")
        if diagnostic in names:
            data = json.loads(z.read(diagnostic))
            item["splits"] = {
                split: {
                    key: value.get(key)
                    for key in (
                        "exact",
                        "denominator",
                        "evaluated",
                        "coverage_complete",
                        "completed",
                        "legal",
                        "prefix_exact",
                        "continuation_exact",
                        "shuffled_exact",
                        "shuffled_denominator",
                    )
                }
                for split, value in data["splits"].items()
            }
        history.append(item)
(ROOT / "source_history.json").write_text(json.dumps(history, indent=2))
