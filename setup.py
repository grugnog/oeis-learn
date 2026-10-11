"""Package build provenance; no runtime dependencies are imported by the hook."""

from pathlib import Path
import json
import runpy
from setuptools import setup
from setuptools.command.build_py import build_py


class QualificationBuild(build_py):
    def run(self):
        super().run()
        root = Path(__file__).resolve().parent
        collect = runpy.run_path(str(root / "src/oeis_learn/experiments/source_snapshot.py"))[
            "collect_snapshot"
        ]
        path = Path(self.build_lib) / "oeis_learn/experiments/qualification-sources.json"
        path.write_text(json.dumps(collect(root), sort_keys=True, separators=(",", ":")))
        import shutil
        governance = Path(self.build_lib) / "oeis_learn/evaluation/governance"
        governance.mkdir(parents=True, exist_ok=True)
        for name, source in (("constitution.md", root / ".specify/memory/constitution.md"),
                             ("adoption.md", root / "specs/007-experiment-foundation/validation/adoption.md"),
                             ("review-inputs.json", root / "specs/007-experiment-foundation/validation/review-inputs.json"),
                             ("tasks.md", root / "specs/007-experiment-foundation/tasks.md")):
            shutil.copyfile(source, governance / name)
        schemas = Path(self.build_lib) / "oeis_learn/cli/contracts"
        schemas.mkdir(parents=True, exist_ok=True)
        for source in (root / "specs/005-trustworthy-synthesis-readiness/contracts").glob("*.schema.json"):
            shutil.copyfile(source, schemas / source.name)


setup(cmdclass={"build_py": QualificationBuild})
