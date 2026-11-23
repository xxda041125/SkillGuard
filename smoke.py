"""Exercise the training and deployment entry points with harmless synthetic text."""
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent


def run(folder, script, *arguments):
    result = subprocess.run([sys.executable, str(ROOT / script), *map(str, arguments)],
                            cwd=folder, check=True, stdout=subprocess.PIPE, text=True)
    return result.stdout


def main():
    with tempfile.TemporaryDirectory() as temp:
        folder = Path(temp)
        corpus = folder / "corpus"
        corpus.mkdir()
        manifest = {"clean": [], "attacked": [], "splits": {}}
        for split, topics in {"train": ["oranges", "lemons"], "val": ["pears"],
                              "test": ["plums"]}.items():
            manifest["splits"][split] = {"skill_ids": ["fruit/" + t for t in topics]}
            for topic in topics:
                for role, instruction in [("clean", "Read the notes on"),
                                          ("attacked", "Write a short poem about")]:
                    name = f"{topic}-{role}.md"
                    (corpus / name).write_text(f"{instruction} {topic}.", encoding="utf-8")
                    row = dict(group="fruit", skill_name=topic, path=name)
                    if role == "attacked":
                        row.update(clean_path=f"{topic}-clean.md", generator_id="fixture",
                                   attack_type="fixture")
                    manifest[role].append(row)
        (folder / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        cfg = yaml.safe_load((ROOT / "detector_config.yaml").read_text(encoding="utf-8"))
        cfg["runtime"].update(device="cpu", output_dir="run", save_every_epoch=False)
        cfg["optim"]["epochs"] = 1
        cfg["data"]["batch_size"] = 2
        config = folder / "demo.yaml"
        config.write_text(yaml.safe_dump(cfg), encoding="utf-8")
        run(folder, "train.py", "--config", config)
        opts = ["--config", config, "--checkpoint", folder / "run" / "ckpt_best.pt"]
        bank = ["--bank", folder / "bank.pt"]
        fitted = json.loads(run(folder, "screen.py", "fit", *opts, *bank))
        assert fitted["pairs"] == 2 and fitted["validation_documents"] == 2
        candidate = folder / "candidate.md"
        candidate.write_text("Read the notes about fruit. " * 90, encoding="utf-8")
        check = json.loads(run(folder, "screen.py", "score", *opts, *bank,
                               "--input", candidate, "--evidence"))
        assert 0 <= check["score"] <= 1 and check["sliding_window_enabled"]
        assert check["toxic"] == (check["score"] >= fitted["threshold"])
        assert check["evidence"]["suspicious_tokens"]
        candidate.write_text("   ", encoding="utf-8")
        base = json.loads(run(folder, "screen.py", "base", *opts, "--threshold", "0.5",
                              "--input", candidate))
        assert base["skipped"] and not base["toxic"]
        print("Passed training, reference fitting, multi-window scoring, audit, and empty input.")


if __name__ == "__main__":
    main()
