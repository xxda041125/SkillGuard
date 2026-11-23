"""Paper-facing training entry point for Counterfactual SkillGuard.

Default behavior trains the compact residual detector configured by
detector_config.yaml:

    L = w_cls * L_det + w_residual * L_res + w_direction * L_dir

"""
from __future__ import annotations

import sys
from pathlib import Path


def main() -> None:
    here = Path(__file__).resolve().parent
    experiment_dir = here / "experiments" / "training"
    if str(experiment_dir) not in sys.path:
        sys.path.insert(0, str(experiment_dir))

    if "--config" not in sys.argv:
        sys.argv.extend(["--config", str(here / "detector_config.yaml")])
    else:
        pos = sys.argv.index("--config") + 1
        sys.argv[pos] = str(Path(sys.argv[pos]).resolve())

    from run_experiment import main as run_training

    run_training()


if __name__ == "__main__":
    main()
