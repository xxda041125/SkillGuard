# SkillGuard

This archive accompanies Algorithm 1. It contains the detector implementation, a configuration for the paper's training split, an index of the corpus, and a small executable example. Run the commands below from this directory using Python 3.10.

## Check the data accounting

`python verify_manifest.py` uses only the Python standard library. It verifies that each identity belongs to one partition, each variant points to that identity's clean document, and the counts match the manuscript.

| Partition | Clean identities | Matched pairs |
| --- | ---: | ---: |
| Training | 160 | 2,513 |
| Validation | 20 | 320 |
| Test | 20 | 319 |

Each pair refers to one clean document and one variant. Multiple variants can share the same clean document. The manifest contains relative paths and SHA-256 digests for 200 clean texts and 3,152 variants. For a local copy of the corpus, run `python verify_manifest.py --corpus PATH_TO_CORPUS` to check all 3,352 files byte for byte.

## Run the synthetic example

```shell
python -m pip install -r requirements.txt
python smoke.py
```

The example needs the pretrained encoder and tokenizer named `sentence-transformers/all-MiniLM-L6-v2`. Transformers loads them on first use or from its local cache. The example creates four synthetic skill identities with harmless text, trains for one epoch on CPU, fits a two-pair memory, scores a multi-window document with evidence, and checks the empty-input result. It deletes temporary files when finished. Its labels serve only to exercise the code. The smoke run was verified with Python 3.10.11, PyTorch 2.11.0, Transformers 5.8.1, NumPy 2.2.6, PyYAML 6.0.3, tqdm 4.67.3, and scikit-learn 1.7.2.

## Train and screen using the paper split

Place the indexed document tree under `corpus/`, with `clean/` and `attacked_by_generator/` directly inside it. The archive supplies the manifest. The full document texts and pretrained encoder must be obtained separately.

```shell
python verify_manifest.py --corpus corpus
python train.py
python screen.py fit --checkpoint runs/paper/ckpt_best.pt --bank runs/paper/bank.pt
python screen.py score --checkpoint runs/paper/ckpt_best.pt --bank runs/paper/bank.pt --input candidate/SKILL.md --evidence
```

Training uses one truncated sequence per document, its full-text operation graph, and four labeled documents per training unit. The neighbor is a different training identity from the same domain, with its own matched clean and variant texts. The training entry point retains the checkpoint with the highest validation F1 over five epochs.

With the checkpoint frozen, `fit` encodes the first 32 windows at most per document with a 256-token limit and a 128-token overlap. It stores the paired training vectors, training-set extrema, and a validation-selected threshold. This implementation computes extrema over all training documents against the complete training bank. For the default calibrated operating point it chooses the threshold maximizing validation F1, breaking ties at the lowest threshold. The separate 40-document, false-positive-constrained component-ablation protocol in the paper is outside this command's scope.

`score` uses the base score with weight 0.75 and the paired-reference score with weight 0.25 after normalization. It returns JSON with the calibrated score, threshold, decision (`toxic`), and optional evidence. The candidate can be any UTF-8 Markdown file. To screen with the base score, use `screen.py base` with the same checkpoint and input and supply the base mode's independently selected validation threshold via `--threshold`.

## Map from the paper to the code

| Paper component | Implementation |
| --- | --- |
| Paired training units | `src/data/dataset.py` |
| Binary, residual, and direction losses | `src/losses.py`, `experiments/training/run_experiment.py` |
| Semantic representation, attention, and window selection | `src/models/encoder.py`, `anchors.py`, `noise_attention.py`, and `detective.py` in the same directory |
| Textual operations and graph conditioning | `src/data/behavior_extractor.py`, `src/models/behavior_gnn.py` |
| Frozen memory, scoring, and audit evidence | `screen.py`, `gate.py` |

The archive supports inspection and rerunning the detector pipeline. The original detector checkpoints, victim-run logs, and external evaluation records are outside the archive. The synthetic run does not establish the paper's reported effectiveness or latency.
