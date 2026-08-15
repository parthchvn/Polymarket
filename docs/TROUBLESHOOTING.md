# Troubleshooting

Failures actually hit in operation, their causes, and fixes. Add to
this file when something new bites; the cost of writing it down is
minutes, the cost of rediscovering it is hours.

## Collector / databases

**`database is locked`** — two writers on one SQLite file. The
collector OWNS `runs/forward.sqlite` exclusively; analysis and LLM
work run on a snapshot (`scripts/overnight.sh` merges into
`runs/nightly/work.sqlite`). Never point normalize/pipeline at the
live collector database. See docs/OPERATIONS.md.

**All-zero cycle + `no such table: raw_responses`** — the snapshot
step read an empty or freshly-created database. Usually: the script
ran from the wrong directory (sqlite3.connect CREATES missing files),
or the collector was never started. The guard now refuses loudly;
if you see the refusal, start the collector (`make collect`) and
check for a phantom `scripts/runs/` tree.

**Collector crashed overnight** — historically caused by an analysis
process holding a long write transaction on the same file (LLM
normalization holds transactions for minutes). The snapshot-merge
pattern exists precisely for this; don't work around it.

## Ollama / LLM stages

**`model_unavailable` on every online screen** — expected before the
first two pipeline cycles: the online basis screens with the
PREVIOUS cycle's model. Two cycles in, screened rows appear.

**extract-claims runs forever, zero rows** — pre-guard versions
committed once per batch and processed oldest-first, so one
pathological article blocked everything invisibly. Current code
commits per article, caps degenerate outputs, records failures, and
prioritizes newest/relevant/real-bodied articles. If throughput still
looks wrong: `ollama ps` (is the model loaded and busy?), and check
`articles_failed` + `failed_examples` in the report.

**Qwen slow** — qwen3:8b at temperature 0 with structured output is
minutes per article when bodies are long; batched relevance (v2b,
default) is one call per claim across ALL markets. `--limit 5` per
loop iteration interleaves politely with the 2h rescore.

## Imports

**hf download stuck at 0%** — the Xet storage path in some hf-hub
versions. Use curl with resume:
`curl -L -C - --retry 10 -o <dest> <resolve URL>`.

**import-sii bars step takes hours** — fixed: bars are built in one
ordered pass with an index. If an old database predates the fix,
`build-execution-bars --db ...` rebuilds them in minutes; bars are
derived data.

**duckdb remote scans of quant.parquet time out** — expected; a
condition_id filter still streams the whole column over HTTP.
Download once, import locally, delete the parquet.

## Git workflow

**`git am` fails: "patch does not apply" / "already exists"** — the
patch was generated against a different main than yours, or contains
commits already merged. `git am --abort`, then `git fetch origin &&
git reset --hard origin/main` on a fresh branch and re-apply; if the
mbox has multiple commits and only the first is stale, `git am
--skip` past it. Never resolve mid-am by switching branches — abort
first.

**"There isn't anything to compare" on GitHub** — the branch on
GitHub doesn't have your local commits (push didn't run or went
before the am), or the work was already merged. Check
`git log --oneline origin/<branch> -3` vs local; push with
`--force-with-lease` if the branch history was rebuilt.

**Pushed but no PR** — pushes never auto-create PRs. Open
`github.com/parthchvn/Polymarket/compare/main...<branch>` and click
Create pull request.

## Interpreting refusals

Refused stages and gates are not errors: each refusal names the data
that flips it (more complete bars, more markets, more utc-days, more
decisions, imported annotations). The system is designed to refuse
rather than fabricate; treat a refusal as a to-do list item, not a
bug report.
