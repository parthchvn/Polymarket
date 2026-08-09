# Annotation protocol

How human-comparable labels are produced for reasoning evaluation
(gate 2), and how they must be described. The protocol exists so the
evaluation set stays independent of the machinery it evaluates, and
so no number is ever presented as something it is not.

## Batches

`export-annotation-batch` samples decisions deterministically
(seeded, stratified) and renders each as a STRICT pre-decision
record: only information available before the decision, explicit
missingness included. Records predating collection coverage arrive
visibly evidence-starved — that is honest, and `INSUFFICIENT_EVIDENCE`
is the correct label for them, not a failure of the annotator.

## Reviewers and naming

Every imported label file is attributed to a named reviewer. The
names are honest identities, not roles:

- human reviewers import under their own names (`ricky`, `sarth`);
- LLM-produced labels import as an explicitly non-human reviewer
  (`claude-llm`), never under a human name.

A model labeling as a human would manufacture fake inter-rater
agreement that flows into gold labels and gate 2. This is the one
rule in the protocol with no exceptions.

## LLM-drafted, human-reviewed

The efficient path, and the one actually used:

1. an LLM annotator labels the full batch (`label`, `confidence`,
   per-record reasoning in `notes`), imported as `claude-llm`;
2. each human reviewer independently reads the drafted file and
   changes ONLY what they disagree with — no coordination between
   reviewers; the disagreements are the data;
3. human files import under human names; Cohen's kappa is computed
   between human reviewers; gold labels form only where reviewers
   are unanimous.

Reviewing drafts is roughly an order of magnitude faster than
labeling from scratch and leaves the human gold standard intact,
because agreement statistics are computed between humans.

## Disclosure language

Wherever these labels are cited (reports, decks, papers), the
protocol is described as:

> labels were LLM-drafted and human-reviewed

and never as blind independent human annotation. Model-vs-annotator
agreement involving `claude-llm` may be cited as an LLM-annotator
diagnostic; it is never called human agreement.

## Cadence

Batches are re-exported as coverage grows: early batches measure
abstention calibration (does the model refuse where an annotator
must?); later batches, drawn from coverage-rich periods, evaluate
the substantive templates.
