# Hand-written WER smoke fixture

Ten short booking utterances written by hand (`references.txt`) and a hypothesis
file with deliberate ASR-style errors (`hypotheses.txt`). It only exercises
`scripts/wer_benchmark.py` and `src/eval/wer.py`. It is **not** a benchmark and
its WER must never be reported as Whisper's WER.

Expected result after normalisation (lower-case, no punctuation): 102 reference
words, 2 substitutions (`Lopez/Lopes`, `am/I'm`), 2 deletions (`the`, `I`),
1 insertion (`please`), WER = 5/102 = 0.049.
