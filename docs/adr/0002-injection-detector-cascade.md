# ADR 0002. Prompt-injection detection: two classifiers in a cascade

- **Status.** Accepted, 2026-10-19.
- **Amends.** [ADR 0001](0001-injection-detector.md). The rules and the ONNX runtime stay as decided there. Only "one classifier, one pass per request" changes.

## Context

ADR 0001 chose rules plus one small ONNX classifier, picked by measurement on the dev split. The measurement used these candidates:

- **Llama Prompt Guard 2, 22M.** It caught 89.9% of the dev attacks a detector can recognise from the message alone. Its threshold was set so that it wrongly refuses at most 1% of on-topic recruiter questions. Its wrong refusals are questions that mention "instructions" or "ignore".
- **Horizon-Labs prompt-injection-guard-small.** Alone, it wrongly refuses short or misspelled questions ("react exp", "waht projcts").

The two models make different mistakes. The mean of their two scores, at the same recruiter limit, catches 97.9% of attacks. That figure is on the dev attacks neither model trained on, and on the same set Prompt Guard alone catches 95.4%. The mean also brings the success rate of mutated attacks (encodings, leetspeak, spacing) down to between 0.2% and 5.8%.

No single candidate reached the target of 95% while keeping recruiter false refusals at 1% or below.

## Decision

The injection classifier stage scores a message with Prompt Guard 2 first, over the same views (lines and decoded payloads) as before.

- If the score can't reach the threshold whatever Horizon says, the message passes on Prompt Guard alone. Because the decision rule is the mean of the two scores, this happens when Prompt Guard is below `2t − 1`.
- Otherwise Horizon scores the same views. The message is refused when the mean of the two scores reaches the threshold `t`.

The cascade is exact: it refuses exactly the messages the full mean would refuse. Most messages never reach Horizon.

The weights stay equal. A weight fitted on dev data was rejected, so the held-out run doesn't measure a choice tuned to that data.

`calibration.json` records both models: directory, ONNX file and sha256 for each. Startup refuses a model that isn't the file the threshold was measured on.

## Consequences

- **Memory and disk.** The API image carries both models, an extra 268 MB (Horizon's int8 file) to 563 MB (fp32). Which file ships is decided by measuring recall and latency, and recorded in the calibration.
- **Latency.** Typical requests pay for Prompt Guard only. Requests above `2t − 1` also pay for Horizon (about 34 ms at 4 threads). The benchmark reports the cascade's measured cost per request and the share of requests that reach Horizon.
- **Held-out evidence is thinner for the pair.** Horizon trained on the deepset, Gandalf and jackhhao collections, so the pair's held-out score is clean only on the others. The held-out report gives:
  - the pair's numbers on the collections neither model saw;
  - Prompt Guard's numbers on every collection.
- **Licences.** Horizon is Apache-2.0, and its notice joins Prompt Guard's in `THIRD_PARTY_NOTICES.md`. Prompt Guard's Llama 4 licence duties from ADR 0001 still apply.
- **The conversion was checked against Meta's original.**
  - Meta's original weights were run through transformers 4.57 as the reference, and compared with the third-party ONNX files on 2,000 dev texts.
  - The fp32 conversion matches within 7.7e-6, so it ships.
  - The int8 file is up to 0.89 away and makes a different decision on 3.5% of texts, so it doesn't ship.
  - The shipped tokenizer (the `tokenizers` library, with padding off) produces exactly transformers 4.57's token ids. A regression test pins those ids.
  - transformers 5.x tokenizes this model differently, so nothing may compute reference scores with it.
