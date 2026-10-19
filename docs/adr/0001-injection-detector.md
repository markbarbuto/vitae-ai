# ADR 0001. Prompt-injection detection: rules plus a small ONNX classifier

- **Status.** Accepted, 2026-10-18.
- **Decides.** How the input pipeline detects prompt injection and jailbreak attempts.

## Context

The input pipeline needs to refuse prompt injection and jailbreak attempts before they reach the generation model. The target is to catch at least 95% of injection and jailbreak prompts from public corpora this project did not write, while wrongly refusing at most 1% of ordinary questions. Both are measured on held-out data.

These constraints are fixed:

- **CPU only.** Every model runs acceptably on CPU in Docker.
- **No hosted inference and no `transformers` library.** Models are downloaded once, pinned by repository and hash, and run locally.
- **Anyone can run it.** An owner who has never seen the code gets from clone to a guarded assistant without help. Every extra account, form or token they need is friction.
- **Llama Guard 3 1B moderates input in a later stage.** It is a different job. It classifies harm categories (weapons, self-harm and so on), not instructions trying to take over the assistant.

Rules alone match known phrasings: "ignore previous instructions", role-play setups, encoded payloads. Public corpora are full of paraphrases, other languages and obfuscation that a hand-written list misses. A rules-only detector would need its target lowered to pass.

## Options considered

**(a) Rules plus a small classifier exported to ONNX, run in the API process with `onnxruntime` and `tokenizers`.**

- Rules catch the cheap, known cases in microseconds and explain themselves in logs.
- A classifier catches paraphrases and other languages.
- Both run on CPU without a model server, and `onnxruntime` and `tokenizers` are small and need no `transformers`.

**(b) Rules plus a GGUF classifier served by llama.cpp.**

- It would sit with the other model servers.
- Prompt-injection classifiers are encoder models (DeBERTa or BERT families). llama.cpp is built for generative decoders, so this means either an unsupported architecture or a generative model asked to classify, which is slower and less calibrated.
- It also can't be measured until the model servers exist.

**(c) Rules only.**

- No model, no download, no licence.
- It can't meet the target on public corpora, so the target would have to drop.

## Decision

**Option (a).** The input pipeline runs the rule guard first and the classifier second. Either one can refuse with the `injection` reason. The classifier sees the normalized message and splits anything longer than its window into overlapping windows.

**The classifier is chosen by measurement, not reputation.** Each candidate is scored on the same held-out corpora, leaving out any corpus that candidate was trained on, and timed on this machine's CPU. The winner is the one that meets the recall and false-positive targets at the lowest CPU cost. These are the candidates, all with ONNX weights available:

| Candidate | Licence | Size | Notes |
|---|---|---|---|
| Meta Llama Prompt Guard 2, 22M | Llama 4 Community License | about 71M parameters with embeddings | Covers injection and jailbreak; multilingual. Meta's repository is gated by manual approval. An ungated third-party ONNX conversion exists. |
| ProtectAI deberta-v3-base-prompt-injection-v2 | Apache-2.0 | 184M | ONNX in the repository; ungated. English only. Doesn't cover jailbreaks. Archived. |
| Horizon-Labs prompt-injection-guard-small | Apache-2.0 | about 141M, int8 ONNX available | Ungated; multilingual. Only weeks old; the published scores are the authors' own. |

**Default if the measurements are close: Prompt Guard 2 22M.** It is the smallest, it covers jailbreaks as well as injection, and its published recall at 1% false positives is the strongest of the three. Its licence adds little, because Llama Guard 3 already brings Llama licence duties. If its gating can't be avoided without an unverified third-party copy, the best Apache-2.0 candidate wins instead.

**Any third-party conversion is checked before it is trusted.** Its scores on the held-out corpora must match the original model's within a fixed tolerance.

## Consequences

- **Download.** The model joins the pinned, hash-checked download step with the other models. The API image grows by the model's size, about 70 to 300 MB depending on the winner and quantization.
- **Per-request cost.** One classifier pass per request, measured and reported as the injection stage's cost. It doesn't count in the non-model overhead budget.
- **Licence obligations.**
  - **Prompt Guard:** a licence copy and Notice file in `THIRD_PARTY_NOTICES.md`, and "Built with Llama" in the documentation. The same duties Llama Guard 3 already brings.
  - **An Apache-2.0 model:** its notice only.
- **CI.** CI downloads the model at test time, which works only for an ungated source. A gated model would need a Hugging Face token, and secrets aren't available to pull requests from forks.
- **Regression gate.** The guard benchmark reruns the held-out scores on every change. If a later model or rule change lowers recall or raises false positives, CI fails.
- **Limits.** Rules plus a classifier still miss new attack styles. The output guard and the canary leak markers stay as the second line of defence, and every bypass found becomes a permanent regression case.
