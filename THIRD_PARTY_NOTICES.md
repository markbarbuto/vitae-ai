# Third-party notices

Vitae is MIT licensed. It downloads the models and datasets below at pinned revisions, checked by sha256 (`backend/bench/manifest.json`). None of them is committed to this repository.

## Models the server runs

### Llama Prompt Guard 2, 22M

Built with Llama.

- Source: Meta, `meta-llama/Llama-Prompt-Guard-2-22M`. The ONNX conversion used is `gravitee-io/Llama-Prompt-Guard-2-22M-onnx`.
- Licence: Llama 4 Community License Agreement, https://www.llama.com/llama4/license/. Use is also subject to the Llama 4 Acceptable Use Policy.
- Notice: Llama 4 is licensed under the Llama 4 Community License, Copyright © Meta Platforms, Inc. All Rights Reserved.

### Horizon-Labs prompt-injection-guard-small

- Source: `Horizon-Labs/prompt-injection-guard-small`, fine-tuned from `jhu-clsp/mmBERT-small` (MIT).
- Licence: Apache License 2.0, https://www.apache.org/licenses/LICENSE-2.0.

## Datasets used only to evaluate the guards

These are downloaded to a local cache to measure the guards. They are never shipped or redistributed.

| Dataset | Licence |
|---|---|
| deepset/prompt-injections | Declared as both Apache-2.0 and CC-BY-4.0. Attributed here under CC-BY-4.0 |
| Lakera/gandalf_ignore_instructions | MIT. Citation: arXiv:2501.07927 |
| jackhhao/jailbreak-classification | Apache-2.0 |
| NVIDIA/garak payloads, DAN prompts and PromptInject templates | Apache-2.0, © 2023 Leon Derczynski and NVIDIA. The PromptInject files are MIT, © 2020 Agency Enterprise |
| reshabhs/SPML_Chatbot_Prompt_Injection | MIT. Citation: arXiv:2402.11755 |
| JailbreakBench/JBB-Behaviors | MIT |
| gretelai/gretel-pii-masking-en-v1 | Apache-2.0 |
| gretelai/synthetic_pii_finance_multilingual | Apache-2.0 |
| diasks2/pragmatic_segmenter (Golden Rules) | MIT, © 2015 Kevin S. Dias |
| nipunsadvilkar/pySBD (English test cases) | MIT |
