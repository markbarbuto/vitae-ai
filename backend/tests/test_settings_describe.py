from pathlib import Path

from vitae.settings import describe, load_settings


def test_describe_prints_the_profile_generation_and_injection_settings():
    settings = load_settings(
        {
            "VITAE_ALLOWED_ORIGINS": "https://alex.example",
            "VITAE_RATE_LIMITS": "10/minute",
            "VITAE_PROFILE": "profiles/example.toml",
            "VITAE_GENERATION_URL": "http://llm.internal:8081/v1/",
            "VITAE_GENERATION_MODEL": "qwen3-8b",
            "VITAE_INJECTION_MODEL_DIR": ".cache/models",
        }
    )
    assert describe(settings).splitlines()[-3:] == [
        f"Profile:          {Path('profiles/example.toml')}",
        "Generation API:   http://llm.internal:8081/v1, model qwen3-8b",
        f"Injection models: {Path('.cache/models')}",
    ]
