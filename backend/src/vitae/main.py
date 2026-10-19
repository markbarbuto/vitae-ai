"""Production wiring, `uvicorn vitae.main:app --workers 1 --no-proxy-headers`. No logic. There is no way to make it run a
scripted generator.

Run exactly one worker. Rate limits, sessions and the generation queue live in this process's
memory, so a second worker would double every limit and split each session between two processes.

Always pass --no-proxy-headers. Otherwise uvicorn rewrites the peer address from a client-supplied
X-Forwarded-For whenever the peer is loopback, before settings.client_ip decides whom to trust.

Startup fails loudly on a bad VITAE_* variable, a bad profile, a calibration.json that is still the
placeholder, or injection models that are not the files the threshold was calibrated on."""

import os
import sys

from vitae.app import create_app
from vitae.engine import Engine
from vitae.generation import OpenAICompatible
from vitae.guards.injection import Calibration, CascadeClassifier
from vitae.observe import JsonLogObserver
from vitae.profile import ProfileError, load_profile
from vitae.settings import SettingsError, load_settings

try:
    settings = load_settings(os.environ)
    profile = load_profile(settings.profile_path)
except (SettingsError, ProfileError) as error:
    sys.exit(str(error))
calibration = Calibration.load()
if (problem := calibration.model_problem(settings.injection_model_dir)) is not None:
    sys.exit(problem)

observer = JsonLogObserver(sys.stdout)
engine = Engine.build(
    settings,
    profile,
    generator=OpenAICompatible(settings.generation_url, settings.generation_model),
    classifier=CascadeClassifier.load(settings.injection_model_dir, calibration),
    observer=observer,
)
app = create_app(settings, engine, observer)
