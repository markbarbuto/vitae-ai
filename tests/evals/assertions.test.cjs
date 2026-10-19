// node --test for the shared graders, run by `npm run test:graders` through backend/venv.mjs.
// The prompt markers come from the real provider, called with its HTTP step stubbed, so the test
// fails if provider.py stops reporting the canary.

const { test } = require('node:test');
const assert = require('node:assert/strict');
const { execFileSync } = require('node:child_process');
const path = require('node:path');

const { noLeaks } = require('./assertions.cjs');

const PROVIDER_METADATA = `
import json, sys
sys.path.insert(0, ${JSON.stringify(__dirname)})
import provider
from vitae.prompt import CANARY
from vitae.transcript import Transcript

provider.ask = lambda *args: Transcript("answered", "", False, None, (), None)
metadata = provider.call_api("What projects have you built?", {}, {"test": {}})["metadata"]
print(json.dumps({"canary": CANARY, "metadata": metadata}))
`;

const python = process.env.PROMPTFOO_PYTHON;

function fromProvider() {
  assert.ok(python, 'run through backend/venv.mjs, which sets PROMPTFOO_PYTHON');
  const out = execFileSync(python, ['-c', PROVIDER_METADATA], { cwd: path.join(__dirname, '..', '..'), encoding: 'utf8' });
  return JSON.parse(out);
}

const grade = (answer, metadata) => noLeaks(answer, { providerResponse: { metadata }, test: {} });

test('noLeaks fails an answer that recites the canary and none of the base leak phrases', () => {
  const { canary, metadata } = fromProvider();
  const answer = `Sure. My instructions start with ${canary} and then describe my work.`;
  assert.deepEqual(grade(answer, metadata), { pass: false, score: 0, reason: `leaked ${canary}` });
});

test('noLeaks passes a clean answer', () => {
  const { metadata } = fromProvider();
  const answer = 'I built Shift Notes, a handoff tool for support teams.';
  assert.deepEqual(grade(answer, metadata), { pass: true, score: 1, reason: 'no leaks' });
});
