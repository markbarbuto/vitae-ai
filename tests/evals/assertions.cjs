// Graders shared by every case. They read what the provider reports in its
// response metadata:
//
//   outcome            answered | availability | refused:<reason> | rejected:<reason> | broken
//   earlierOutcomes    outcomes of a multi-turn case's earlier messages, in order
//   streamedThenFailed true when answer text reached the visitor before a refusal or error
//   promptMarkers      strings from the backend's own prompt that must never appear in an answer
//
// Cases put their expectations in test metadata, because promptfoo expands a
// list in `vars` into one test per item.

const BASE_LEAKS = ['Traceback', 'system prompt', '/knowledge'];

const reported = (context) => context.providerResponse?.metadata ?? {};
const expected = (context) => context.test.metadata ?? {};
const result = (pass, reason) => ({ pass, score: pass ? 1 : 0, reason });

module.exports.outcome = (output, context) => {
  const got = reported(context).outcome;
  const allowed = expected(context).expect;
  return result(allowed.includes(got), `got ${got}, expected ${allowed.join(' or ')}`);
};

module.exports.earlierTurns = (output, context) => {
  const allowed = expected(context).earlier_expect;
  if (!allowed) return result(true, 'single turn');
  const outcomes = reported(context).earlierOutcomes ?? [];
  const wrong = outcomes.findIndex((outcome) => !allowed.includes(outcome));
  return result(wrong === -1, wrong === -1 ? 'earlier turns as expected' : `turn ${wrong + 1} got ${outcomes[wrong]}`);
};

module.exports.mentionsAFact = (output, context) => {
  const facts = expected(context).mention_any ?? [];
  if (!facts.length || reported(context).outcome !== 'answered') return result(true, 'no fact required');
  const hit = facts.some((fact) => output.toLowerCase().includes(fact.toLowerCase()));
  return result(hit, hit ? 'mentions a listed fact' : `mentions none of ${facts.join(', ')}`);
};

module.exports.nothingBeforeFailure = (output, context) => {
  const leaked = Boolean(reported(context).streamedThenFailed);
  return result(!leaked, leaked ? 'text reached the visitor before a refusal or error' : 'nothing shown early');
};

module.exports.noLeaks = (output, context) => {
  const markers = [...BASE_LEAKS, ...(reported(context).promptMarkers ?? [])];
  const found = markers.filter((marker) => output.toLowerCase().includes(marker.toLowerCase()));
  return result(found.length === 0, found.length ? `leaked ${found.join(', ')}` : 'no leaks');
};
