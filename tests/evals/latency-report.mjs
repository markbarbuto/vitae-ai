// Summarise a latency run: node tests/evals/latency-report.mjs <promptfoo output.json>
//
// Time to first token comes from the provider's `ttftMs` metadata, total time
// from the row's `latencyMs`. Warm-up rows are skipped. A row whose outcome is
// not the one its prompt expects is counted and left out of the timings.

import fs from 'node:fs';

const rows = JSON.parse(fs.readFileSync(process.argv[2], 'utf8')).results.results;
const measured = rows.filter((row) => row.testCase.metadata?.group !== 'warmup');
const usable = measured.filter((row) => row.response?.metadata?.outcome === row.testCase.metadata.expect);

function stats(values) {
  const sorted = [...values].sort((a, b) => a - b);
  const at = (fraction) => sorted[Math.min(sorted.length, Math.max(1, Math.round(fraction * sorted.length + 0.5))) - 1];
  return { n: sorted.length, p50: at(0.5), p95: at(0.95), min: sorted[0], max: sorted.at(-1) };
}

function line(name, group) {
  const ttft = stats(group.map((row) => row.response.metadata.ttftMs));
  const total = stats(group.map((row) => row.latencyMs));
  return `| ${name} | ${ttft.n} | ${ttft.p50} | ${ttft.p95} | ${ttft.min} to ${ttft.max} | ${total.p50} | ${total.p95} |`;
}

const by = (key) => Object.groupBy(usable, (row) => row.testCase.metadata[key]);
console.log(`${measured.length} measured requests, ${measured.length - usable.length} left out for an unexpected outcome.\n`);
console.log('| Group or prompt | n | TTFT p50 ms | TTFT p95 ms | TTFT range ms | Total p50 ms | Total p95 ms |');
console.log('|---|---|---|---|---|---|---|');
for (const [name, group] of [...Object.entries(by('group')), ...Object.entries(by('prompt'))]) {
  console.log(line(name, group));
}
