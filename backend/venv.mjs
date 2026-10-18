// Run a command with backend/.venv active: `node backend/venv.mjs python -m vitae.schema contract`.
// npm scripts use it instead of `uv run` so they work without uv on PATH, and instead of a
// pythonExecutable path in the promptfoo configs, which differs between Windows and POSIX.
// promptfoo picks the interpreter up from PROMPTFOO_PYTHON.

import { spawnSync } from 'node:child_process';
import { existsSync } from 'node:fs';
import path from 'node:path';

const venv = path.join(import.meta.dirname, '.venv');
const bin = path.join(venv, process.platform === 'win32' ? 'Scripts' : 'bin');
const python = path.join(bin, process.platform === 'win32' ? 'python.exe' : 'python');

if (!existsSync(python)) {
  console.error(`No Python at ${python}. Run \`uv sync --project backend\` first.`);
  process.exit(1);
}

// Windows spells it Path; adding a second PATH key would leave the child with two.
const pathKey = Object.keys(process.env).find((key) => key.toUpperCase() === 'PATH') ?? 'PATH';
const env = {
  ...process.env,
  VIRTUAL_ENV: venv,
  PROMPTFOO_PYTHON: python,
  [pathKey]: `${bin}${path.delimiter}${process.env[pathKey] ?? ''}`,
};
const { status, error } = spawnSync(process.argv.slice(2).join(' '), { env, shell: true, stdio: 'inherit' });
if (error) throw error;
process.exit(status ?? 1);
