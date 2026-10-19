// Run a command with backend/.venv active: `node backend/venv.mjs python -m vitae.schema contract`.
// npm scripts use it instead of `uv run` so they work without uv on PATH, and instead of a
// pythonExecutable path in the promptfoo configs, which differs between Windows and POSIX.
// promptfoo picks the interpreter up from PROMPTFOO_PYTHON.

import { existsSync } from 'node:fs';
import path from 'node:path';
// Node runs Windows .cmd shims such as promptfoo only through cmd.exe, which re-parses the
// arguments. cross-spawn escapes them for it, so each argument arrives exactly as given.
import spawn from 'cross-spawn';

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
const [command, ...args] = process.argv.slice(2);
const { status, error } = spawn.sync(command, args, { env, stdio: 'inherit' });
if (error) throw error;
process.exit(status ?? 1);
