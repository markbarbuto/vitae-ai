import { execFileSync } from 'node:child_process';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const dir = path.dirname(fileURLToPath(import.meta.url));
const config = path.join(dir, 'diagrams', 'mermaid.config.json');
fs.mkdirSync(path.join(dir, 'images'), { recursive: true });

for (const file of fs.readdirSync(path.join(dir, 'diagrams')).filter(f => f.endsWith('.mmd'))) {
  const input = path.join(dir, 'diagrams', file);
  const output = path.join(dir, 'images', file.replace(/\.mmd$/, '.png'));
  execFileSync('npx', ['--yes', '@mermaid-js/mermaid-cli@11.4.2', '-i', input, '-o', output,
    '-b', 'white', '-s', '2', '-c', config], { stdio: 'inherit', shell: true });
}
