// Screenshots of the portfolio's assistant section, the visual reference for the chat UI.
//
//   node tests/visual/capture.mjs --write   capture into tests/visual/golden/
//   node tests/visual/capture.mjs           capture into tests/visual/out/ and diff against tests/visual/golden/
//
// The portfolio runs from its own folder with VITE_ASSISTANT_API_URL empty, so it
// answers with canned text after an 800 ms timer. A fake clock holds that timer,
// which makes the typing state stable instead of a race.

import { execFileSync, spawn } from 'node:child_process';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import pixelmatch from 'pixelmatch';
import { chromium } from 'playwright';
import { PNG } from 'pngjs';

const here = path.dirname(fileURLToPath(import.meta.url));
const portfolio = path.resolve(process.env.PORTFOLIO_DIR ?? path.join(here, '..', '..', '..', 'portfolio'));
const write = process.argv.includes('--write');
const goldenDir = path.join(here, 'golden');
const outDir = write ? goldenDir : path.join(here, 'out');
const port = 4317;
const url = `http://127.0.0.1:${port}/`;
const answerDelayMs = 800;

// Tall enough that the whole assistant section fits at scroll 0. A full-page
// screenshot would be simpler but does not paint the chat's scrollbar.
const viewports = {
  desktop: { viewport: { width: 1280, height: 1400 }, deviceScaleFactor: 1 },
  mobile: { viewport: { width: 390, height: 1400 }, deviceScaleFactor: 2, isMobile: true, hasTouch: true },
};
const themes = ['light', 'dark'];

const bubbles = (page) => page.locator('[aria-live="polite"] > .fade-up');
const typingDots = (page) => page.locator('[aria-live="polite"] > div:not(.fade-up)');
const chips = (page) => page.locator('.prompt-chip');

async function ask(page, chipIndex) {
  const before = await bubbles(page).count();
  await chips(page).nth(chipIndex).dispatchEvent('click');
  await bubbles(page).nth(before).waitFor();
  await typingDots(page).waitFor();
}

async function answer(page) {
  const before = await bubbles(page).count();
  await page.clock.runFor(answerDelayMs);
  await bubbles(page).nth(before).waitFor();
  await typingDots(page).waitFor({ state: 'detached' });
}

const states = {
  empty: async () => {},
  typing: async (page) => ask(page, 0),
  answered: async (page) => {
    await ask(page, 0);
    await answer(page);
  },
  long: async (page) => exchanges(page, 2),
  'extra-long': async (page) => {
    await exchanges(page, 3);
    const { scrollHeight, clientHeight } = await page.locator('[aria-live="polite"]').evaluate(
      (chat) => ({ scrollHeight: chat.scrollHeight, clientHeight: chat.clientHeight }),
    );
    if (scrollHeight <= clientHeight) throw new Error('extra-long chat does not scroll, so it shows no max height or scrollbar');
  },
};

async function exchanges(page, count) {
  for (let index = 0; index < count; index += 1) {
    await ask(page, index);
    await answer(page);
  }
}

async function assistantClip(page) {
  return page.evaluate(() => {
    const chipRow = document.querySelector('.prompt-chip-fade');
    const card = chipRow.nextElementSibling;
    const label = card.nextElementSibling;
    const rects = [chipRow, card, label].map((element) => element.getBoundingClientRect());
    const pad = { x: 24, top: 16, bottom: 16 };
    const left = Math.max(0, Math.min(...rects.map((r) => r.left)) - pad.x);
    const right = Math.min(document.documentElement.clientWidth, Math.max(...rects.map((r) => r.right)) + pad.x);
    const top = Math.min(...rects.map((r) => r.top)) - pad.top;
    const bottom = Math.max(...rects.map((r) => r.bottom)) + pad.bottom;
    if (window.scrollY !== 0 || bottom > window.innerHeight) throw new Error('assistant section does not fit the viewport');
    return { x: Math.floor(left), y: Math.floor(top), width: Math.ceil(right - left), height: Math.ceil(bottom - top) };
  });
}

async function startPortfolio() {
  const vite = path.join(portfolio, 'node_modules', 'vite', 'bin', 'vite.js');
  const server = spawn(process.execPath, [vite, '--port', String(port), '--strictPort', '--host', '127.0.0.1'], {
    cwd: portfolio,
    env: { ...process.env, VITE_ASSISTANT_API_URL: '' },
    stdio: 'ignore',
  });
  for (let attempt = 0; attempt < 120; attempt += 1) {
    try {
      if ((await fetch(url)).ok) return server;
    } catch {}
    await new Promise((resolve) => setTimeout(resolve, 250));
  }
  server.kill();
  throw new Error(`portfolio dev server did not start at ${url}`);
}

function diff(name) {
  const golden = PNG.sync.read(fs.readFileSync(path.join(goldenDir, name)));
  const actual = PNG.sync.read(fs.readFileSync(path.join(outDir, name)));
  if (golden.width !== actual.width || golden.height !== actual.height) {
    return `size ${actual.width}x${actual.height}, golden ${golden.width}x${golden.height}`;
  }
  const output = new PNG({ width: golden.width, height: golden.height });
  const pixels = pixelmatch(golden.data, actual.data, output.data, golden.width, golden.height, { threshold: 0, includeAA: true });
  if (pixels === 0) return null;
  fs.writeFileSync(path.join(outDir, `diff-${name}`), PNG.sync.write(output));
  return `${pixels} pixels differ`;
}

const server = await startPortfolio();
// Playwright hides scrollbars by default. The chat's scrollbar is part of the reference.
const browser = await chromium.launch({ ignoreDefaultArgs: ['--hide-scrollbars'] });
fs.mkdirSync(outDir, { recursive: true });
const failures = [];
try {
  for (const [viewportName, device] of Object.entries(viewports)) {
    for (const theme of themes) {
      for (const [stateName, reach] of Object.entries(states)) {
        const context = await browser.newContext({ ...device, colorScheme: theme, reducedMotion: 'no-preference' });
        await context.route('**/_vercel/**', (route) => route.abort());
        await context.addInitScript((value) => window.localStorage.setItem('portfolio-theme', value), theme);
        const page = await context.newPage();
        await page.clock.install({ time: new Date('2026-01-01T12:00:00Z') });
        await page.goto(url, { waitUntil: 'load' });
        await page.evaluate(() => document.fonts.ready);
        await chips(page).first().waitFor();
        await reach(page);
        const name = `${viewportName}-${theme}-${stateName}.png`;
        await page.screenshot({
          path: path.join(outDir, name),
          clip: await assistantClip(page),
          animations: 'disabled',
          caret: 'hide',
        });
        await context.close();
        const problem = write ? null : diff(name);
        if (problem) failures.push(`${name}: ${problem}`);
        console.log(`${problem ? 'DIFF' : write ? 'WROTE' : 'SAME'} ${name}${problem ? `  ${problem}` : ''}`);
      }
    }
  }
} finally {
  await browser.close();
  server.kill();
}
if (write) {
  const commit = execFileSync('git', ['-C', portfolio, 'rev-parse', '--short', 'HEAD'], { encoding: 'utf8' }).trim();
  console.log(`captured from portfolio commit ${commit}`);
}
if (failures.length) {
  console.error(`${failures.length} image(s) differ from tests/visual/golden`);
  process.exit(1);
}
