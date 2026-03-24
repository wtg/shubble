// Generate public/config.json from environment variables.
// Used by `npm run dev` and `npm run build` for local and Docker dev.
// In production, this is handled by envsubst in docker/frontend/entrypoint.sh instead.

import { readFileSync, writeFileSync } from 'fs';
import { dirname, join } from 'path';
import { fileURLToPath } from 'url';

const __dirname = dirname(fileURLToPath(import.meta.url));

// Load .env from repo root (env vars already set in the process take precedence)
try {
  const envFile = readFileSync(join(__dirname, '..', '..', '.env'), 'utf-8');
  for (const line of envFile.split('\n')) {
    const trimmed = line.trim();
    if (!trimmed || trimmed.startsWith('#')) continue;
    const eqIndex = trimmed.indexOf('=');
    if (eqIndex === -1) continue;
    const key = trimmed.slice(0, eqIndex);
    const value = trimmed.slice(eqIndex + 1);
    if (!(key in process.env)) {
      process.env[key] = value;
    }
  }
} catch {
  // .env file is optional
}

const config = {
  apiBaseUrl: process.env.BACKEND_URL || 'http://localhost:8000',
  deployMode: process.env.DEPLOY_MODE || 'development',
  mapkitKey: process.env.MAPKIT_KEY || '',
  staticETAs: process.env.STATIC_ETAS === 'true'
};

writeFileSync(
  join(__dirname, '..', 'public', 'config.json'),
  JSON.stringify(config, null, 2) + '\n'
);
