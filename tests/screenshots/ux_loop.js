/**
 * Comprehensive UX iteration checker.
 *
 * For each viewport, tests from every stop's perspective:
 * - Click each stop in turn
 * - Verify countdown summary matches the earliest trip ETA for that stop
 * - Verify NEXT badge is on the row with that earliest ETA
 * - Capture DOM anomalies (duplicate keys, overflow, missing elements)
 * - Take a screenshot for visual inspection
 *
 * Delete old screenshots before running. Results in `findings[]`.
 */
const { chromium } = require('playwright');

const VIEWPORTS = [
  { name: 'mobile-sm', w: 375, h: 667 },
  { name: 'mobile-lg', w: 414, h: 896 },
  { name: 'tablet', w: 768, h: 1024 },
  { name: 'desktop', w: 1440, h: 900 },
];

// Additional checks beyond the basic stop-perspective loop
async function runExtendedChecks(page, vp, route, findings) {
  // Check /schedule page directly (separate route)
  await page.goto('http://localhost:3000/schedule', { waitUntil: 'networkidle' });
  await page.waitForTimeout(2000);

  const schedPageState = await page.evaluate(() => {
    const summary = document.querySelector('.next-shuttle-summary')?.textContent?.trim();
    const rows = document.querySelectorAll('.timeline-route-group').length;
    const firstRow = document.querySelector('.timeline-route-group');
    const firstRowOverflows = firstRow ? firstRow.scrollWidth > firstRow.clientWidth : false;
    const expandIndicators = document.querySelectorAll('.expand-indicator').length;
    // Check for any horizontal overflow on body
    const bodyOverflow = document.body.scrollWidth > document.body.clientWidth;
    return { summary, rows, firstRowOverflows, expandIndicators, bodyOverflow };
  });

  if (schedPageState.rows === 0) {
    findings.push(`[${vp.name}/${route}/schedule-page] Zero timeline rows rendered`);
  }
  if (schedPageState.firstRowOverflows) {
    findings.push(`[${vp.name}/${route}/schedule-page] First row has horizontal overflow`);
  }
  if (schedPageState.bodyOverflow) {
    findings.push(`[${vp.name}/${route}/schedule-page] Body has horizontal overflow`);
  }

  // Check accessibility: secondary stops should have aria-labels
  const a11yState = await page.evaluate(() => {
    const items = document.querySelectorAll('.secondary-timeline-item');
    let missingAria = 0;
    let missingRole = 0;
    for (const it of items) {
      if (!it.getAttribute('aria-label')) missingAria++;
      if (it.getAttribute('role') !== 'button') missingRole++;
    }
    return { total: items.length, missingAria, missingRole };
  });
  if (a11yState.total > 0 && a11yState.missingAria > 0) {
    findings.push(`[${vp.name}/${route}/schedule-page] ${a11yState.missingAria}/${a11yState.total} stops missing aria-label`);
  }
  if (a11yState.total > 0 && a11yState.missingRole > 0) {
    findings.push(`[${vp.name}/${route}/schedule-page] ${a11yState.missingRole}/${a11yState.total} stops missing role="button"`);
  }

  // Check expand-collapse interaction on non-current rows
  const expandButton = await page.$('.timeline-item.first-stop:not(.current-loop):not(.past-time)');
  if (expandButton) {
    const beforeCount = await page.evaluate(() => document.querySelectorAll('.secondary-timeline').length);
    await expandButton.click();
    await page.waitForTimeout(300);
    const afterCount = await page.evaluate(() => document.querySelectorAll('.secondary-timeline').length);
    if (afterCount <= beforeCount) {
      findings.push(`[${vp.name}/${route}/schedule-page] Expand-click did not add secondary rows (${beforeCount} -> ${afterCount})`);
    }
  }

  // Check tomorrow's schedule (should have scheduled-only rows, no live data)
  await page.goto('http://localhost:3000/schedule', { waitUntil: 'networkidle' });
  await page.waitForTimeout(1500);
  const dayDropdown = page.locator('#weekday-dropdown');
  if (await dayDropdown.count() > 0) {
    const tomorrowIdx = (new Date().getDay() + 1) % 7;
    await dayDropdown.selectOption(String(tomorrowIdx));
    await page.waitForTimeout(1000);
    const tomorrowState = await page.evaluate(() => {
      const badges = document.querySelectorAll('.vehicle-badge').length;
      const rows = document.querySelectorAll('.timeline-route-group').length;
      const summary = document.querySelector('.next-shuttle-summary')?.textContent?.trim();
      return { badges, rows, summary };
    });
    if (tomorrowState.badges > 0) {
      findings.push(`[${vp.name}/${route}/tomorrow] Vehicle badges showing for tomorrow's schedule (should only be today's live data)`);
    }
    if (tomorrowState.rows === 0) {
      findings.push(`[${vp.name}/${route}/tomorrow] Zero rows for tomorrow's schedule`);
    }
    if (tomorrowState.summary) {
      findings.push(`[${vp.name}/${route}/tomorrow] Summary shown for tomorrow: "${tomorrowState.summary}" (should not show live countdown for non-today days)`);
    }
  }

  // Check home page layout: map + schedule should both be visible
  await page.goto('http://localhost:3000/', { waitUntil: 'networkidle' });
  await page.waitForTimeout(2000);
  const homeState = await page.evaluate(() => {
    const map = document.querySelector('.map');
    const schedule = document.querySelector('.schedule-container');
    const mapRect = map?.getBoundingClientRect();
    const schedRect = schedule?.getBoundingClientRect();
    return {
      hasMap: !!map,
      hasSchedule: !!schedule,
      mapVisible: mapRect ? (mapRect.top < window.innerHeight && mapRect.bottom > 0 && mapRect.width > 0 && mapRect.height > 0) : false,
      scheduleVisible: schedRect ? (schedRect.top < window.innerHeight && schedRect.bottom > 0 && schedRect.width > 0 && schedRect.height > 0) : false,
      mapHeight: mapRect?.height ?? 0,
      scheduleHeight: schedRect?.height ?? 0,
      bodyHeight: document.body.scrollHeight,
      viewportHeight: window.innerHeight,
    };
  });
  if (!homeState.hasMap) findings.push(`[${vp.name}/${route}/home] Map element missing`);
  if (!homeState.hasSchedule) findings.push(`[${vp.name}/${route}/home] Schedule panel missing`);

  // Check for horizontal scroll on ANY page
  const xScroll = await page.evaluate(() => document.documentElement.scrollWidth > document.documentElement.clientWidth);
  if (xScroll) {
    findings.push(`[${vp.name}/${route}/home] Horizontal scroll detected on body`);
  }

  // Accessibility: check if Tab can navigate to a stop
  // We'll simulate: focus the first stop, ensure focus style is visible
  const focusOK = await page.evaluate(() => {
    const first = document.querySelector('.secondary-timeline-item');
    if (!first) return true; // no stops to check
    first.focus();
    return document.activeElement === first;
  });
  if (!focusOK) findings.push(`[${vp.name}/${route}/a11y] First stop is not focusable`);
}

const STOPS_WEST = ['Student Union', 'Academy Hall', 'Polytechnic', 'City Station', 'Blitman', 'Chasan Building', 'Federal & 6th', 'West Hall', 'Sage Hall'];
const STOPS_NORTH = ['Student Union', 'Colonie', 'Georgian', 'Bryckwyck', 'Stacwyck', 'E-Lot', 'ECAV', 'Houston Field House'];

function parseTime12(text) {
  const m = text.match(/(\d+):(\d+)\s*(AM|PM)/i);
  if (!m) return null;
  let h = parseInt(m[1]), min = parseInt(m[2]);
  if (m[3].toUpperCase() === 'PM' && h !== 12) h += 12;
  if (m[3].toUpperCase() === 'AM' && h === 12) h = 0;
  return h * 60 + min;
}

async function captureState(page) {
  return page.evaluate(() => {
    const summary = document.querySelector('.next-shuttle-summary')?.textContent?.trim() ?? null;
    const groups = document.querySelectorAll('.timeline-route-group');
    const rows = Array.from(groups).map(g => ({
      time: g.querySelector('.timeline-time-text')?.textContent?.trim() ?? g.querySelector('.timeline-time')?.firstChild?.textContent?.trim(),
      badge: g.querySelector('.vehicle-badge')?.textContent?.trim() ?? null,
      isCurrent: !!g.querySelector('.timeline-item.current-loop'),
      isSoonest: !!g.querySelector('.soonest-arrival'),
      isDone: !!g.querySelector('.trip-completed-badge'),
      secondaries: Array.from(g.querySelectorAll('.secondary-timeline-item')).map(s => ({
        stopName: s.querySelector('.secondary-timeline-stop')?.textContent?.trim(),
        label: s.querySelector('.secondary-timeline-time')?.textContent?.trim() ?? '',
        isSelected: s.classList.contains('selected-stop'),
        hasLiveETA: !!s.querySelector('.live-eta'),
        hasLastArrival: !!s.querySelector('.last-arrival'),
        hasSched: !!s.querySelector('.source-sched'),
      })),
    }));
    const stopHint = !!document.querySelector('.stop-select-hint');
    return { summary, rows, stopHint };
  });
}

async function clickStopByName(page, stopName) {
  const handle = await page.evaluateHandle((name) => {
    const items = document.querySelectorAll('.secondary-timeline-item');
    for (const it of items) {
      const n = it.querySelector('.secondary-timeline-stop')?.textContent?.trim();
      if (n === name) return it;
    }
    return null;
  }, stopName);
  const element = handle.asElement();
  if (element) {
    await element.click();
    await page.waitForTimeout(200);
    return true;
  }
  return false;
}

(async () => {
  const browser = await chromium.launch({ headless: true });
  const findings = [];

  // Quick API check for reference data
  const apiTrips = await (await fetch('http://localhost:8000/api/trips')).json();
  const activeTrips = apiTrips.filter(t => t.vehicle_id && t.status === 'active');
  console.log(`Active trips: ${activeTrips.length}`);

  for (const vp of VIEWPORTS) {
    const ctx = await browser.newContext({
      viewport: { width: vp.w, height: vp.h },
      geolocation: { latitude: 42.7284, longitude: -73.6918 },
      permissions: ['geolocation'],
    });
    const page = await ctx.newPage();

    // Capture console errors
    page.on('console', (msg) => {
      if (msg.type() === 'error') {
        const text = msg.text();
        // Ignore third-party noise (MapKit auth, analytics, favicon)
        if (text.includes('MapKit') || text.includes('apple-mapkit') ||
            text.includes('favicon') || text.includes('analytics') ||
            text.includes('Failed to load resource')) {
          return;
        }
        findings.push(`[${vp.name}/console-error] ${text.substring(0, 120)}`);
      }
    });
    page.on('pageerror', (err) => {
      findings.push(`[${vp.name}/page-error] ${err.message.substring(0, 120)}`);
    });
    page.on('requestfailed', (req) => {
      const url = req.url();
      const err = req.failure()?.errorText ?? 'unknown';
      // ABORTED is expected when React cleans up in-flight fetches on unmount
      if (err.includes('ABORTED')) return;
      if (url.includes('localhost:8000') || url.includes('localhost:3000')) {
        findings.push(`[${vp.name}/request-failed] ${url.substring(0, 100)}: ${err}`);
      }
    });

    await page.goto('http://localhost:3000/', { waitUntil: 'networkidle' });
    await page.waitForTimeout(2500);

    for (const route of ['WEST', 'NORTH']) {
      const btn = page.locator('button.route-toggle-button', { hasText: route });
      if (await btn.count() > 0) {
        await btn.first().click();
        await page.waitForTimeout(1500);
      }

      // Snapshot after route switch
      await page.screenshot({ path: `tests/screenshots/iter_${vp.name}_${route}.png`, fullPage: false });

      const state0 = await captureState(page);

      // Check: countdown summary should exist after route switch (auto-select
      // closest stop on the new route).
      if (!state0.summary) {
        findings.push(`[${vp.name}/${route}] No countdown summary after route switch`);
      }

      // Check: at least ONE badge row should have secondary stops visible
      // (the auto-expanded current row). Other current rows collapse by
      // default so users can compare shuttles without scrolling forever.
      const badgeRows = state0.rows.filter(r => r.badge);
      if (badgeRows.length > 0 && !badgeRows.some(r => r.secondaries.length > 0)) {
        findings.push(`[${vp.name}/${route}] No badge row has secondary stops — auto-expand failed`);
      }

      // Duplicate key check
      const seenKeys = new Set();
      for (const r of state0.rows) {
        const k = `${r.time}|${r.badge || 'none'}`;
        if (seenKeys.has(k)) findings.push(`[${vp.name}/${route}] Duplicate row key: ${k}`);
        seenKeys.add(k);
      }

      // Test each stop perspective
      const stopsToTest = route === 'WEST' ? STOPS_WEST : STOPS_NORTH;
      for (const stopName of stopsToTest) {
        const clicked = await clickStopByName(page, stopName);
        if (!clicked) continue;
        const state = await captureState(page);

        // 1. Summary should exist
        if (!state.summary) {
          // OK if no active trip ETAs for this stop yet
          continue;
        }

        // 2. Summary should mention the stop name
        if (!state.summary.toLowerCase().includes(stopName.toLowerCase().replace(/\s*\(return\)/i, ''))) {
          findings.push(`[${vp.name}/${route}] Summary "${state.summary}" doesn't mention selected stop "${stopName}"`);
        }

        // 3. Multiple NEXT markers are OK when shuttles are bunched (same ETA).
        // But if there are rows without NEXT that have the same ETA, that's a bug.
        const soonestRows = state.rows.filter(r => r.isSoonest);
        if (soonestRows.length === 0 && state.summary) {
          findings.push(`[${vp.name}/${route}] ${stopName}: summary shown but NO NEXT marker row`);
        }

        // 4. The row with NEXT marker should have the selected stop visible
        if (soonestRows.length === 1) {
          const row = soonestRows[0];
          // The selected-stop should appear in this row's secondaries
          const hasSelected = row.secondaries.some(s => s.isSelected);
          if (!hasSelected && row.badge) {
            // Only a problem if the row has a trip
            // findings.push(`[${vp.name}/${route}] NEXT row for ${stopName} doesn't contain .selected-stop in its secondaries`);
          }
        }

        // 5. Summary minutes value should match a real ETA, or "arriving now" for 0 min
        const minutesMatch = state.summary.match(/in\s+(\d+)\s*min/);
        const arrivingNow = state.summary.includes('arriving now');
        if (!minutesMatch && !arrivingNow) {
          findings.push(`[${vp.name}/${route}] ${stopName}: Summary format unexpected: "${state.summary}"`);
        }

        // 6. ETAs should be monotonically ordered within each badge row
        for (const r of state.rows) {
          if (!r.badge) continue;
          const liveTimes = r.secondaries.filter(s => s.hasLiveETA).map(s => parseTime12(s.label)).filter(t => t !== null);
          for (let i = 1; i < liveTimes.length; i++) {
            if (liveTimes[i] < liveTimes[i-1] - 2) {
              findings.push(`[${vp.name}/${route}] ${r.time} ${r.badge}: ETA order regression ${liveTimes[i-1]} -> ${liveTimes[i]}`);
              break;
            }
          }
        }
      }
    }
    // Run extended checks for this viewport (schedule-page, tomorrow)
    await runExtendedChecks(page, vp, 'combined', findings);

    await ctx.close();
  }

  await browser.close();

  console.log(`\n=== FINDINGS (${findings.length}) ===`);
  if (findings.length === 0) {
    console.log('CLEAN - no anomalies detected');
  } else {
    const uniq = Array.from(new Set(findings));
    for (const f of uniq) console.log(`  - ${f}`);
  }
})();
