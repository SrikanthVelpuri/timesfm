# Portfolio Site &mdash; AA Pricing Forecast with TimesFM

A static, single-page portfolio site that walks through this project end-to-end
across nine tabs: Overview, My Role, Timeline, Architecture, Fine-Tuning,
Inference, User Stories, Scenarios, and Dashboard.

The site is **100% static HTML/CSS/JS** with one CDN dependency (Chart.js). It
works locally over `file://`, on any static host, and on **GitHub Pages**.

---

## Local preview

The simplest way:

```bash
cd timesfm-forecasting/examples/airline-pricing-forecast/portfolio

# Python 3 (built-in)
python -m http.server 8000

# Then open http://localhost:8000 in a browser.
```

You can also just double-click `index.html` &mdash; the site is designed to work
on `file://` URLs (no module imports, all Chart.js comes from CDN).

---

## Hosting on GitHub Pages

You said you want to fork this as a personal project. Here is the path of
least resistance.

### Option A &mdash; Serve from a subfolder of `main`

This is the easiest if you keep the existing repo structure.

1. Fork the repo to your GitHub account.
2. In your fork, go to **Settings &rarr; Pages**.
3. Under **Build and deployment**, set:
   - **Source:** `Deploy from a branch`
   - **Branch:** `main`
   - **Folder:** `/ (root)`
4. The GitHub Actions runner deploys the repo. The portfolio will be at:
   ```
   https://<your-username>.github.io/<repo-name>/timesfm-forecasting/examples/airline-pricing-forecast/portfolio/
   ```

### Option B &mdash; Move portfolio to `docs/` (cleaner URL)

GitHub Pages has special support for a top-level `docs/` folder, which gives
you a shorter URL.

```bash
# In your fork, from the repo root:
mkdir -p docs
cp -r timesfm-forecasting/examples/airline-pricing-forecast/portfolio/* docs/
git add docs/
git commit -m "Publish AA pricing portfolio under docs/ for GitHub Pages"
git push
```

Then in **Settings &rarr; Pages**, set **Folder** to `/docs`. Your site will be
at:

```
https://<your-username>.github.io/<repo-name>/
```

### Option C &mdash; Dedicated repo (cleanest URL)

If you want this to be the *only* thing on a repo (e.g., your portfolio
site), create a new repo named exactly `<your-username>.github.io` and copy
the contents of this `portfolio/` folder to its root. GitHub Pages
automatically serves repos with that name from the root, with no extra
config:

```
https://<your-username>.github.io/
```

---

## File layout

```
portfolio/
├── index.html             single-page app with all 9 tabs
├── .nojekyll              tells GitHub Pages to skip Jekyll processing
├── README.md              this file
├── assets/
│   ├── css/
│   │   └── style.css      ~600 lines, no framework
│   └── js/
│       ├── app.js         tab navigation, hash-based routing
│       └── charts.js      Chart.js dashboards
└── data/
    └── (reserved for future JSON-driven dashboards)
```

The `.nojekyll` file is important &mdash; without it, GitHub Pages tries to
process the site through Jekyll, which strips files starting with `_` and
can interfere with paths.

---

## What the tabs cover

| Tab | Contents |
| --- | -------- |
| **Overview** | KPI cards, feature list, tech stack |
| **My Role** | Applied Scientist &rarr; ML Engineer transition table + 5 lessons |
| **Timeline** | 16-week Gantt chart + sprint cards + retro |
| **Architecture** | TimesFM 2.5 pipeline + AA covariate taxonomy |
| **Fine-Tuning** | MAPE progression chart, cost comparison chart, per-family LoRA chart |
| **Inference** | Latency-by-config chart, throughput vs cost chart, dead-end table |
| **User Stories** | 10 story cards with ML-Eng skill picked up per story |
| **Scenarios** | 6 production scenario cards + frequency chart |
| **Dashboard** | Coverage drift chart, forecast vs actual chart, per-route MAPE, SLOs |

All numbers in the charts are illustrative based on the synthetic example in
[`forecast_aa_pricing.py`](../forecast_aa_pricing.py). Replace
`assets/js/charts.js` data with your real numbers when this becomes a real
project.

---

## Customizing for your own use

To make this your own:

1. Edit the brand block in `index.html` (look for `<span class="brand-mark">AA</span>`)
   to your initials or a logo.
2. Edit the colors in `assets/css/style.css` &mdash; CSS variables at the top
   under `:root { ... }`.
3. Update the metric numbers in the KPI cards in `index.html` and the
   datasets in `assets/js/charts.js`.
4. Replace the user-stories cards with your real epics.
5. Update the timeline Gantt with your actual sprint structure.

Everything is in plain HTML/JS so it&apos;s easy to edit by hand &mdash; no
build step, no framework, no node_modules.

---

## Why no framework?

I deliberately built this with vanilla HTML/CSS/JS so that:

- It works on any static host with zero config.
- It works on `file://` (so reviewers without a server can preview it).
- There are no version pins to rot over time.
- Anyone forking it can edit it without an `npm install`.

Chart.js is the only dependency, and it&apos;s loaded from a CDN with a pinned
version (`@4.4.1`) for stability.

---

## Source files this site references

The site links back to the markdown docs in the parent folder:

- [../README.md](../README.md)
- [../user_stories.md](../user_stories.md)
- [../timeline.md](../timeline.md)
- [../model_architecture.md](../model_architecture.md)
- [../finetuning_aa_data.md](../finetuning_aa_data.md)
- [../inference_optimization.md](../inference_optimization.md)
- [../scenarios.md](../scenarios.md)

When hosted on GitHub Pages those links resolve to the rendered markdown via
GitHub&apos;s file viewer (since GH Pages does not render `.md` files directly
unless you publish through Jekyll). If you want them rendered inline, the
cleanest path is to convert the markdown to HTML during your own deploy
step &mdash; but for a portfolio site that links to source docs in a repo,
the GitHub-rendered fallback is more than enough.
