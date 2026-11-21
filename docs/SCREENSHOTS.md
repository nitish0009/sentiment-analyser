# Optional Screenshot Capture Guide

This guide explains how to capture interface images for a report or presentation
from your own running copy of the app. No screenshots are bundled with the
project.

---

## Prerequisites

1. The project is set up and the model is trained:

   ```bash
   python scripts/download_dataset.py
   python src/train.py
   ```

2. The app is running:

   ```bash
   python -m streamlit run streamlit_app.py
   ```

3. Open <http://127.0.0.1:8501> and wait for the sidebar badge **Model ready**.

Chrome at a **1440 × 1000** viewport works well. Keep the sidebar visible so the
project name and model status appear in each image.

---

## Test inputs

**Positive**

> This movie was absolutely wonderful. The acting was brilliant, the story was
> gripping, and the ending left me smiling for hours.

**Negative**

> This was a dull, badly acted mess. The plot made no sense and I was bored from
> start to finish.

**Low confidence**

> I liked parts of it and disliked other parts.

**Sample cards**

The six clear examples and three tricky cases are defined in `SAMPLES` and
`TRICKY_SAMPLES` at the top of `streamlit_app.py`. They load and run automatically when the
corresponding button is pressed.

---

## Manual capture steps

Create a local output folder first:

```bash
mkdir -p my_screenshots
```

1. **Home** — open `/` and capture the viewport.
2. **Pipeline** — scroll down to the heading “How the pipeline works” and capture.
3. **Analyzer positive** — open `/analyzer`, paste the positive text, press
   **Analyze sentiment**, wait for the card, capture.
4. **Analyzer negative** — replace the text, analyze again, capture.
5. **Low confidence** — paste the mixed text, analyze, capture the warning.
6. **Performance metrics** — open `/performance`, capture the top.
7. **Comparison** — scroll to “Naive Bayes vs Logistic Regression” and capture.
8. **Confusion** — scroll to “Confusion matrices” and capture.
9. **Features** — scroll to “What the model learned” and capture.
10. **Method** — scroll to “Dataset and method” and capture.
11. **Samples** — open `/samples` and capture.
12. **Autoload** — press an **Analyze this review** button, wait for the analyzer,
    capture.

For sections below the fold, scroll the Streamlit main container, not just the
browser window. In Chrome DevTools:

```js
const main = document.querySelector('[data-testid="stMain"]');
main.scrollTop = 900;   // adjust as needed
```

---

## Playwright capture

Playwright is optional but makes the process repeatable. Install it in the same
environment:

```bash
pip install playwright
python -m playwright install chromium
```

The following script saves two example captures to `my_screenshots/`. It uses the
Chrome channel because the app was tested in Chrome.

```python
from pathlib import Path

from playwright.sync_api import sync_playwright

BASE = "http://127.0.0.1:8501"
OUT = Path("my_screenshots")
OUT.mkdir(exist_ok=True)

POSITIVE = (
    "This movie was absolutely wonderful. The acting was brilliant, the story "
    "was gripping, and the ending left me smiling for hours."
)

with sync_playwright() as p:
    browser = p.chromium.launch(channel="chrome", headless=True)
    page = browser.new_page(viewport={"width": 1440, "height": 1000})
    page.goto(BASE, wait_until="networkidle")
    page.wait_for_timeout(3000)
    page.screenshot(path=str(OUT / "home.png"))

    page.get_by_role("link", name="Sentiment Analyzer").click()
    page.wait_for_timeout(2500)
    page.locator("textarea").first.fill(POSITIVE)
    page.get_by_role("button", name="Analyze sentiment").click()
    page.wait_for_timeout(3000)
    page.screenshot(path=str(OUT / "analyzer_positive.png"))

    browser.close()
```

### Scrolling to a section inside Streamlit

Streamlit keeps the page inside a scrollable main container. `scrollIntoView` on
the browser window may not move it. Use JavaScript on the container, or find a
heading and adjust its `scrollTop`:

```python
page.evaluate(
    """([text]) => {
        const main = document.querySelector('[data-testid="stMain"]');
        const nodes = [...main.querySelectorAll('h1,h2,h3,p,div,span')];
        const el = nodes.find(
            e => e.children.length === 0 && e.textContent.trim() === text
        );
        if (!el) return false;
        main.scrollTop += el.getBoundingClientRect().top - 30;
        return true;
    }""",
    ["Confusion matrices"],
)
page.wait_for_timeout(700)
page.screenshot(path=str(OUT / "performance_confusion.png"))
```

---

## Composition tips

- Use a **1440 × 1000** viewport so the two-column analyzer layout fits without
  wrapping.
- Keep browser zoom at 100%.
- Wait for the spinner to finish before capturing. A result card that says
  “Analyzing…” is not useful.
- Include the sidebar in every shot; it shows the page name and model status.
- Do not crop the probability bars or the low-confidence warning.
- Use the same text inputs listed above so the documented numbers match.
- Save as PNG for sharp text.

---

## Verifying a fresh capture

After capturing, check:

- the **Model ready** badge is visible;
- the analyzer card shows a class, a percentage and both probability bars;
- the low-confidence image actually shows the warning;
- the performance page shows **89.65%** for Logistic Regression;
- the comparison chart shows both models and the legend below the axis;
- no browser developer console errors are visible;
- the image width is 1440 px.

A quick image check:

```bash
file my_screenshots/*.png
```

---

## Troubleshooting

| Problem | Fix |
|---|---|
| App shows “Model not trained” | Run `python src/train.py`. |
| Results are stale after code changes | Restart Streamlit or press **Reload saved model**. |
| Charts are missing | Re-run training; `reports/` should contain the PNGs. |
| Browser screenshots are blank | Wait longer for Streamlit's websocket to finish the first render. |
| Page does not scroll with `scrollIntoView` | Scroll `[data-testid="stMain"]` via JavaScript. |
| Port 8501 is busy | Stop the old server or use `--server.port 8502`. |

---

## Privacy

Only use synthetic demo reviews in screenshots. Do not capture personal data,
credentials or API keys.
