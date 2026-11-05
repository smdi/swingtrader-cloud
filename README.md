# SwingTrader (cloud)

The evening update of the SwingTrader paper portfolio, run by GitHub Actions every weekday
(19:37 and 22:23 IST, plus a 06:11 IST catch-up). Paper trading only: no broker, no orders, no passwords.

- `code/` - the app's own code (rules unchanged) and `code/cloud/` (rebuild, run, export).
- `.github/workflows/evening.yml` - the timer and the **Run workflow** button.
- branch `data` - the page (`index.html`) and its data, latest copy only.

To run it by hand: Actions → Evening update → Run workflow.

The page: https://smdi.github.io/swingtrader-cloud/ (GitHub Pages, from the `data` branch).
