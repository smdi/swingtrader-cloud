"""One evening of SwingTrader on Claude (cloud/__init__.py).

    cd <folder>/code && python -m cloud.run --data <folder> --out <out>
    cd <app>/code && python -m cloud.run --data <site> --in-place      (GitHub Actions)

<folder> holds the page's published files (code/..., data/db/..., data/manifest.json).
The run rebuilds the database, runs the app's evening update (pipeline.evening_job.tick:
NSE's daily file, corporate actions, the 25% stop, red-flag announcements, the
month-end decision and its trades at the next official open, the experiments),
applies the retention policy and writes into <out> the files to publish, listed in
<out>/publish.json: {"changed": [published paths], "removed": [published paths]}.
Kite is never used: trades are made at NSE's official open by the evening update.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import traceback
from datetime import datetime
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent.parent


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="folder with data/db/... and data/manifest.json")
    ap.add_argument("--out", help="folder for the files to publish")
    ap.add_argument("--in-place", action="store_true",
                    help="write the changed files into --data itself and delete the removed ones (GitHub Actions)")
    ap.add_argument("--now", help="IST time 'YYYY-MM-DD HH:MM' (default: now)")
    ap.add_argument("--no-update", action="store_true", help="only rebuild and export (no NSE)")
    args = ap.parse_args(argv)
    sys.path.insert(0, str(APP_DIR))
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    from cloud import datafiles, views
    from core.timeutil import IST
    from pipeline import evening_job
    from pipeline.nse_eod import NseEodClient
    from store import db as store

    log = []

    def say(msg):
        log.append(f"{datetime.now(IST):%H:%M:%S} {msg}")
        print(msg, flush=True)

    db_path = store.default_path()
    db_path.parent.mkdir(parents=True, exist_ok=True)
    got = datafiles.restore(Path(args.data), db_path, APP_DIR)
    say(f"restored: {sum(got['rows'].values())} rows in {len(got['rows'])} tables")
    now = datetime.strptime(args.now, "%Y-%m-%d %H:%M") if args.now else datetime.now(IST).replace(tzinfo=None)
    conn = store.connect(db_path)
    status = "ok"
    if not args.no_update:
        client = NseEodClient(APP_DIR / "data" / "nse_research")
        try:
            done = evening_job.tick(conn, client, now, kite=None, news=True, log=say)
            say(f"evening update: {done}")
        except Exception as exc:                      # the page still shows what is there, and the error
            status = "error"
            say(f"evening update failed: {exc}")
            traceback.print_exc()
        finally:
            try:
                client.close()
            except Exception:
                pass
    removed = datafiles.apply_retention(conn, now.date())
    if any(removed.values()):
        say(f"retention: {removed}")
    datafiles.trim_delivery_cache(APP_DIR / datafiles.DELIVERY_CACHE, datafiles.keep_from(conn, now.date())["delivery"])
    if not args.out and not args.in_place:
        ap.error("give --out or --in-place")
    out = Path(args.data) if args.in_place else Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    res = datafiles.export(conn, out, APP_DIR, got["manifest"])
    v = views.write(views.build(conn, now.date(), now, log + [f"status: {status}"]), out)
    conn.close()
    if args.in_place:                                  # retention dropped a month: its file goes too
        for p in res["removed"]:
            (out / p).unlink(missing_ok=True)
    publish = {"status": status, "changed": res["changed"] + v + ["data/manifest.json"], "removed": res["removed"]}
    (out / "publish.json").write_text(json.dumps(publish, indent=1), encoding="utf-8")
    say(f"to publish: {len(publish['changed'])} files changed, {len(publish['removed'])} removed")
    return 0 if status == "ok" else 2


if __name__ == "__main__":
    sys.exit(main())
