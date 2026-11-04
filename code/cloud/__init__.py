"""SwingTrader on Claude (4 Nov 2026): the app's own evening update, run once each
weekday evening by a scheduled Claude session, with no Kite and no local PC.

The book, the experiments and the prices they need live as files published with
the claude.ai page (data/db/*); each run rebuilds a SQLite database from them,
runs pipeline.evening_job.tick - the same code and rules as the PC app - and
writes back the files that changed plus the page's views (data/view/*).
"""
