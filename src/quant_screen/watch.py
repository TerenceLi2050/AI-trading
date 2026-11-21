from datetime import datetime, time, timedelta
import time as clock

import pandas as pd

from .sessions import CST, load_sessions


def next_check(now, status, schedule, settled_after):
    if status:
        return now + timedelta(minutes=30)
    today = pd.Timestamp(now.date())
    dates = pd.date_range(today, periods=15)
    for day in dates:
        when = datetime.combine(day.date(), time.fromisoformat(settled_after), CST)
        if when > now and (schedule is None or day not in schedule.index or schedule.at[day, "is_session"] == 1):
            return when
    return datetime.combine((today + pd.Timedelta(days=1)).date(), time.fromisoformat(settled_after), CST)


def watch_daily(run, calendar_path, settled_after):
    try:
        while True:
            try:
                status = run()
                calendar = load_sessions(calendar_path)
            except (OSError, ValueError, RuntimeError) as error:
                print(f"Daily run failed: {error}", flush=True)
                status, calendar = 2, None
            now = datetime.now(CST)
            when = next_check(now, status, calendar, settled_after)
            print(f"Next check {when.isoformat(timespec='minutes')}", flush=True)
            clock.sleep(max(1, (when - now).total_seconds()))
    except KeyboardInterrupt:
        return 0
