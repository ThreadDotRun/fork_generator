# /home/tdrsvr/Desktop/slop/streams/datafeeder.py
# Datafeeder: event stream -> validated events -> one raw K=8 vector per tick.
#
# Free live source: Wikimedia EventStreams, recentchange (public, no key, SSE):
#   https://stream.wikimedia.org/v2/stream/recentchange
# Offline sources with the same event schema: Replay (JSONL) and Synthetic (regimes,
# injected faults). The live client is NOT exercised in the offline sandbox.
#
# Rejection boundary (the harness's verifier):
#   normalize()   drops malformed events (bad type, missing/NaN timestamp or length)
#   TickAggregator drops late events (bucket already closed) and holds the previous
#                 channels when a summary is non-finite. Every rejection is counted.
# Nothing is guessed. A quiet tick emits rate 0 and holds the level channels.
#
# Live caveat: ticks close when the next event crosses a boundary, so a silent live
# stream stalls the clock. A heartbeat is needed before live use (not built yet).

import json
import math
import random
import sys
import time
import urllib.error
import urllib.request

import numpy as np

K = 8
WIKI_URL = "https://stream.wikimedia.org/v2/stream/recentchange"
EVENT_TYPES = ("edit", "new", "log", "categorize")


def normalize(obj):
    """Schema check. Returns a compact event dict, or None if malformed."""
    try:
        t = float(obj["timestamp"])
        typ = obj["type"]
        if typ not in EVENT_TYPES:
            return None
        ln = obj.get("length") or {}
        old, new = ln.get("old"), ln.get("new")
        dlen = abs(float(new) - float(old)) if (old is not None and new is not None) else 0.0
        if not (math.isfinite(t) and math.isfinite(dlen)):
            return None
        return dict(t=t, type=typ, bot=bool(obj.get("bot", False)),
                    wiki=str(obj["wiki"]), dlen=dlen)
    except (KeyError, TypeError, ValueError):
        return None


class TickAggregator:
    """Buckets validated events into ticks of tick_s seconds of event time.
    Channels per tick:
      0 log1p(events/s)        1 bot fraction         2 new-page fraction
      3 log-event fraction     4 categorize fraction  5 log1p(mean |dlen|)
      6 log1p(distinct wikis)  7 large-change fraction (|dlen| > 1000)"""

    def __init__(self, tick_s=1.0):
        self.tick_s = float(tick_s)
        self.cur = None
        self.buf = []
        self.last = np.zeros(K)
        self.rejected = 0      # late / out-of-order events
        self.held = 0          # ticks whose summary was non-finite
        self.ticks = 0

    def _summarize(self, evs):
        n = len(evs)
        if n == 0:
            v = self.last.copy()
            v[0] = 0.0
            return v
        typ = [e["type"] for e in evs]
        v = np.array([
            math.log1p(n / self.tick_s),
            float(np.mean([e["bot"] for e in evs])),
            typ.count("new") / n,
            typ.count("log") / n,
            typ.count("categorize") / n,
            math.log1p(float(np.mean([e["dlen"] for e in evs]))),
            math.log1p(len({e["wiki"] for e in evs})),
            float(np.mean([e["dlen"] > 1000 for e in evs])),
        ], float)
        if not np.all(np.isfinite(v)):
            self.held += 1
            v = self.last.copy()
            v[0] = math.log1p(n / self.tick_s)
        return v

    def _close(self):
        v = self._summarize(self.buf)
        self.buf = []
        self.last = v.copy()
        self.ticks += 1
        return v

    def push(self, ev):
        idx = int(ev["t"] // self.tick_s)
        if self.cur is None:
            self.cur = idx
        if idx < self.cur:
            self.rejected += 1
            return []
        out = []
        while idx > self.cur:
            out.append(self._close())
            self.cur += 1
        self.buf.append(ev)
        return out

    def flush(self):
        return [self._close()] if self.buf else []


class Feeder:
    """Source -> normalize -> aggregate. Iterate to get raw K-vectors, one per tick."""

    def __init__(self, source, tick_s=1.0):
        self.source = source
        self.agg = TickAggregator(tick_s)
        self.bad = 0           # malformed events

    def __iter__(self):
        for obj in self.source:
            ev = normalize(obj)
            if ev is None:
                self.bad += 1
                continue
            yield from self.agg.push(ev)
        yield from self.agg.flush()

    def counts(self):
        return dict(malformed=self.bad, late=self.agg.rejected,
                    held=self.agg.held, ticks=self.agg.ticks)


# ---------------------------------------------------------------- sources
class Replay:
    """JSONL file (one event per line) or an iterable of dicts. Bad lines become None."""

    def __init__(self, path=None, items=None):
        self.path, self.items = path, items

    def __iter__(self):
        if self.items is not None:
            yield from self.items
            return
        with open(self.path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    yield None


class WikimediaSSE:
    """Live Wikimedia recentchange over Server-Sent Events, with reconnect and backoff.
    Untested in the offline sandbox. Needs network access on the host."""

    def __init__(self, url=WIKI_URL, timeout=30, max_backoff=60):
        self.url, self.timeout, self.max_backoff = url, timeout, max_backoff

    def __iter__(self):
        delay = 1.0
        while True:
            try:
                req = urllib.request.Request(self.url, headers={
                    "Accept": "text/event-stream",
                    "User-Agent": "slop-datafeeder/0.1 (research)"})
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    delay = 1.0
                    for raw in resp:
                        line = raw.decode("utf-8", "replace").strip()
                        if line.startswith("data:"):
                            try:
                                yield json.loads(line[5:].strip())
                            except json.JSONDecodeError:
                                yield None
            except (OSError, urllib.error.URLError) as e:
                print(f"[sse] {e}; retry in {delay:.0f}s", file=sys.stderr)
                time.sleep(delay)
                delay = min(delay * 2, self.max_backoff)


class Synthetic:
    """Offline firehose with regimes. Each regime is a tuple:
       (seconds, events/s, bot_frac, new_frac, log_frac, cat_frac,
        median |dlen|, big_change_frac, n_wikis).
    p_malformed injects bad events (bad type, null timestamp, NaN length, null record);
    p_late injects events stamped 30 s in the past (exercises the late-event rejection)."""

    def __init__(self, regimes, seed=0, p_malformed=0.0, p_late=0.0):
        self.regimes, self.seed = regimes, seed
        self.p_malformed, self.p_late = p_malformed, p_late

    def __iter__(self):
        rng = random.Random(self.seed)
        t = 0.0
        for (dur, rate, bot_p, new_p, log_p, cat_p, mdl, big_p, nw) in self.regimes:
            end = t + dur
            while True:
                t += rng.expovariate(rate)
                if t >= end:
                    break
                u = rng.random()
                if u < new_p:
                    typ = "new"
                elif u < new_p + log_p:
                    typ = "log"
                elif u < new_p + log_p + cat_p:
                    typ = "categorize"
                else:
                    typ = "edit"
                dlen = rng.lognormvariate(math.log(mdl), 1.0)
                if rng.random() < big_p:
                    dlen += 5000.0
                obj = dict(timestamp=t, type=typ, bot=rng.random() < bot_p,
                           wiki=f"w{rng.randrange(nw)}",
                           length=dict(old=1000, new=1000 + dlen))
                u2 = rng.random()
                if u2 < self.p_malformed:
                    obj = rng.choice([dict(obj, timestamp=None), dict(obj, type="bogus"),
                                      None, dict(obj, length=dict(old=1000, new=float("nan")))])
                elif u2 < self.p_malformed + self.p_late:
                    obj = dict(obj, timestamp=t - 30.0)
                yield obj
            t = end
