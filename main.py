"""gta6-trophy-rarity-forecast-01 entry: a denominator-aware trophy earn-rate forecaster.

For each trophy of GTA VI (PS5) on Exophase, forecast five quantiles (5/25/50/75/95%) of its earn rate
(percent of tracked players who earned it) at the case's target instant. Scored by mean pinball loss on
the logit scale (lower is better), against house baselines: persistence (last rate, random-walk logit
spread 0.5*sqrt(days)) and a logit-vs-log-time trend fit over the last 24h.

Model (Python 3 standard library only, offline, deterministic):

  rate(t) = earners(t) / players_tracked(t)

The two sides move for different reasons, so they are modelled separately on log-time
x = ln(1 + hours since 2026-11-19T00:00Z):

  * players_tracked: Exophase refreshes counts in jumps; between refreshes the page is identical, so the
    history is first collapsed to unique reads. ln(players) is fit against x with a concave quadratic
    (saturating growth), falling back to a line, then to a launch-surge prior when the page has never
    refreshed in the history (a stale pre-refresh page early after release understates the tracked base
    by a large factor; the first refresh dilutes every rate).
  * earners per trophy: on a stale, never-refreshed page, earners = rate * players / 100 are grown
    with a modest absolute log-time slope (newly counted players at the next refresh have barely
    earned anything, so the median rate falls by earners-growth / players-growth). With a real
    history, the rate itself is modelled as approaching a kind-specific ceiling logistically in
    log-time (ln(rate/(ceiling-rate)) fit on recent unique reads, shrunk toward a kind prior
    approach speed; ceilings and priors from vendored trophy metadata: kind/stage/missable/count
    below, from our own trophy guide data). Saturation is built in, so extrapolation cannot explode
    when the tracked base grows by orders of magnitude.
  * median rate = earners_forecast / players_forecast on the stale branch, the ceiling-logistic
    value otherwise. A trophy already at 100% stays saturated. A trophy nobody has earned yet gets
    a small kind/tier prior level instead of zero.
  * quantiles: normal on the logit scale around the median, with a spread that grows with the horizon,
    is wider for rare trophies (few earners: noisy, refresh-sensitive) and wider when the history holds
    few unique reads (an unseen refresh can move every rate at once - the practice cases' lesson).

Every step is defensive: any surprise falls back to a widened persistence forecast for that trophy (or
the whole case), quantiles are clamped to [0, 100] and forced non-decreasing, so the output is always
the exact shape the grader requires. No randomness, no wall clock, sorted file order: two runs write
byte-identical output.

Run outside the grader with:
    mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
"""

import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.645, -0.674, 0.0, 0.674, 1.645]

# GTA VI released 2026-11-19; log-time is measured from that midnight UTC.
RELEASE_EPOCH = datetime(2026, 11, 19, tzinfo=timezone.utc).timestamp()

# Vendored trophy metadata from our own GTA VI trophy guide data (the house cases carry only
# id/name/description/tier/points/secret). name -> (kind, roadmap stage 1-7, missable, required count).
TROPHY_META = {
    'Vice Got a Price!': ('meta', 7, False, None),
    'Just Like Paradise': ('story', 1, False, None),
    'Paradise Lost': ('story', 1, False, None),
    'T**s Ass Drug Money': ('story', 1, False, None),
    'Vice City, Baby!': ('story', 1, False, None),
    'Top Of The World, Ma': ('story', 1, False, None),
    'You Gotta Know When to Get Off': ('story', 1, False, None),
    'Trust': ('story', 1, False, None),
    'Break the Bank': ('story', 1, False, None),
    'Family Matters': ('story', 1, False, None),
    'DWNFALL': ('story', 1, False, None),
    'Hi-Jacked': ('story', 1, False, None),
    'Snow Storm': ('story', 1, False, None),
    'A Long Road': ('meta', 7, False, None),
    'Cunning Stunts': ('collectibles', 5, False, 250),
    'Master of None': ('activities', 4, False, None),
    'The Shootist': ('activities', 4, False, None),
    'Top Fun': ('activities', 4, False, None),
    'Earn Cash, Not Kudos': ('activities', 4, False, 20),
    'Baited My Tackle': ('activities', 4, False, None),
    'Stick Fiddler': ('activities', 4, False, None),
    'Road to Rehabilitation': ('activities', 3, False, None),
    'This Was No Boat Accident': ('activities', 3, False, None),
    'The Lone Rider': ('activities', 3, False, None),
    'Strangers & Freaks': ('activities', 3, False, 8),
    'Supply & Demand': ('activities', 3, False, None),
    'Spin On This!': ('activities', 3, False, None),
    'Positive Mental Attitude': ('collectibles', 5, False, None),
    'Fair Game': ('activities', 3, False, None),
    "Don't Touch My Deck": ('collectibles', 5, False, 3),
    'You WAiNKer!': ('collectibles', 5, False, 100),
    'One Last Thing': ('activities', 4, False, 50),
    'Hot Together': ('activities', 4, False, None),
    'Retail Therapy': ('activities', 4, False, 3),
    'Whip It Good': ('collectibles', 5, False, 100),
    'Gunslinger': ('activities', 4, False, 5),
    'Catch Me If You Can': ('activities', 2, True, None),
    'Challenging Tank': ('activities', 2, True, None),
    'Sneakthief': ('activities', 4, False, None),
    'Maximum Withdrawal': ('activities', 4, False, 5),
    'A Classic Blitz Play': ('activities', 4, False, None),
    'The Goodtime State': ('collectibles', 5, False, None),
    'Pest Control': ('activities', 4, False, None),
    'In The Beginning…': ('activities', 2, True, None),
    'Capital Offense': ('activities', 2, True, None),
    'Let’s Go Bowling': ('activities', 4, False, None),
    'Personal Best': ('activities', 4, False, None),
    'Actions Speak Louder': ('story', 2, True, None),
    'Bad Luck, I Guess': ('story', 2, True, None),
    'The Truth Is Out There': ('story', 2, True, None),
    'Ten Pin': ('activities', 4, False, None),
    'Dime on the Dollar': ('activities', 6, False, 250000),
    'Treat Yo Self': ('activities', 6, False, 100000),
    'Little Boxes': ('activities', 6, False, None),
    'On the Spot': ('activities', 6, False, 10),
}
# The same table by trophy id (ids are stable in the cases), in case a name ever differs.
META_BY_ID = {i + 1: v for i, v in enumerate(TROPHY_META.values())}


def clamp_pct(x: float) -> float:
    return min(99.99, max(0.01, x))


def logit(pct: float) -> float:
    p = clamp_pct(pct)
    return math.log(p / (100.0 - p))


def sigmoid_pct(lg: float) -> float:
    if lg >= 0:
        return 100.0 / (1.0 + math.exp(-lg))
    e = math.exp(lg)
    return 100.0 * e / (1.0 + e)


def round4(x: float):
    """Round to 4 decimals, half up; whole numbers come back as int (matches the starter's bytes)."""
    scaled = x * 10000
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 10000
    return int(value) if float(value).is_integer() else value


def parse_epoch(s) -> float:
    """ISO-8601 UTC string -> epoch seconds; NaN-safe fallback 0.0."""
    try:
        return datetime.fromisoformat(str(s).replace("Z", "+00:00")).timestamp()
    except Exception:
        return 0.0


def read_epoch(h):
    """Epoch of a history read's "at", or None when the timestamp is missing/unparseable.

    parse_epoch's 0.0 fallback would silently map a corrupt read to log-time 0 and poison every
    fit; callers that build series use this instead and skip such reads outright.
    """
    try:
        return datetime.fromisoformat(str(h.get("at")).replace("Z", "+00:00")).timestamp()
    except Exception:
        return None


def log_time(epoch: float) -> float:
    hours = (epoch - RELEASE_EPOCH) / 3600.0
    return math.log1p(max(0.0, hours))


def normsinv(p: float) -> float:
    """Inverse standard normal CDF (Acklam's approximation), for non-standard levels."""
    a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00, 3.754408661907416e+00]
    plow, phigh = 0.02425, 1 - 0.02425
    if p < plow:
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
               ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
    if p > phigh:
        q = math.sqrt(-2 * math.log(1 - p))
        return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
                 ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
    q = p - 0.5
    r = q * q
    return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / \
           (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1)


def z_for_levels(levels) -> list:
    if list(levels) == LEVELS:
        return list(Z)
    out = []
    for lv in levels:
        try:
            p = float(lv)
        except Exception:
            p = 0.5
        out.append(normsinv(min(0.999, max(0.001, p))))
    return out


def ols_line(xs, ys):
    """Least-squares slope/intercept; returns None when degenerate."""
    n = len(xs)
    if n < 2:
        return None
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx <= 1e-12:
        return None
    slope = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx
    return slope, my - slope * mx


def ols_quad(xs, ys):
    """Least-squares quadratic via normal equations; returns (a, b, c) or None."""
    n = len(xs)
    if n < 4:
        return None
    sx = [sum(x ** k for x in xs) for k in range(5)]
    sy = [sum(y * (x ** k) for x, y in zip(xs, ys)) for k in range(3)]
    m = [[sx[0], sx[1], sx[2], sy[0]], [sx[1], sx[2], sx[3], sy[1]], [sx[2], sx[3], sx[4], sy[2]]]
    for col in range(3):
        piv = max(range(col, 3), key=lambda r: abs(m[r][col]))
        if abs(m[piv][col]) < 1e-12:
            return None
        m[col], m[piv] = m[piv], m[col]
        for r in range(3):
            if r != col:
                f = m[r][col] / m[col][col]
                for cc in range(col, 4):
                    m[r][cc] -= f * m[col][cc]
    return m[0][3] / m[0][0], m[1][3] / m[1][1], m[2][3] / m[2][2]


def unique_reads(history):
    """Collapse consecutive identical reads (same players + rates): the page only changes at a refresh.
    Keeps the last read of each run, so x marks how long that page was current.
    Reads with an unparseable timestamp are dropped: they cannot be placed on the time axis."""
    out = []
    for h in history:
        if read_epoch(h) is None:
            continue
        try:
            key = (h.get("players_tracked"), tuple(h.get("rates") or ()))
        except Exception:
            continue
        if out and out[-1][0] == key:
            out[-1] = (key, h)
        else:
            out.append((key, h))
    return [h for _, h in out]


def players_forecast(uniq, x_last, x_target, checkpoint_epoch):
    """Returns (players_at_target, d ln(players)/dx at x_last, used_fallback)."""
    pts = []
    for h in uniq:
        p = h.get("players_tracked")
        if isinstance(p, (int, float)) and p > 0:
            pts.append((log_time(parse_epoch(h.get("at"))), math.log(float(p))))
    if not pts:
        return 1.0, 0.0, True
    p_last = math.exp(pts[-1][1])
    dx = max(0.0, x_target - x_last)
    if len(pts) >= 4:
        xs = [q[0] for q in pts]
        ys = [q[1] for q in pts]
        quad = ols_quad(xs, ys)
        if quad is not None and quad[2] <= 0:
            a, b, c = quad
            slope = min(3.0, max(0.0, b + 2 * c * x_last))
            pred = math.exp(a + b * x_target + c * x_target * x_target)
            return max(p_last, pred), slope, False
        line = ols_line(xs, ys)
        if line is not None:
            slope = min(3.0, max(0.0, line[0]))
            return p_last * math.exp(slope * dx), slope, False
    if len(pts) >= 2:
        line = ols_line([q[0] for q in pts], [q[1] for q in pts])
        if line is not None:
            slope = min(3.0, max(0.0, line[0]))
            return p_last * math.exp(slope * dx), slope, False
    # One unique page only: it has never refreshed inside the history. Early after release the tracked
    # base is a stale pre-refresh count and the first refresh multiplies it (launch-night surge);
    # later, an unrefreshed page still tends to jump at the next refresh, just less violently.
    hours = (checkpoint_epoch - RELEASE_EPOCH) / 3600.0
    if hours < 36:
        growth = 4.0
    elif hours < 96:
        growth = 1.8
    else:
        growth = 1.25
    slope = math.log(growth) / max(dx, 0.25)
    return p_last * growth, min(3.0, slope), True


def prior_delta(kind, stage, missable, count, tier):
    """Earners' log-slope prior relative to the players slope (per unit log-time)."""
    if kind == "story":
        d = 0.35 if stage <= 1 else 0.50  # late-story earners accelerate as the cohort reaches them
    elif kind == "collectibles":
        d = -0.05
    elif kind == "meta":
        d = -0.25
    else:  # activities and anything unknown
        d = 0.10
    if missable:
        d -= 0.20
    if count is not None:
        if count >= 1000:
            d -= 0.30  # money grinds ($100k/$250k) crawl in week one
        elif count >= 50:
            d -= 0.15
    if tier == "gold":
        d -= 0.10
    return d


def prior_level(kind, tier, days):
    """Median rate (%) for a trophy nobody has earned yet, by kind/tier, growing mildly with horizon."""
    base = {"meta": 0.04, "collectibles": 0.20, "story": 0.40}.get(kind, 0.25)
    if tier == "platinum":
        base = 0.04
    elif tier == "gold":
        base = min(base, 0.10)
    elif tier == "silver":
        base = min(base, 0.18)
    return base * (1.0 + 0.15 * min(days, 7.0))


def ceiling_prior(kind, stage, missable, count, tier, rate_last):
    """Plausible week-one ceiling (%) for a trophy's earn rate, by kind and friends.

    Rates approach a kind-specific ceiling as the launch cohort progresses: early story trophies
    saturate high, late story lower, activities/collectibles stay rare, meta/platinum near zero.
    The observed rate always lifts the ceiling above itself.
    """
    if kind == "story":
        cap = 90.0 if stage <= 1 else 35.0
    elif kind == "collectibles":
        cap = 4.0
    elif kind == "meta":
        cap = 0.6
    else:
        cap = 12.0
    if missable:
        cap *= 0.7
    if count is not None:
        if count >= 1000:
            cap *= 0.5
        elif count >= 50:
            cap *= 0.65
    if tier == "gold":
        cap *= 0.6
    elif tier == "silver":
        cap *= 0.85
    if rate_last is not None:
        cap = max(cap, rate_last * 1.2, rate_last + 0.1)
    return min(99.5, max(0.05, cap))


def prior_u_slope(kind, stage, missable, count, tier):
    """Prior slope of ln(rate/(ceiling-rate)) per unit log-time (approach speed toward the ceiling)."""
    if kind == "story":
        s = 0.65 if stage <= 1 else 0.9
    elif kind == "collectibles":
        s = 0.55
    elif kind == "meta":
        s = 0.5
    else:
        s = 0.6
    if missable:
        s -= 0.15
    if count is not None and count >= 50:
        s -= 0.15
    if tier == "gold":
        s -= 0.10
    return s


def make_row(tid, median_pct, sigma, zs):
    center = logit(median_pct)
    qs = [round4(sigmoid_pct(center + z * sigma)) for z in zs]
    qs = [min(100.0, max(0.0, float(q))) for q in qs]
    for i in range(1, len(qs)):  # enforce non-decreasing after rounding
        if qs[i] < qs[i - 1]:
            qs[i] = qs[i - 1]
    return {"id": tid, "q": [int(q) if float(q).is_integer() else q for q in qs]}


def persistence_row(tid, rate, days, zs, widen=1.0):
    sigma = max(0.45, 0.5 * math.sqrt(max(days, 0.0))) * widen
    return make_row(tid, rate if isinstance(rate, (int, float)) else 0.0, sigma, zs)


def forecast_case(case: dict) -> dict:
    trophies = case.get("trophies") or []
    history = case.get("history") or []
    zs = z_for_levels(case.get("levels") or LEVELS)
    checkpoint = parse_epoch(case.get("at"))
    target = parse_epoch(case.get("target"))
    horizon_min = case.get("horizon_minutes")
    if not isinstance(horizon_min, (int, float)) or horizon_min <= 0:
        horizon_min = max(0.0, (target - checkpoint) / 60.0) if target > checkpoint else 90.0
    days = horizon_min / 1440.0
    if target <= checkpoint:
        target = checkpoint + horizon_min * 60.0
    x_target = log_time(target)

    last_rates = []
    if history:
        last_rates = history[-1].get("rates") or []

    def last_rate(i):
        # Latest non-null, non-negative rate for trophy i, scanning newest first.
        for h in reversed(history):
            rates = h.get("rates") or []
            if i < len(rates) and isinstance(rates[i], (int, float)) and rates[i] >= 0:
                return float(rates[i])
        return None

    if not history:
        rows = []
        for t in trophies:
            kind, stage, missable, count = TROPHY_META.get(t.get("name"), META_BY_ID.get(t.get("id"), ("activities", 3, False, None)))
            rows.append(make_row(t.get("id"), prior_level(kind, t.get("tier"), days), 1.8, zs))
        return {"trophies": rows}

    uniq = unique_reads(history)
    x_last = log_time(parse_epoch(uniq[-1].get("at"))) if uniq else log_time(checkpoint)
    if x_last > x_target:
        x_last = x_target
    dx = max(0.0, x_target - x_last)
    players_pred, players_slope, players_fallback = players_forecast(uniq, x_last, x_target, checkpoint)
    n_unique = len(uniq)
    stale = 0.85 if n_unique <= 2 else (0.35 if n_unique <= 5 else 0.15)
    players_unc = 0.25 if players_fallback else 0.10
    horizon_sigma = 0.5 * math.sqrt(max(days, 0.0))

    rows = []
    for i, t in enumerate(trophies):
        tid = t.get("id")
        rate_last = last_rate(i)
        try:
            kind, stage, missable, count = TROPHY_META.get(
                t.get("name"), META_BY_ID.get(tid, ("activities", 3, False, None)))
            if rate_last is not None and rate_last >= 99.99:
                # Saturated: every tracked player has it; it stays at the ceiling.
                rows.append(make_row(tid, 100.0, 0.9, zs))
                continue
            # Series over unique reads: earners (for the stale-page branch) and raw rates (fitted branch).
            xs, ys, rs = [], [], []
            e_last = 0.0
            for h in uniq:
                rates = h.get("rates") or []
                p = h.get("players_tracked")
                # Negative rates (hence negative earners) are corrupt reads: treat as missing.
                if i < len(rates) and isinstance(rates[i], (int, float)) and rates[i] >= 0 \
                        and isinstance(p, (int, float)) and p > 0:
                    e = float(rates[i]) / 100.0 * float(p)
                    xs.append(log_time(parse_epoch(h.get("at"))))
                    ys.append(math.log(e + 0.5))
                    rs.append(float(rates[i]))
                    e_last = e
            if rate_last is None or (e_last <= 0.0 and (rate_last or 0.0) <= 0.0):
                median = prior_level(kind, t.get("tier"), days)
                sigma = min(2.3, math.sqrt(horizon_sigma ** 2 + 1.1 ** 2 + stale ** 2))
                rows.append(make_row(tid, median, max(0.45, sigma), zs))
                continue
            if players_fallback:
                # Stale page (never refreshed in the history): the players jump at the next refresh is
                # a denominator correction - newly counted players have barely earned anything - so
                # earners get a modest absolute slope instead of inheriting the surge, and the median
                # rate falls by roughly earners-growth / players-growth.
                prior_slope = 0.55 + prior_delta(kind, stage, missable, count, t.get("tier"))
                slope = min(4.0, max(-0.5, prior_slope))
                e_pred = max(0.0, math.exp(math.log(e_last + 0.5) + slope * dx) - 0.5)
                median = 100.0 * e_pred / players_pred if players_pred > 0 else (rate_last or 0.0)
            else:
                # Fitted branch: the rate approaches a kind-specific ceiling logistically in log-time.
                # u = ln(rate / (ceiling - rate)) is fit on recent unique reads and shrunk toward a
                # kind prior approach speed; saturation is built in, so extrapolation cannot explode
                # the way a raw earners trend does when the tracked base grows orders of magnitude.
                ceiling = ceiling_prior(kind, stage, missable, count, t.get("tier"), rate_last)
                us = []
                for x, r in zip(xs, rs):
                    rr = min(max(r, 0.005), ceiling * 0.999)
                    us.append((x, math.log(rr / (ceiling - rr))))
                u_last = us[-1][1] if us else math.log(0.01 / ceiling)
                win = [(x, u) for x, u in us if x >= x_last - 1.6]
                if len(win) < 3:
                    win = us
                slope_u = prior_u_slope(kind, stage, missable, count, t.get("tier"))
                if len(win) >= 2:
                    line = ols_line([w[0] for w in win], [w[1] for w in win])
                    if line is not None:
                        span = max(w[0] for w in win) - min(w[0] for w in win)
                        wgt = min(0.8, len(win) / 8.0) * min(1.0, span / 1.2)
                        slope_u = wgt * line[0] + (1.0 - wgt) * slope_u
                slope_u = min(3.0, max(-1.5, slope_u))
                u_pred = u_last + slope_u * dx
                median = ceiling / (1.0 + math.exp(-u_pred)) if u_pred >= 0 else \
                    ceiling * math.exp(u_pred) / (1.0 + math.exp(u_pred))
                # Shrink the trend move toward persistence: a late surge of newly tracked players
                # (who have barely earned anything) dilutes rates in a way no trend can foresee,
                # so only part of the extrapolated logit move is taken.
                lg_move = logit(median) - logit(rate_last)
                median = sigmoid_pct(logit(rate_last) + 0.7 * lg_move)
            median = min(99.9, max(0.01, median))
            rarity = 0.9 / math.sqrt(e_last + 2.0)
            sigma = math.sqrt(horizon_sigma ** 2 + rarity ** 2 + stale ** 2 + players_unc ** 2)
            sigma = min(2.3, max(0.45, sigma))
            rows.append(make_row(tid, median, sigma, zs))
        except Exception:
            rows.append(persistence_row(tid, rate_last, days, zs, widen=2.0))
    return {"trophies": rows}


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        try:
            case = json.loads(path.read_text(encoding="utf-8"))
            forecast = forecast_case(case)
        except Exception:
            # Last-resort valid output: persistence on whatever rates can be salvaged.
            try:
                case = json.loads(path.read_text(encoding="utf-8"))
                trophies = case.get("trophies") or []
                history = case.get("history") or []
                last = (history[-1].get("rates") or []) if history else []
                days = float(case.get("horizon_minutes") or 90) / 1440.0
                forecast = {"trophies": [
                    persistence_row(t.get("id"), last[i] if i < len(last) else 0.0, days, Z, widen=2.0)
                    for i, t in enumerate(trophies)]}
            except Exception:
                continue
        (OUTPUT_DIR / f"{path.stem}.out").write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
