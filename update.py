#!/usr/bin/env python3
"""
Top Flight 18 Elite — Where Are They Now · updater v2

Runs every morning on GitHub Actions. Reads players.json + the previous data.json,
asks Claude (web fetch + web search) for what changed, writes data.json for the app.

Cadence
  Mon, Thu, Sat stats, team record, results (with box scores and her line)
  Mon + Thu     the written piece: Monday recap / Thursday preview
  Mon only      full-season schedules (times in Central), standings, program socials, news + coach items
  as needed     profile facts (photo, jersey, class, height, hometown)

Env:  ANTHROPIC_API_KEY (required) · TRACKER_MODEL (Haiku, daily stat pulls) · TRACKER_STRONG_MODEL (Sonnet: schedules, standings, news, juco sites) · TRACKER_WRITER_MODEL (Sonnet: recap/preview)
      TRACKER_FULL=1 forces the Mon/Thu work to run today (the first run does this automatically)
"""

import base64, json, os, re, sys, time, threading, urllib.request, urllib.error, urllib.parse
from html.parser import HTMLParser
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

API_URL = "https://api.anthropic.com/v1/messages"
API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
MODEL = os.environ.get("TRACKER_MODEL", "claude-sonnet-5")      # research calls (Haiku proved too sloppy: merged rows, missed box scores)
WRITER_MODEL = os.environ.get("TRACKER_WRITER_MODEL", "claude-sonnet-5")   # Monday recap / Thursday preview only
HERE = os.path.dirname(os.path.abspath(__file__))
PLAYERS_FILE, DATA_FILE = os.path.join(HERE, "players.json"), os.path.join(HERE, "data.json")

DRY_RUN = os.environ.get("TRACKER_DRY_RUN", "") in ("1", "true", "True")   # fetch + parse everything live, call no model, write a report, leave data.json alone
TOKEN_BUDGET = int(os.environ.get("TRACKER_TOKEN_BUDGET", "900000"))   # input tokens per run; optional work stops at 70%, everything at 100%
WORKERS = int(os.environ.get("TRACKER_WORKERS", "4"))
CALL_TIMEOUT = 240
USAGE = {"in": 0, "out": 0, "calls": 0}
LOCK = threading.Lock()
STAT_KEYS = ["mp", "sp", "k", "e", "ta", "a", "bhe", "sa", "se", "srv", "dig", "re", "bs", "ba", "be"]
MILESTONES = {"k": ("kill", [1, 25, 50, 100, 150, 200, 300]), "a": ("assist", [1, 50, 100, 200, 300, 500, 750]),
              "dig": ("dig", [1, 50, 100, 150, 200, 300, 400]), "sa": ("ace", [1, 10, 25, 50]),
              "bs": ("solo block", [1]), "ba": ("block assist", [1])}

STRONG_MODEL = os.environ.get("TRACKER_STRONG_MODEL", WRITER_MODEL)   # schedules/standings, news, and the juco girls
SEARCH = {"type": "web_search_20250305", "name": "web_search", "max_uses": 3}
SYSTEM = ("You maintain a small, family-friendly tracker of college volleyball players for their parents. "
          "You read the page text and documents you are given (and web_search result snippets when told to). Return ONLY valid JSON "
          "matching the requested structure — no prose, no code fences. Accuracy beats completeness: never invent a "
          "number, score, date, or link; use null when something is not in the material. Dates are ISO YYYY-MM-DD.")


def log(m): print(f"[{datetime.now().strftime('%H:%M:%S')}] {m}", flush=True)


def spent(): return USAGE["in"] / TOKEN_BUDGET


class BudgetExceeded(RuntimeError): pass


# ---------------------------------------------------------------- API
def call_claude(prompt, tools=None, max_tokens=5000, system=SYSTEM, model=None, docs=None, exempt=False):
    """One retry on any failure (network, HTTP, bad JSON). Refuses to start once the run budget is spent (unless exempt)."""
    if DRY_RUN:
        return {"title": "dry run", "body": [], "spotlight": None, "items": [], "schedule": [], "standing": None, "standings_url": None,
                "team_record": {}, "stats": {}, "results": [], "blurb": ""}
    if spent() >= 1.0 and not exempt: raise BudgetExceeded(f"run token budget spent ({USAGE['in']:,} input tokens)")
    for attempt in (1, 2):
        try:
            return _call(prompt, tools, max_tokens, system, model, docs)
        except Exception as e:
            if attempt == 2: raise
            log(f"    retrying after: {str(e)[:120]}"); time.sleep(6)


def _call(prompt, tools, max_tokens, system, model, docs=None):
    content = [{"type": "document", "source": {"type": "base64", "media_type": "application/pdf", "data": base64.b64encode(b).decode()}, "title": t} for t, b in (docs or [])]
    content.append({"type": "text", "text": prompt})
    convo = [{"role": "user", "content": content}]; asked_again = False
    for _ in range(6):
        body = {"model": model or MODEL, "max_tokens": max_tokens, "system": system, "messages": convo}
        if tools: body["tools"] = tools
        req = urllib.request.Request(API_URL, data=json.dumps(body).encode(), method="POST", headers={
            "x-api-key": API_KEY, "anthropic-version": "2023-06-01", "content-type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=CALL_TIMEOUT) as r:
                data = json.loads(r.read())
        except urllib.error.HTTPError as e:
            raise RuntimeError(f"API HTTP {e.code}: {e.read().decode('utf-8', 'replace')[:600]}") from None
        if data.get("stop_reason") == "pause_turn":
            convo.append({"role": "assistant", "content": data["content"]}); continue
        text = "".join(b.get("text", "") for b in data.get("content", []) if b.get("type") == "text")
        u = data.get("usage", {}); st = u.get("server_tool_use", {})
        with LOCK:
            USAGE["in"] += u.get("input_tokens", 0) or 0; USAGE["out"] += u.get("output_tokens", 0) or 0; USAGE["calls"] += 1
        log(f"    tokens {u.get('input_tokens')}/{u.get('output_tokens')} fetch={st.get('web_fetch_requests', 0)} search={st.get('web_search_requests', 0)} · run {spent():.0%} of budget")
        try:
            return parse_json(text)
        except ValueError:
            if asked_again: raise
            asked_again = True
            convo.append({"role": "assistant", "content": data["content"]})
            convo.append({"role": "user", "content": "Stop researching. Using only what you have already found, return the requested JSON object now — "
                                                     "nothing before or after it, null for anything you could not find, empty lists where nothing applies."})
    raise RuntimeError("too many continuations")


def parse_json(text):
    """Return the last complete JSON object in the text (models sometimes narrate around it)."""
    text = text.replace("```json", "").replace("```", "")
    dec, found, i = json.JSONDecoder(), None, text.find("{")
    while i != -1:
        try:
            obj, end = dec.raw_decode(text, i)
            if isinstance(obj, dict): found = obj
            i = text.find("{", end)
        except ValueError:
            i = text.find("{", i + 1)
    if found is None: raise ValueError("no JSON in response: " + text[:200])
    return found


def pages(p):
    note = ("\n" + p["fetch_note"]) if p.get("fetch_note") else ""
    return (f"Official pages:\n- Roster/bio: {p['roster_url']}\n- Schedule: {p['schedule_url']}\n- Stats: {p['stats_url']}"
            f"{note}\nIf a page fails or has no data, fall back to web_search (\"{p['name']} {p['school_short']} volleyball\").")


# ---------------------------------------------------------------- cumulative-stats PDF parsing (deterministic; the model never picks her row)
# NCAA cumulative-stats sheet column order (after "# Player"):
CUME_COLS = ["sp","k","ks","e","ta","pct","a","as_","sa","se","sas","re","dig","digs","bs","ba","blk","blks","be","bhe","pts"]

def stats_from_cume(text, name, jersey=None):
    """Find HER row in a cumulative-stats PDF text and return the 15 stat fields the app uses. Deterministic; None if not found."""
    last, first = name.split()[-1], name.split()[0]
    flat = re.sub(r"[ \t]+", " ", text) + "\n"
    # row = optional jersey, "Last, First", then 21 numeric tokens (numbers, .333, -.250, 0)
    pat = re.compile(rf"(?:^|\n)\s*(\d{{1,2}})?\s*{re.escape(last)}\s*,\s*{re.escape(first)}\b[^\n\d-]*((?:-?\.?\d+(?:\.\d+)?\s+){{20,22}})", re.I)
    rows = [(m.group(1), m.group(2).split()) for m in pat.finditer(flat)]
    if jersey: rows = [r for r in rows if not r[0] or str(r[0]) == str(jersey)] or rows
    if not rows: return None
    nums = rows[-1][1][:21]
    if len(nums) < 21: return None
    v = dict(zip(CUME_COLS, nums))
    i = lambda k: int(float(v[k])) if re.fullmatch(r"-?\d+(\.0+)?", v[k]) else None
    return {"mp": None, "sp": i("sp"), "k": i("k"), "e": i("e"), "ta": i("ta"), "a": i("a"), "bhe": i("bhe"),
            "sa": i("sa"), "se": i("se"), "srv": None, "dig": i("dig"), "re": i("re"), "bs": i("bs"), "ba": i("ba"), "be": i("be")}

def mp_from_html_rows(html_lines, stats):
    """Matches played isn't on the PDF. Find the HTML stats row with the same SP/K/E/TA numbers (names there can be wrong, numbers aren't) and read MP."""
    for ln in html_lines:
        cells = [c.strip() for c in ln.split("\t")]
        nums = [c for c in cells if re.fullmatch(r"-?[\d.]+", c)]
        for o in (0, 1):                                  # with or without a leading jersey-number cell
            if len(nums) >= o + 9:
                try:
                    sp, mp, ms, pts, ptss, k, ks, e, ta = nums[o:o + 9]
                    if int(sp) == stats["sp"] and int(k) == stats["k"] and int(e) == stats["e"] and int(ta) == stats["ta"]: return int(mp)
                except ValueError: pass
    return None


def results_from_cume(text):
    """Results block of a cumulative sheet: '9/12/2026 vs LSU New OrleansW 3-1 22-25, ...' -> [{date, opponent, home_away, result}]."""
    out = []
    for m in re.finditer(r"(\d{1,2})/(\d{1,2})/(\d{4})\s+(at |vs )?([A-Za-z][^\n]*?)\s*([WL])\s+(\d)-(\d)", text):
        mo, dy, yr, ha, opp, wl, a, b = m.groups()
        out.append({"date": f"{yr}-{int(mo):02d}-{int(dy):02d}", "opponent": opp.strip(" .*"),
                    "home_away": "away" if ha == "at " else "neutral" if ha == "vs " else "home", "result": f"{wl} {a}-{b}"})
    return out


def apply_pdf_results(results, pdf_results):
    """Overwrite the model's W/L and score with the sheet's (matched by date, + opponent's first word when a date has two matches), and add any sheet result the model missed."""
    results = [r for r in (results if isinstance(results, list) else []) if isinstance(r, dict)]
    for x in pdf_results:
        same_day = [r for r in results if r.get("date") == x["date"]]
        pdf_same_day = [y for y in pdf_results if y["date"] == x["date"]]
        matched = any(same_opp(r.get("opponent"), x["opponent"]) for r in same_day)
        for r in same_day:
            if same_opp(r.get("opponent"), x["opponent"]) and not r.get("box_url") and x.get("box_url"): r["box_url"] = x["box_url"]
        if not matched and len(same_day) < len(pdf_same_day):
            results.append({"date": x["date"], "opponent": x["opponent"], "home_away": x["home_away"], "result": x["result"], "box_url": x.get("box_url"), "player_line": None})
    for r in results or []:
        if not isinstance(r, dict) or not r.get("date"): continue
        cands = [x for x in pdf_results if x["date"] == r["date"]]
        if len(cands) > 1:
            w = (r.get("opponent") or "").lower().split()[:1]
            cands = [x for x in cands if w and w[0] in x["opponent"].lower()] or cands
        if len(cands) == 1: r["result"] = cands[0]["result"]
    return results


MONTHS = {m: i for i, m in enumerate(["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"], 1)}


def results_from_schedule_lines(lines, year, home_city=None, home_host=None):
    """Completed games straight from a Sidearm schedule page's game lines: 'Sep 18 (Fri) 5:00 PM | Marshall | ... L, 3-0 ... [href .../boxscore/6819]'."""
    out = []
    for ln in lines:
        d = re.search(r"\b(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\.?[/ ]?\s*\|?\s*(\d{1,2})\b", ln)
        r = re.search(r"\b([WL]),?\s*\|?\s*(\d)\s*-\s*(\d)\b", ln)
        if not r and re.search(r"\bFinal\b", ln):                       # Presto style: "Final | 3 | Opponent | 1 | McHenry"
            r2 = re.search(r"\b([0-3])\b[^|\d]{0,60}\|[^|]*\|?\s*\b([0-3])\b", ln)
            if r2: r = re.match(r"([WL]),?\s*(\d)\s*-\s*(\d)", f"? {r2.group(1)}-{r2.group(2)}")
        if not d or not r: continue
        if re.search(r"exhibition|scrimmage|preseason|[-_]EXH[-_.]", ln, re.I): continue
        date = f"{year}-{MONTHS[d.group(1)]:02d}-{int(d.group(2)):02d}"
        box = re.search(r"\[href (\S*boxscore\S*)\]", ln, re.I)
        plain = re.sub(r"\[(href|img) [^\]]*\]", " ", ln)                      # judge home/away on visible text only, never on URLs
        if home_city and re.search(r"\b" + re.escape(home_city.split(",")[0]) + r"\b", plain, re.I): ha = "home"
        elif re.search(r"\bvs\.?\b", plain[:250]) and "@" in plain[:300]: ha = "neutral"
        elif re.search(r"\bat\b|@", plain[:250]): ha = "away"
        elif re.search(r"\bvs\.?\b", plain[:250]): ha = "home"
        else: ha = "away"
        # opponent: split the line into parts; prefer the part that links out to another school's site; else the first plausible name
        parts = [x.strip() for x in re.split(r"\s*\|\s*|\t+", ln) if x.strip()]
        own_host = (home_host or "").lower()
        junk = re.compile(r"^(Final|Live Stats?|Box Score( \(PDF\))?|Recap|Watch|Listen|Tickets|History|Stats|Gallery|Video|Photos|ESPN\+?|Flo\w*|vs\.?|at|@.*|/|\(?\w{3}\)?)$|"
                          r"^(OVC|MVC|Big South|NSIC|GAC|GSC|ECC|ISCC)\b|^\d|^\[img|^[WL],?$|\d\s*-\s*\d|\b(AM|PM|TBA|a\.m\.|p\.m\.)\b|,\s*[A-Z][a-z]{1,3}\.?$|,\s*[A-Z]{2}$|Arena|Center|Pavilion|Gym|Fieldhouse|Stadium", re.I)
        opp, fallback = None, None
        for part in parts:
            clean = re.sub(r"\[(href|img) [^\]]*\]", "", part).strip(" *")
            if not clean or junk.search(clean) or re.search(r"\b(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\b", clean): continue
            hrefs = re.findall(r"\[href (\S+)\]", part)
            ext = [h for h in hrefs if own_host and own_host not in h and not re.search(r"espn|flo|hudl|youtube|twitter|x\.com|instagram|facebook", h, re.I)]
            if ext and not opp: opp = clean
            if fallback is None: fallback = clean
        opp = re.sub(r"^(vs\.?|at|@)\s+", "", (opp or fallback or ""), flags=re.I)
        opp = re.sub(r"\s*\([^)]*\)\s*", " ", opp).strip()                                   # drop "(Hall of Fame Challenge at Malone)"
        opp = re.sub(r"\s+(Greek Night|Family Weekend|Student-Athlete Day|.*Day|.*Night|CAB Collab|Senior .*)$", "", opp).strip()
        if not opp or len(opp) > 60: continue
        out.append({"date": date, "opponent": opp, "home_away": ha, "result": f"{r.group(1)} {r.group(2)}-{r.group(3)}", "box_url": box.group(1) if box else None, "player_line": None})
    return out


MONTH_FULL = {m: i for i, m in enumerate(["january", "february", "march", "april", "may", "june", "july", "august", "september", "october", "november", "december"], 1)}


def events_from_sidearm_next(raw_html, base):
    """Newer Sidearm pages render each game as an <article> with a plain-English summary. Returns (results, upcoming, synthesized_lines)."""
    results, upcoming, lines = [], [], []
    for art in re.findall(r"<article.*?</article>", raw_html, re.S):
        text = re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", re.sub(r"<!--.*?-->", "", art))).strip()
        m = re.search(r"(Completed|Upcoming) Event:\s*\w+\s+(versus|at|vs\.?)\s+(.+?)\s+on\s+([A-Z][a-z]{2,8})\.?\s+(\d{1,2}),\s+(\d{4})(?:\s+at\s+([\d:]+\s*[AP]M))?", text)
        if not m: continue
        kind, prep, opp, mon, day, yr, tm = m.groups()
        mi = MONTH_FULL.get(mon.lower()) or MONTHS.get(mon[:3].title())
        if not mi: continue
        date = f"{yr}-{mi:02d}-{int(day):02d}"
        ha = "away" if prep == "at" else "home"
        if re.search(r"\bneutral\b", art, re.I): ha = "neutral"
        box = re.search(r'href="([^"]*boxscore[^"]*)"', art, re.I)
        stream = re.search(r'href="(https?://[^"]*(?:espn|flosports|flo\.|hudl|nsicnetwork|youtube)[^"]*)"', art, re.I)
        if kind == "Completed":
            r = re.search(r"\b(Win|Loss)\b\s*,?\s*(\d)\s*,?\s*to\s*,?\s*(\d)", text)
            if not r: continue
            res = f"{'W' if r.group(1) == 'Win' else 'L'} {r.group(2)}-{r.group(3)}"
            results.append({"date": date, "opponent": opp.strip(), "home_away": ha, "result": res, "box_url": urllib.request.urljoin(base, box.group(1)) if box else None, "player_line": None})
            lines.append(f"{mon[:3]} {int(day)} | {prep} {opp.strip()} | {res} | [href {results[-1]['box_url'] or ''}] Box Score")
        else:
            upcoming.append({"date": date, "time": tm, "opponent": opp.strip(), "home_away": ha, "stream_name": ("ESPN+" if stream and "espn" in stream.group(1).lower() else "stream") if stream else None, "stream_url": stream.group(1) if stream else None})
            lines.append(f"{mon[:3]} {int(day)} | {tm or 'TBA'} | {prep} {opp.strip()}" + (f" | [href {stream.group(1)}] Watch" if stream else ""))
    return results, upcoming, lines


TZ_ABBR = {"ET": "America/New_York", "EST": "America/New_York", "EDT": "America/New_York", "CT": "America/Chicago", "CST": "America/Chicago", "CDT": "America/Chicago",
           "MT": "America/Denver", "MST": "America/Denver", "MDT": "America/Denver", "PT": "America/Los_Angeles", "PST": "America/Los_Angeles", "PDT": "America/Los_Angeles"}


def to_central(time_text, date_iso, school_tz):
    """'5:00 PM' / '6 PM' / '12:30 p.m. CT' / '2:30pm EST / 1:30pm CST' -> ('5:00 PM', '4:00 PM CT'). Returns (as_listed, central) or (None, None)."""
    if not time_text: return None, None
    m = re.search(r"(\d{1,2})(?::(\d{2}))?\s*([AaPp])\.?\s*[Mm]\.?\s*(ET|EST|EDT|CT|CST|CDT|MT|MST|MDT|PT|PST|PDT)?", time_text)
    if not m: return time_text.strip(), None
    h, mi, ap, abbr = int(m.group(1)), int(m.group(2) or 0), m.group(3).upper(), (m.group(4) or "").upper()
    h = (h % 12) + (12 if ap == "P" else 0)
    listed = f"{(h - 1) % 12 + 1}:{mi:02d} {ap}M" + (f" {abbr}" if abbr else "")
    try:
        y, mo, d = (int(x) for x in date_iso.split("-"))
        src = ZoneInfo(TZ_ABBR.get(abbr, school_tz or "America/Chicago"))
        dt = datetime(y, mo, d, h, mi, tzinfo=src).astimezone(ZoneInfo("America/Chicago"))
        return listed, f"{(dt.hour - 1) % 12 + 1}:{dt.minute:02d} {'PM' if dt.hour >= 12 else 'AM'} CT"
    except Exception:
        return listed, None


STREAM_HOSTS = [(r"espn", "ESPN+"), (r"flosports|flovolleyball|flo\.", "FloSports"), (r"nsicnetwork", "NSIC Network"), (r"hudl", "Hudl"), (r"youtube", "YouTube"),
                (r"midco", "Midco Sports"), (r"bigsouth", "Big South Network"), (r"ovcdigital|ovc", "OVC Digital"), (r"glvc|gac", "GAC Network"), (r"stretchinternet|boxcast|vcloud|livestream", "School stream")]


def upcoming_from_schedule_lines(lines, year, home_city, home_host, school_tz, today):
    """Games with a date but no result yet: date, time (listed + Central), opponent, home/away, stream link."""
    out = []
    for ln in lines:
        if re.search(r"\b[WL],?\s*\|?\s*\d\s*-\s*\d\b|\bFinal\b|Box Score", ln): continue
        d = re.search(r"\b(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\.?[/ ]?\s*\|?\s*(\d{1,2})\b", ln)
        if not d: continue
        date = f"{year}-{MONTHS[d.group(1)]:02d}-{int(d.group(2)):02d}"
        if date < today.isoformat(): continue
        fake = [{"date": date, "opponent": "", "home_away": "", "result": "W 0-0", "box_url": None}]
        probe = results_from_schedule_lines([ln + " | W, | 0-0"], year, home_city, home_host)   # reuse the opponent/home-away logic
        if not probe: continue
        opp, ha = probe[0]["opponent"], probe[0]["home_away"]
        tm = re.search(r"\d{1,2}(?::\d{2})?\s*[AaPp]\.?\s*[Mm]\.?(?:\s*(?:ET|EST|EDT|CT|CST|CDT|MT|MST|MDT|PT|PST|PDT))?", ln)
        listed, ct = to_central(tm.group(0) if tm else None, date, school_tz)
        stream_name = stream_url = None
        for h in re.findall(r"\[href (\S+)\]", ln):
            for pat, name in STREAM_HOSTS:
                if re.search(pat, h, re.I) and not re.search(r"facebook|twitter|instagram|x\.com", h, re.I):
                    stream_name, stream_url = name, h; break
            if stream_url: break
        loc = re.search(r"\b([A-Z][A-Za-z.\s]+,\s*(?:[A-Z]{2}|[A-Z][a-z]{1,4}\.?))\b", re.sub(r"\[(href|img) [^\]]*\]", "", ln))
        out.append({"date": date, "time": listed, "time_ct": ct, "opponent": opp, "home_away": ha, "location": loc.group(1).strip() if loc else None, "stream_name": stream_name, "stream_url": stream_url})
    return out


def attach_box_links(results, raw_html, base):
    """Some layouts keep box-score links outside the game block. Collect every box-score href in page order and attach by opponent slug, else by order."""
    hrefs = [urllib.request.urljoin(base, h) for h in re.findall(r"""href=["']([^"']*boxscore[^"']*)["']""", raw_html, re.I)]
    seen, ordered = set(), []
    for h in hrefs:
        if h not in seen: seen.add(h); ordered.append(h)
    used = set()
    for r in results:
        if r.get("box_url"): used.add(r["box_url"]); continue
        slug = re.sub(r"[^a-z0-9]+", "-", (r.get("opponent") or "").lower()).strip("-")
        pick = next((h for h in ordered if h not in used and slug and slug.split("-")[0] in h.lower()), None)
        if pick: r["box_url"] = pick; used.add(pick)
    for r in results:                                   # second pass: leftovers by order (oldest first)
        if not r.get("box_url"):
            pick = next((h for h in ordered if h not in used), None)
            if pick: r["box_url"] = pick; used.add(pick)
    return results


HDR_MAP = {"sp": "sp", "mp": "mp", "ms": "ms", "k": "k", "e": "e", "ta": "ta", "a": "a", "ast": "a", "sa": "sa", "se": "se", "dig": "dig", "digs": "dig",
           "re": "re", "bs": "bs", "ba": "ba", "be": "be", "bhe": "bhe", "pct": "pct", "k/s": "ks", "a/s": "as_", "sa/s": "sas", "dig/s": "digs", "tb": "tb", "b/s": "bps", "pts": "pts", "pts/s": "ptss", "blk": "tb", "blk/s": "bps", "srv": "srv", "att": "srv"}


def stats_from_html_tables(text, name, jersey=None):
    """Classic Sidearm stats pages render real tables. Map columns by header, align on the Player column, take HER first (season/overall)
    offense row and defense row, merge. 'TA' in a defense table is reception attempts, not attack attempts. None if not found."""
    last, first = name.split()[-1].lower(), name.split()[0].lower()
    found, header, hi, in_conf = {}, None, None, False
    def is_her(c):
        c = re.sub(r"\s*,\s*", ", ", re.sub(r"\s+", " ", c.lower()))
        return c.startswith(last + ",") or c == f"{first} {last}" or c.endswith(f"{last}, {first}") or f"{last}, {first}" in c
    for ln in text.split("\n"):
        plain = re.sub(r"\[(href|img) [^\]]*\]", "", ln)
        if re.fullmatch(r"\s*conference\s*", plain, re.I) and header is not None: in_conf = True   # second block of tables = conference-only; skip it
        if re.fullmatch(r"\s*overall\s*", plain, re.I): in_conf = False
        cells = [c.strip() for c in plain.split("\t")]
        if len(cells) < 6: continue
        low = [re.sub(r"^(attack|set|serve|block|dig|defense|offense|recept)\s*/\s*", "", c.lower()) for c in cells]
        if "player" in low and any(h in low for h in ("sp", "k", "dig", "ba")):
            header, hi = low, low.index("player"); continue
        if header is None or in_conf: continue
        pi = next((i for i, c in enumerate(cells) if is_her(c)), None)
        if pi is None: continue
        if jersey and pi > 0 and cells[pi - 1].isdigit() and cells[pi - 1] != str(jersey): continue
        off = pi - hi
        is_offense = "k" in header and "ta" in header
        for i, h in enumerate(header):
            j = i + off
            if not (0 <= j < len(cells)) or h not in HDR_MAP: continue
            if h == "ta" and not is_offense: continue
            key = HDR_MAP[h]
            if key in ("sp", "mp", "ms", "k", "e", "ta", "a", "sa", "se", "dig", "re", "bs", "ba", "be", "bhe", "srv") and re.fullmatch(r"-?\d+(\.0+)?", cells[j] or ""):
                found.setdefault(key, int(float(cells[j])))                                  # first (overall) table wins
    if "sp" not in found or not any(k in found for k in ("k", "dig", "a")): return None
    return {k: found.get(k) for k in STAT_KEYS}


PRESTO_LABELS = {"matches": "mp", "sets": "sp", "kills": "k", "errors": "e", "total attacks": "ta", "assists": "a", "service total attempts": "srv",
                 "service aces": "sa", "service errors": "se", "reception errors": "re", "digs": "dig", "block solo": "bs", "block assist": "ba",
                 "block errors": "be", "ball handling errors": "bhe"}


def stats_from_presto_profile(text):
    """PrestoSports player page: 'STATISTICS CATEGORY | OVERALL | CONF' rows like 'Kills\t383\t102' (or 'Kills 383 102' as text). First number = overall."""
    found = {}
    for ln in text.split("\n"):
        plain = re.sub(r"\[(href|img) [^\]]*\]", "", ln).strip()
        m = re.match(r"^([A-Za-z][A-Za-z ]+?)\s*[\t ]+(-|\d+(?:\.\d+)?)\s*(?:[\t ]+(-|\d+(?:\.\d+)?))?\s*$", plain)
        if not m: continue
        label = m.group(1).strip().lower()
        if label in PRESTO_LABELS and PRESTO_LABELS[label] not in found and m.group(2) != "-":
            found[PRESTO_LABELS[label]] = int(float(m.group(2)))
    if "sp" not in found: return None
    return {k: found.get(k) for k in STAT_KEYS}


def results_from_presto_profile(text, year):
    """'RECENT GAMES' lines on a Presto profile: 'Oct 6 Elgin Community College W, 3-1' / 'Oct 3 vs. Delta W, 3-2' / 'Sep 23 at Oakton L, 3-1'."""
    out = []
    for ln in text.split("\n"):
        m = re.match(r"^\s*(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\.?\s+(\d{1,2})\s+(vs\.?\s+|at\s+)?(.+?)\s+([WL]),?\s*(\d)-(\d)\s*$", re.sub(r"\[(href|img) [^\]]*\]", "", ln).strip())
        if not m: continue
        mon, day, prep, opp, wl, a, b = m.groups()
        ha = "away" if (prep or "").startswith("at") else "neutral" if (prep or "").startswith("vs") else "home"
        out.append({"date": f"{year}-{MONTHS[mon]:02d}-{int(day):02d}", "opponent": opp.strip(), "home_away": ha, "result": f"{wl} {a}-{b}", "box_url": None, "player_line": None})
    return out


def pdf_text(b):
    """Text of a PDF via pypdf. Installs it on the fly if the workflow didn't. Returns "" if unavailable."""
    try:
        import io
        try:
            from pypdf import PdfReader
        except ImportError:
            import subprocess
            log("    installing pypdf"); subprocess.run([sys.executable, "-m", "pip", "install", "--quiet", "pypdf"], check=False, timeout=120)
            from pypdf import PdfReader
        return "\n".join((pg.extract_text() or "") for pg in PdfReader(io.BytesIO(b)).pages)
    except Exception as e:
        log(f"    pypdf failed: {str(e)[:60]}"); return ""


# ---------------------------------------------------------------- page fetching (Python browses, Claude only reads)
UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/128.0 Safari/537.36"


def http_get(url, timeout=30, binary=False):
    req = urllib.request.Request(url, headers={"User-Agent": UA, "Accept": "*/*", "Accept-Language": "en-US,en;q=0.9"})
    for attempt in (1, 2):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                data = r.read()
                return data if binary else data.decode(r.headers.get_content_charset() or "utf-8", "replace")
        except urllib.error.HTTPError:
            raise                                                # a real refusal (403/404/405) is final
        except Exception:
            if attempt == 2: raise                               # dropped connection / timeout: one retry after a pause
            time.sleep(4)


class Blocks(HTMLParser):
    """Turns HTML into text lines. Rows/list items/divs become lines; cells become tab-separated; hrefs and img srcs are kept as [href] / [img] tags."""
    BLOCK = {"tr", "li", "p", "h1", "h2", "h3", "h4", "section", "article", "table", "thead", "tbody", "ul", "ol"}   # these end a line
    SEP = {"div", "br", "dt", "dd", "span"}                                                                         # these just separate parts within a line
    SKIP = {"script", "style", "noscript", "svg", "head", "nav", "footer", "iframe"}
    CONTAINER = re.compile(r"schedule-game|sidearm-schedule-game|s-game|schedule__game|event-row|game-item|schedule-event", re.I)
    def __init__(self, base):
        super().__init__(convert_charrefs=True); self.out, self.cur, self.skip, self.base = [], [], 0, base
        self.stack = []            # open tags; entries marked True start a game container
        self.depth = 0             # >0 while inside a game container
    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        if tag in self.SKIP: self.skip += 1
        is_container = tag in ("li", "div", "article", "tr") and self.CONTAINER.search(a.get("class", "") or "") and self.depth == 0
        if is_container: self.flush(); self.depth = 1
        elif self.depth: self.depth += 1 if tag in self.BLOCK or tag in self.SEP or tag in ("td", "th", "a", "span") else 0
        self.stack.append((tag, is_container))
        if tag in ("td", "th"): self.cur.append("\t")
        elif tag in self.SEP or (self.depth and tag in self.BLOCK and not is_container): self.cur.append(" | ")
        if tag in self.BLOCK and not self.depth: self.flush()
        if tag == "a" and a.get("href"): self.cur.append(f" [href {urllib.request.urljoin(self.base, a['href'])}] ")
        if tag == "img" and (a.get("data-src") or a.get("src")): self.cur.append(f" [img {urllib.request.urljoin(self.base, a.get('data-src') or a['src'])}] ")
    def handle_endtag(self, tag):
        if tag in self.SKIP: self.skip = max(0, self.skip - 1)
        # pop to the matching open tag
        is_container = False
        for i in range(len(self.stack) - 1, -1, -1):
            if self.stack[i][0] == tag:
                is_container = self.stack[i][1]; del self.stack[i:]; break
        if self.depth:
            if is_container: self.depth = 0; self.flush(); return
            if tag in self.BLOCK or tag in self.SEP or tag in ("td", "th", "a", "span"): self.depth = max(1, self.depth - 1)
            return
        if tag in self.BLOCK: self.flush()
    def handle_data(self, d):
        if not self.skip:
            self.cur.append(d)
            if self.handle_data_len() > (6000 if self.depth else 1200): self.flush()      # pages without li/tr structure still get line breaks
    def handle_data_len(self): return sum(len(x) for x in self.cur)
    def flush(self):
        line = re.sub(r"[\s\xa0]+", " ", "".join(self.cur).replace("\t", "\x00")).replace("\x00", "\t")
        line = re.sub(r"(\s*\|\s*)+", " | ", line).strip(" \t|")
        if line: self.out.append(line)
        self.cur = []
    def text(self):
        self.flush(); return "\n".join(self.out)


def page_text(url):
    raw = http_get(url); b = Blocks(url); b.feed(raw); return b.text(), raw


def via_reader(url):
    """Last resort for sites that block datacenter traffic: a public reader proxy returns the page as markdown text."""
    md = http_get("https://r.jina.ai/" + url, timeout=20)
    lines = [re.sub(r"^\|\s*|\s*\|$", "", ln).replace(" | ", "\t") if ln.strip().startswith("|") else ln for ln in md.split("\n")]
    return "\n".join(l for l in lines if not re.fullmatch(r"[\s\-|:]*", l)), md


def browser_get(url, timeout=30):
    """Real headless Chromium (Playwright) for sites that block plain requests. Returns rendered HTML; raises if Playwright isn't installed or the page fails."""
    import subprocess, sys as _sys
    code = (
        "import sys\n"
        "from playwright.sync_api import sync_playwright\n"
        "with sync_playwright() as p:\n"
        "    try: b = p.chromium.launch(channel='chrome', args=['--disable-blink-features=AutomationControlled'])\n"
        "    except Exception: b = p.chromium.launch(args=['--disable-blink-features=AutomationControlled'])\n"
        "    ctx = b.new_context(user_agent=%r, locale='en-US', viewport={'width':1280,'height':900})\n"
        "    pg = ctx.new_page(); pg.goto(sys.argv[1], wait_until='networkidle', timeout=%d); pg.wait_for_timeout(1500)\n"
        "    sys.stdout.write(pg.content()); b.close()\n" % (UA, timeout * 1000))
    r = subprocess.run([_sys.executable, "-c", code, url], capture_output=True, text=True, timeout=timeout + 30)
    if r.returncode != 0 or len(r.stdout) < 500: raise RuntimeError("browser fetch failed: " + (r.stderr.strip().splitlines() or ["no output"])[-1][:120])
    return r.stdout


def first_working(cands, marker=None, notes=None):
    """Try candidate URLs in order; return (text, raw, url) for the first that loads and (if given) contains the marker. Falls back to a reader proxy."""
    tried_hosts = set()
    for attempt in ("direct", "reader", "browser"):
        for u in cands or []:
            if attempt == "browser":
                h = urllib.parse.urlparse(u).netloc
                if h in tried_hosts: continue
                tried_hosts.add(h)
            try:
                if attempt == "direct": text, raw = page_text(u)
                elif attempt == "reader": text, raw = via_reader(u)
                else:
                    raw = browser_get(u)
                    if re.search(r"Human Verification|captcha|Access Denied|Just a moment", raw[:5000], re.I): raise RuntimeError("site presented a CAPTCHA / bot challenge")
                    b = Blocks(u); b.feed(raw); text = b.text()
                    if DRY_RUN:
                        os.makedirs(os.path.join(HERE, "dry-run-pages"), exist_ok=True)
                        fn = re.sub(r"[^a-z0-9]+", "_", u.lower())[:120] + ".html"
                        open(os.path.join(HERE, "dry-run-pages", fn), "w", encoding="utf-8").write(raw)
                if marker is None or marker.lower() in text.lower(): 
                    if notes is not None and attempt != "direct": notes.append(f"{u.split('/')[2]} via {attempt}")
                    return text, raw, u
                if notes is not None: notes.append(f"{u.split('/')[2]}: no '{marker}'")
            except Exception as e:
                if notes is not None: notes.append(f"{u.split('/')[2]} ({attempt}): {str(e)[:40]}")
    return None, "", None


def keep_lines(text, patterns, context=0, max_chars=24000):
    """Keep lines matching any pattern (plus neighbours); fall back to the head of the page if nothing matches."""
    lines = text.split("\n"); rx = [re.compile(p, re.I) for p in patterns]; keep = set()
    for i, ln in enumerate(lines):
        if any(r.search(ln) for r in rx):
            for j in range(max(0, i - context), min(len(lines), i + context + 1)): keep.add(j)
    out = "\n".join(lines[i] for i in sorted(keep)) if keep else text
    return out[:max_chars]


def find_pdf(raw_html, base):
    """The season cumulative stats sheet, if the page links one. Prefers direct S3 'cume' links, then anything that looks like it."""
    direct = re.search(r"""(https?://[^"'\s]*(?:amazonaws\.com|sidearm)[^"'\s]*cume\.pdf[^"'\s]*)""", raw_html, re.I)
    if direct: return direct.group(1)
    links = [urllib.request.urljoin(base, m) for m in re.findall(r"""(?:href|src)=["']([^"']+(?:\.pdf|/pdf/?)(?:[?#][^"']*)?)["']""", raw_html, re.I)]
    for pat in (r"cume\.pdf", r"stats?/\d{4}/pdf", r"stats/.*(?:season|overall|cume)"):
        for u in links:
            if re.search(pat, u, re.I): return u
    return None


def fetch_pdf(url, max_bytes=3_000_000, hops=1):
    b = http_get(url, binary=True)
    if b[:2] == b"\x1f\x8b":
        import gzip; b = gzip.decompress(b)
    b = b.lstrip()
    if b.startswith(b"%PDF") and len(b) <= max_bytes: return b
    if hops and b[:200].lower().lstrip().startswith((b"<!doctype", b"<html")):
        html = b.decode("utf-8", "replace")
        real = (re.search(r"""(https?://[^"'\s]+(?<!conf)cume\.pdf[^"'\s]*)""", html, re.I)
                or re.search(r"""(https?://[^"'\s]+(?:cume|overall|season)[^"'\s]*\.pdf[^"'\s]*)""", html, re.I)
                or re.search(r"""(https?://[^"'\s]*amazonaws\.com/[^"'\s]+\.pdf[^"'\s]*)""", html, re.I)
                or re.search(r"""(?:src|href)=["']([^"']+\.pdf[^"']*)["']""", html, re.I))
        if real: return fetch_pdf(urllib.request.urljoin(url, real.group(1)), max_bytes, hops - 1)
    raise ValueError(f"not a usable PDF (starts {b[:12]!r}, {len(b)} bytes)")


def record_from(text):
    """'Overall 4-4 · Conf 0-0' style record text from a Sidearm schedule page, if present."""
    o = re.search(r"Overall\D{0,12}(\d{1,2}-\d{1,2})", text, re.I)
    c = re.search(r"Conf(?:erence)?\D{0,12}(\d{1,2}-\d{1,2})", text, re.I)
    return {"overall": o.group(1) if o else None, "conference": c.group(1) if c else None}


def gather_daily(p, today):
    """Fetch stats (PDF preferred), schedule, and the box scores she still needs. Returns compact text + attachments, or raises."""
    parts, docs, notes, pdf_results, page_results = [], [], [], [], []
    surname = p["name"].split()[-1]
    # schedule page: game blocks with box score links
    if p.get("schedule_candidates"):
        sched_text, sched_raw, used = first_working(p["schedule_candidates"], None, notes)
        if sched_text is None: raise RuntimeError("no schedule source answered")
        notes.append(f"schedule from {used.split('/')[2]}")
    else:
        sched_text, sched_raw = page_text(p["schedule_url"])
    rec = record_from(sched_text)
    if rec["overall"]: notes.append(f"record from page: {rec['overall']}")
    game_lines = [ln for ln in sched_text.split("\n") if re.search(r"\b(Aug|Sep|Oct|Nov|Dec)\b\.?[/ ]?\s*\|?\s*\d{1,2}|\d{1,2}/\d{1,2}(/\d{2,4})?", ln)
                  and re.search(r"\b(vs\.?|at|Final|[WL],?\s*\|?\s*\d-\d|\d\s*-\s*\d|PM|AM|TBA)\b", ln, re.I)]
    parts.append("=== SCHEDULE / RESULTS PAGE (each line is one game; [href ...] are that game's links) ===\n" + "\n".join(game_lines)[:30000])
    page_results = results_from_schedule_lines(game_lines, today.year, p.get("city"), urllib.parse.urlparse(p.get("schedule_url") or p.get("site") or "").netloc)
    if not page_results and "Event:" in sched_raw and "<article" in sched_raw:
        page_results, _up, synth = events_from_sidearm_next(sched_raw, p.get("schedule_url") or p.get("site") or "")
        if synth: game_lines = synth; parts[0] = "=== SCHEDULE / RESULTS PAGE (each line is one game; [href ...] are that game's links) ===\n" + "\n".join(synth)[:30000]
        if page_results: notes.append("results from event articles (newer Sidearm layout)")
    from collections import Counter
    keys = [(r["date"], opp_key(r["opponent"])[:8], r["result"]) for r in page_results]
    counts = Counter(keys)
    if page_results and sum(1 for k in counts if counts[k] == 2) >= 0.8 * len(counts):   # layout renders every game twice (list + table views)
        seen_keys, dedup = set(), []
        for r, k in zip(page_results, keys):
            if k not in seen_keys: seen_keys.add(k); dedup.append(r)
        page_results = dedup
    attach_box_links(sorted(page_results, key=lambda r: r["date"]), sched_raw, p.get("schedule_url") or p.get("site") or "")
    with_box = sum(1 for r in page_results if r.get("box_url"))
    if page_results and with_box / len(page_results) >= 0.8:      # a real game on these sites has a box score; the rest are exhibitions
        page_results = [r for r in page_results if r.get("box_url")]
    if page_results: notes.append(f"{len(page_results)} results parsed from the schedule page")
    else:
        sample = [ln[:160] for ln in game_lines[:3]] or [ln[:160] for ln in sched_text.split("\n") if re.search(r"\b(Sep|Oct)\b", ln)][:3]
        notes.append("no results parsed; sample lines: " + " || ".join(sample))
    # stats: cumulative PDF if the page links one, else the trimmed HTML table
    stats_text, stats_raw, pdf = "", "", None
    try:
        if p.get("stats_candidates"):
            stats_text, stats_raw, used = first_working(p["stats_candidates"], surname, notes)
            if stats_text is None: stats_text, stats_raw = "", ""; notes.append("no stats source had her name")
            else: notes.append(f"stats from {used.split('/')[2]}")
            pdf = find_pdf(stats_raw, used) if used else None
        else:
            stats_text, stats_raw = page_text(p["stats_url"])
            pdf = find_pdf(stats_raw, p["stats_url"])
    except Exception as e:
        notes.append(f"stats page failed ({str(e)[:50]}) — results/schedule still parsed")        # a stats outage must not take the schedule down with it
    parsed = None
    if pdf:
        try:
            pdf_bytes = fetch_pdf(pdf); txt = pdf_text(pdf_bytes)
            parsed = stats_from_cume(txt, p["name"], p.get("jersey")) if txt else None
            pdf_results = results_from_cume(txt) if txt else []
            if pdf_results: notes.append(f"{len(pdf_results)} results on the sheet")
            m_rec = re.search(r"Overall\s*Record:?\s*(\d{1,2}-\d{1,2})", txt or "", re.I)
            if m_rec and not rec.get("overall"): rec["overall"] = m_rec.group(1); notes.append(f"record from sheet: {rec['overall']}")
            if parsed:
                parsed["mp"] = mp_from_html_rows(stats_text.split("\n"), parsed)
                notes.append(f"stats parsed from PDF: sp={parsed['sp']} k={parsed['k']} a={parsed['a']} dig={parsed['dig']}")
            elif txt and re.search(rf"\b{re.escape(surname)}\s*,", txt, re.I) is None:
                parsed = {k: (0 if k not in ("srv", "re", "mp") else None) for k in STAT_KEYS}; notes.append("not on the stats sheet — zeros")
            else:
                docs.append(("Season cumulative stats PDF", pdf_bytes)); notes.append("PDF row not parsed — PDF attached for reading" if txt else "PDF text extraction failed — PDF attached for reading")
        except Exception as e:
            notes.append(f"pdf skipped ({str(e)[:50]})")
    html_stats = stats_from_html_tables(stats_text, p["name"], p.get("jersey")) if stats_text else None
    if p.get("profile_url"):
        try:
            try: prof_text, _ = page_text(p["profile_url"])
            except Exception:
                try: prof_text, _ = via_reader(p["profile_url"]); notes.append("profile via reader")
                except Exception:
                    raw_ = browser_get(p["profile_url"]); b_ = Blocks(p["profile_url"]); b_.feed(raw_); prof_text = b_.text(); notes.append("profile via browser")
                    if DRY_RUN:
                        os.makedirs(os.path.join(HERE, "dry-run-pages"), exist_ok=True)
                        open(os.path.join(HERE, "dry-run-pages", re.sub(r"[^a-z0-9]+", "_", p["profile_url"].lower())[:120] + ".html"), "w", encoding="utf-8").write(raw_)
            prof = stats_from_presto_profile(prof_text)
            if prof:
                html_stats = prof; notes.append(f"stats parsed from player profile: sp={prof['sp']} k={prof['k']} dig={prof['dig']}")
                page_results = page_results or results_from_presto_profile(prof_text, today.year)
        except Exception as e:
            notes.append(f"profile fetch failed ({str(e)[:50]})")
    if html_stats and not parsed:
        parsed = html_stats; notes.append(f"stats parsed from HTML table: sp={parsed['sp']} k={parsed['k']} a={parsed['a']} dig={parsed['dig']}")
    elif html_stats and parsed:
        diffs = [k for k in ("sp", "k", "a", "dig", "ba") if html_stats.get(k) is not None and parsed.get(k) is not None and html_stats[k] != parsed[k]]
        rec_games = sum(int(x) for x in rec["overall"].split("-")) if rec.get("overall") else None
        pdf_partial = bool(pdf_results) and rec_games and len(pdf_results) < 0.7 * rec_games
        if not diffs: notes.append("HTML table agrees with PDF")
        elif pdf_partial or (html_stats.get("sp") or 0) > (parsed.get("sp") or 0):
            notes.append(f"PDF looks partial ({len(pdf_results)} games vs {rec_games} on record) — HTML table kept"); parsed = html_stats
        else: notes.append(f"HTML table disagrees with PDF on {diffs} — PDF kept")
        if parsed.get("mp") is None and html_stats.get("mp") is not None: parsed["mp"] = html_stats["mp"]
    if parsed:
        parts.append(f"=== HER SEASON STAT LINE (authoritative, already extracted from the official stats sheet) ===\n{json.dumps(parsed)}")
    else:
        parts.append("=== STATS PAGE (tab-separated rows; names on this table can be wrong when two players share a number — match by jersey number AND name; the PDF, if attached, is authoritative) ===\n" +
                     keep_lines(stats_text, [r"Player|\bSP\b|Kills|Assists", surname, rf"^\s*{re.escape(str(p.get('jersey') or ''))}\t"], 0, 12000))
    # box scores for matches still missing her line (newest first, max 2)
    need = [m for m in (p.get("recent_matches") or []) if m.get("result") and not m.get("player_line") and m.get("box_url")][:2]
    for m in need:
        try:
            bx, _ = page_text(m["box_url"])
            parts.append(f"=== BOX SCORE {m['date']} vs {m['opponent']} ({m['box_url']}) — individual rows ===\n" +
                         keep_lines(bx, [r"Player|\bSP\b", surname, p["school_short"].split()[0]], 1, 6000))
        except Exception as e:
            notes.append(f"box {m['date']} failed: {str(e)[:60]}")
    # which list reconciles with the posted record? that one is the season truth (sheet preferred); the other only donates box-score links
    def wl_of(lst): 
        w = [fix_result(x["result"])[0] for x in lst if x.get("result")]; return f"{w.count('W')}-{w.count('L')}"
    authoritative = None
    if rec.get("overall"):
        if pdf_results and wl_of(pdf_results) == rec["overall"]: authoritative = "sheet"
        elif page_results and wl_of(page_results) == rec["overall"]: authoritative = "page"
    if authoritative: notes.append(f"results reconcile with record via {authoritative}")
    # sheet results are authoritative for scores; page results supply box-score links and anything the sheet lacks
    for y in pdf_results:
        for x in page_results:
            if x["date"] == y["date"] and same_opp(x["opponent"], y["opponent"]) and x.get("box_url") and not y.get("box_url"):
                y["box_url"] = x["box_url"]
    if authoritative == "sheet": combined = pdf_results
    elif authoritative == "page": combined = page_results
    else: combined = pdf_results + [x for x in page_results if not any(x["date"] == y["date"] and same_opp(x["opponent"], y["opponent"]) for y in pdf_results)]
    for x in combined: x["_authoritative"] = bool(authoritative)
    return "\n\n".join(parts), docs, notes, rec, parsed, combined


# ---------------------------------------------------------------- research calls
INFLATED = re.compile(r"\b(steady part|key (part|piece|contributor)|anchor|staple|mainstay|regular (part|fixture)|integral|cornerstone|go-to|leader on|leading the)\b", re.I)


def role_facts(p, today, season_results=None):
    """Her share of the team's sets and whether she played lately — numbers the writer must match.
    season_results: the full-season results list from the stats sheet when available (the stored list may only go back a few weeks)."""
    st = p.get("stats") or {}
    sp = st.get("sp") or 0
    team_sets = 0
    pool = season_results if season_results and len(season_results) >= len(p.get("recent_matches") or []) else (p.get("recent_matches") or [])
    for m in pool:
        r = re.search(r"(\d)\s*-\s*(\d)", m.get("result") or "")
        if r: team_sets += int(r.group(1)) + int(r.group(2))
    team_sets = max(team_sets, sp)
    share = (sp / team_sets) if team_sets else 0
    word = "has not appeared" if sp == 0 else "limited minutes" if share < 0.15 else "rotational" if share < 0.5 else "regular"
    wk = (today - timedelta(days=7)).isoformat()
    played_wk = any((m.get("date") or "") >= wk and m.get("result") and m.get("player_line") and "did not play" not in (m.get("player_line") or "").lower()
                    for m in p.get("recent_matches") or [])
    team_played_wk = any((m.get("date") or "") >= wk and m.get("result") for m in p.get("recent_matches") or [])
    return {"sets_played": sp, "team_sets": team_sets, "share": round(share, 2), "role_word": word,
            "played_this_week": played_wk, "team_played_this_week": team_played_wk}


def research_daily(p, today, need_lines=()):
    since = (today - timedelta(days=14)).isoformat()
    ident = f"#{p['jersey']} " if p.get("jersey") else ""
    who = (f"{p['name']} ({ident}{p.get('position') or p['club_position']}, freshman, {p.get('hometown_hs') or 'hometown per roster'}). "
           f"{p.get('disambiguation', '')} {p.get('position_note', '')}")
    schema = ('{"team_record": {"overall": "W-L", "conference": "W-L or null"},\n'
              ' "stats": {"mp":0,"sp":0,"k":0,"e":0,"ta":0,"a":0,"bhe":0,"sa":0,"se":0,"srv":null,"dig":0,"re":null,"bs":0,"ba":0,"be":0},\n'
              ' "results": [{"date":"YYYY-MM-DD","opponent":"","home_away":"home","result":"W 3-1","box_url":null,"player_line":null}],\n'
              ' "blurb": ""}')
    text, docs, notes, rec, parsed, pdf_results = None, [], [], {}, None, []
    if not p.get("fetch_note"):
        try: text, docs, notes, rec, parsed, pdf_results = gather_daily(p, today)
        except Exception as e: notes, rec, parsed, pdf_results = [f"fetch failed ({str(e)[:80]}) — used search"], {}, None, []
    rf = role_facts({**p, "stats": parsed or p.get("stats")}, today, pdf_results)
    rules = f"""Rules:
- stats: if a section "HER SEASON STAT LINE" is present, copy it exactly. Otherwise take HER single row from the stats table (match by jersey number AND last name; never add rows together; if two rows could be her, return null stats). Integers; null where a column is not published.
- team_record = the record printed on the schedule/results page (e.g. "Overall 4-4"); copy it, do not tally matches yourself.
- result is written W/L then HER team's sets first: "W 3-1", "L 0-3" — never "L 3-0".
- results = the team's matches from {since} through {today.isoformat()} that have a final score, most recent first, with box_url = the box-score link from that game's block. player_line = her numbers from a BOX SCORE section below if one is present for that match ("7 kills, 3 blocks, 2 digs" / "24 assists, 6 digs" / "did not play"); otherwise null.
- blurb = {"3-4" if p.get("featured") and rf.get("played_this_week") else "2-3"} sentences for her parents. Court-time facts (from code, not negotiable): sets played {rf['sets_played']} of the team's {rf['team_sets']} = {int(rf['share']*100)}% → describe her role as "{rf['role_word']}"; played this week: {rf['played_this_week']}; team played this week: {rf['team_played_this_week']}.
  Order: (1) if the team played and she did not, say so plainly in the first sentence, then what she did last time she played; otherwise what SHE did this week — her line, stated first and warmly; (2) the team's results, plainly, without dwelling on losses ("dropped two at Marshall" not "hit a rough patch"); (3) what is next for her.
  Never describe her role with words bigger than the percentage supports — no "steady part of the rotation", "key contributor", "anchor" for a rotational or limited-minutes player. Treat any position note above as fact and never mention where it came from (no "per the family", no "listed as").
  The stats sheet decides court time: if she is not in it or has 0 sets played, say plainly that she has not appeared in a match yet and move on to the team. Never write about the data itself — no mention of pages, PDFs, box scores, tables, or what could or could not be found. Write only about her and the team.
Return ONLY: {schema}"""
    if text is None or len(text) < 200:      # blocked or empty site: one search-only call
        prompt = f"""Today is {today.isoformat()}. Player: {who}
Her school's site blocks automated reading, so there is no stats table here. Rules override: stats = null for every field unless a search snippet shows her actual season line; never write zeros for lack of information; and never say she has not played — if you cannot tell, say nothing about her court time and describe the team. Use web_search (up to 3 searches: "{p['school']} volleyball {p['name'].split()[-1]}", "{p['school']} volleyball results 2026", "{p['school']} volleyball stats") and read the result snippets only.
{rules}"""
        out = call_claude(prompt, [{**SEARCH, "max_uses": 3}], max_tokens=3000, model=p.get("model")); out["_notes"] = notes; return out
    prompt = f"""Today is {today.isoformat()}. Player: {who}
Below is text pulled from her school's schedule/results page, her stats (PDF attached if available, else the stats table), and any box scores she still needs a line for. Read only what is here — do not guess beyond it.

{text}

{rules}"""
    try:
        out = call_claude(prompt, None, max_tokens=4500, model=p.get("model"), docs=docs)
    except RuntimeError as e:
        if docs and "pdf" in str(e).lower():
            notes.append("API rejected the PDF — read the HTML table instead"); out = call_claude(prompt, None, max_tokens=4500, model=p.get("model"))
        else: raise
    out["_notes"] = notes
    rf2 = role_facts({**p, "stats": parsed or out.get("stats") or p.get("stats")}, today, pdf_results)
    if rf2["share"] < 0.5 and INFLATED.search(out.get("blurb") or ""):
        notes.append("blurb inflated her role — trimmed")
        out["blurb"] = " ".join(x for x in re.split(r"(?<=[.!?])\s+", out["blurb"]) if not INFLATED.search(x)) or out["blurb"]
    out["stats_source"] = "sheet-parsed" if parsed else "model-read"
    if parsed: out["stats"] = parsed                     # code-parsed line overrides whatever the model wrote
    if rec.get("overall"):          # the page's own record beats anything the model tallied
        out["team_record"] = {**(out.get("team_record") or {}), "overall": rec["overall"], **({"conference": rec["conference"]} if rec.get("conference") else {})}
    if pdf_results: out["results"] = apply_pdf_results(out.get("results"), pdf_results)   # the sheet's scores beat the model's; missed games get added
    if pdf_results and all(x.get("_authoritative") for x in pdf_results):
        out["results_authoritative"] = True
        out["results"] = [x for x in out["results"] if any(x.get("date") == y["date"] and same_opp(x.get("opponent"), y["opponent"]) for y in pdf_results)] or out["results"]
    for x in out.get("results") or []: x.pop("_authoritative", None)
    for m in out.get("results") or []:
        if isinstance(m, dict) and m.get("result"): m["result"] = fix_result(m["result"])
    return out


def research_weekly(p, today):
    """Full remaining schedule (Python-fetched) + standings (one search)."""
    shape = ('{"schedule": [{"date":"YYYY-MM-DD","time":"6:00 PM ET","time_ct":"5:00 PM CT","opponent":"","home_away":"home","location":null,"stream_name":null,"stream_url":null}], '
             '"standing": "3rd of 11 OVC or null", "standings_url": null}')
    sched_text, raw = None, ""
    if not p.get("fetch_note"):
        try:
            if p.get("schedule_candidates"): sched_text, raw, _ = first_working(p["schedule_candidates"])
            else: sched_text, raw = page_text(p["schedule_url"])
        except Exception as e: log(f"    schedule fetch failed ({str(e)[:60]}) — using search")
    if not sched_text or len(sched_text) < 200:
        prompt = f"""Today is {today.isoformat()}. Team: {p['school']} volleyball ({p['conference']}). Its site blocks automated reading; use web_search (2 searches) for the remaining 2026 schedule and the conference standings. Times in Central.
Return ONLY: {shape}"""
        return call_claude(prompt, [{**SEARCH, "max_uses": 2}], max_tokens=4000, model=STRONG_MODEL)
    glines = [ln for ln in sched_text.split("\n") if re.search(r"\b(Aug|Sep|Oct|Nov|Dec)\b\.?[/ ]?\s*\|?\s*\d{1,2}", ln)]
    host = urllib.parse.urlparse(p.get("schedule_url") or "").netloc
    code_upcoming = upcoming_from_schedule_lines(glines, today.year, p.get("city"), host, p.get("tz"), today)
    if "Event:" in raw and "<article" in raw:
        _r, _u, synth = events_from_sidearm_next(raw, p.get("schedule_url") or "")
        for u in _u:
            u["time"], u["time_ct"] = to_central(u.get("time"), u["date"], p.get("tz")); u.setdefault("location", None)
        code_upcoming = code_upcoming or _u; glines = synth or glines
    trimmed = "\n".join(glines)[:22000]
    ig = re.search(r"instagram\.com/([A-Za-z0-9_.]+)", raw); xx = re.search(r"(?:twitter|x)\.com/([A-Za-z0-9_]+)", raw)
    prompt = f"""Today is {today.isoformat()}. Team: {p['school']} {p.get('team_name', '')} volleyball ({p['division']}, {p['conference']}); the school is in the {p['tz']} time zone.
Below is the schedule page as text (one game per line; [href ...] are that game's links, including streaming/TV links).

{trimmed}

Return every match from {today.isoformat()} through the end of the season (conference tournament too if listed). time = as listed; time_ct = converted to US Central. stream_name/stream_url = the streaming/TV label and link in that game's block, else null.
Then use web_search ONCE for "{p['conference']} volleyball standings 2026" and report the team's place as "3rd of 11 OVC" with the standings page URL (null if not found).
Return ONLY: {shape}"""
    if code_upcoming:                       # schedule from code; one small call just for standings
        sprompt = f"""Use web_search ONCE for "{p['conference']} volleyball standings 2026" and report {p['school']}'s place as "3rd of 11 OVC" (null if not found) with the standings page URL.
Return ONLY: {{"standing": null, "standings_url": null}}"""
        try: st = call_claude(sprompt, [{**SEARCH, "max_uses": 1}], max_tokens=400, model=STRONG_MODEL)
        except Exception as e: log(f"    standings lookup failed: {str(e)[:60]}"); st = {}
        return {"schedule": code_upcoming, "standing": st.get("standing"), "standings_url": st.get("standings_url"), "socials": {"instagram": ig.group(1) if ig else None, "x": xx.group(1) if xx else None}}
    out = call_claude(prompt, [{**SEARCH, "max_uses": 1}], max_tokens=5000, model=STRONG_MODEL)
    out["socials"] = {"instagram": ig.group(1) if ig else None, "x": xx.group(1) if xx else None}
    return out


def research_profile(p):
    shape = '{"jersey":null,"position":null,"class_year":null,"height":null,"hometown_hs":"Hometown, ST / High School","bio_url":null,"photo_url":null}'
    text = None
    if not p.get("fetch_note"):
        try:
            if p.get("roster_candidates"): text, _, _ = first_working(p["roster_candidates"], p["name"].split()[-1])
            else: text, _ = page_text(p["roster_url"])
        except Exception as e: log(f"    roster fetch failed ({str(e)[:60]}) — using search")
    if not text or p["name"].split()[-1].lower() not in text.lower():
        prompt = f"""Player: {p['name']}, {p['school']} volleyball, freshman. Her school's roster page blocks automated reading; use web_search (2 searches) for her roster entry.
Return ONLY: {shape}"""
        return call_claude(prompt, [{**SEARCH, "max_uses": 2}], max_tokens=800, model=STRONG_MODEL)
    block = keep_lines(text, [re.escape(p["name"].split()[-1])], 3, 6000)
    prompt = f"""Player: {p['name']}, {p['school']} volleyball. Below are the lines from the roster page around her name ([img ...] = image URLs, [href ...] = links).

{block}

photo_url = the [img] URL that is her headshot, or null.
Return ONLY: {shape}"""
    return call_claude(prompt, None, max_tokens=800, model=STRONG_MODEL)


def research_news(p, today, first_run=False):
    surname = p["name"].split()[-1]
    prompt = f"""Today is {today.isoformat()}, mid-season (the 2026 college volleyball season began in late August). Use web_search only — up to 3 searches:
 1. "{p['name']}" volleyball {p['school_short']}
 2. {p['school_short']} volleyball {surname}
From the result titles and snippets, list up to 6 items from THIS SEASON (August 2026 onward; older items only if clearly about her joining this team) that either
(a) mention {p['name']} by name — match recaps, features, roster/signing news, local papers (Daily Herald, Kane County Reporter, Northwest Herald, Elgin Courier-News), conference weekly honors — or
(b) are conference weekly honors or team milestones that name her.
A school recap that names her counts. Skip items that do not name her or the staff. If the snippet shows a date, use it; otherwise use today's date. For recaps, put her line or the quote in "note".
Return ONLY: {{"items": [{{"date":"YYYY-MM-DD","kind":"news","title":"","source":"publication name","url":"https://...","note":null}}]}}"""
    return call_claude(prompt, [{**SEARCH, "max_uses": 2}], max_tokens=1500, model=STRONG_MODEL)


def write_piece(kind, players, today, milestones, reunions):
    wk_ago, wk_ahead = (today - timedelta(days=7)).isoformat(), (today + timedelta(days=7)).isoformat()
    compact = []
    for p in players:
        if p.get("status") != "playing": continue
        played_this_week = any((m.get("date") or "") >= wk_ago and m.get("result") for m in p.get("recent_matches", []))
        compact.append({k: p.get(k) for k in ("name", "school_short", "division", "position", "position_note", "team_record", "stats", "blurb", "stale")}
                       | {"featured": bool(p.get("featured")) and played_this_week}
                       | {"results_last_7": [m for m in p.get("recent_matches", []) if (m.get("date") or "") >= wk_ago],
                          "next_7": [m for m in p.get("upcoming", []) if today.isoformat() <= (m.get("date") or "") <= wk_ahead]})
    if kind == "recap":
        ask = ("Write the MONDAY WEEKEND RECAP. Paragraph 1: the weekend in two or three sentences — who stood out, any milestone, any big team result. "
               "Then ONE short paragraph for EACH girl who had a match this week, in this form: her first name, the results, her line, one human note "
               "(e.g. 'Anna — Morehead State split at Marshall (L 1-3, W 3-2); 5 kills and 4 blocks Saturday, her best block night yet.'). "
               "Skip girls with no match this week. Say plainly if a girl did not see the court. "
               "The girl marked featured:true gets the first per-girl paragraph and 2-3 sentences ONLY IF she played this week; if she did not play, one plain sentence and no more. "
               "The spotlight is earned: it goes to whoever had the best actual line this week, and never to a girl who did not play.")
    else:
        ask = ("Write the THURSDAY WEEKEND PREVIEW. Paragraph 1: the weekend ahead in two or three sentences — the biggest matches, conference openers, anything at stake. "
               "Then ONE short paragraph (1-2 sentences) for EACH girl with a match this weekend: first name, opponent(s), day and Central time, stream, and why it matters "
               "(e.g. 'Kylie — Arkansas Tech at Southern Nazarene, Fri 6 PM CT on FloSports; a win keeps the Suns alone atop the GAC.'). "
               "The girl marked featured:true gets the first per-girl paragraph and 2-3 sentences if she has a match this weekend; otherwise she is not singled out.")
    prompt = f"""Today is {today.isoformat()}. Data for the girls (JSON): {json.dumps(compact, ensure_ascii=False)}
Milestones this week: {json.dumps(milestones)}
Reunions coming up: {json.dumps(reunions[:3])}

{ask}
Rules: title under 12 words, no "Recap:" prefix. body = the paragraphs described above (the intro plus one per girl), each starting with her first name followed by " — ", each under 70 words, plain warm language, positives before results, no hype, no bullet points, no jargon.
Ignore anyone marked stale. spotlight = one girl with the best week and a one-sentence reason, or null.
Return ONLY: {{"title":"", "body":["",""], "spotlight": {{"name":"First Last","note":""}} }}"""
    system = "You write short, warm notes for a group of volleyball moms whose daughters played club together and are now college freshmen. Return ONLY valid JSON."
    out = call_claude(prompt, None, max_tokens=2500, system=system, model=WRITER_MODEL, exempt=True)
    sp = out.get("spotlight") or {}
    if sp.get("name"):
        fn = sp["name"].split()[0].lower()
        earned = any(p["name"].split()[0].lower() == fn and any((m.get("date") or "") >= wk_ago and m.get("result") for m in p.get("recent_matches", [])) for p in players)
        if not earned: out["spotlight"] = None
    return out


# ---------------------------------------------------------------- normalize model output
def as_text(v):
    if v is None or isinstance(v, str): return v
    if isinstance(v, (int, float)): return str(v)
    if isinstance(v, dict): return ", ".join(str(x) for x in v.values() if x not in (None, "", [])) or None
    if isinstance(v, list): return ", ".join(as_text(x) or "" for x in v).strip(", ") or None
    return str(v)


def fix_result(r):
    """W/L first, then HER team's sets: a loss can't be 'L 3-0'."""
    m = re.match(r"\s*([WL])\D*(\d)\s*-\s*(\d)", r or "", re.I)
    if not m: return r
    wl, a, b = m.group(1).upper(), int(m.group(2)), int(m.group(3))
    if (wl == "W" and a < b) or (wl == "L" and a > b): a, b = b, a
    return f"{wl} {a}-{b}"


def clean_match(m):
    if not isinstance(m, dict): return None
    out = {k: as_text(m.get(k)) for k in ("date", "time", "time_ct", "opponent", "home_away", "location", "result", "box_url", "player_line", "stream_name", "stream_url")}
    if out["home_away"]: out["home_away"] = out["home_away"].lower().strip()
    if out["result"]: out["result"] = fix_result(out["result"])
    if out["date"]: out["date"] = out["date"][:10]
    return out if out["date"] and out["opponent"] else None


def clean_matches(lst):
    return [x for x in (clean_match(m) for m in (lst or [])) if x]


# ---------------------------------------------------------------- derived data
def opp_key(o):
    """'UT-Martin' == 'UT Martin' == 'ut martin'."""
    return re.sub(r"[^a-z0-9]", "", (o or "").lower())


def same_opp(a, b):
    """'Michigan St.' ~ 'Michigan State' (one is a prefix of the other); 'Minnesota Crookston' !~ 'Minnesota Duluth'."""
    ka, kb = opp_key(a), opp_key(b)
    return bool(ka and kb) and (ka == kb or ka.startswith(kb) or kb.startswith(ka))


def merge_matches(old, new):
    seen = {}
    for m in (old or []) + (new or []):
        k = (m.get("date"), opp_key(m.get("opponent")))
        seen[k] = {**seen.get(k, {}), **{a: b for a, b in m.items() if b is not None}}
    return sorted(seen.values(), key=lambda m: m.get("date") or "", reverse=True)


def season_stats_list(x):
    if not x or x.get("sp") is None: return []
    sp = x.get("sp") or 0
    per = lambda n: f"{(n or 0) / sp:.2f}" if sp else "–"
    if (x.get("a") or 0) > (x.get("k") or 0):
        return [{"label": "Assists", "value": x.get("a")}, {"label": "Assists/set", "value": per(x.get("a"))}, {"label": "Digs", "value": x.get("dig")}, {"label": "Aces", "value": x.get("sa")}]
    if (x.get("dig") or 0) > 2 * (x.get("k") or 0):
        return [{"label": "Digs", "value": x.get("dig")}, {"label": "Digs/set", "value": per(x.get("dig"))}, {"label": "Aces", "value": x.get("sa")}]
    hit = f"{((x.get('k') or 0) - (x.get('e') or 0)) / x['ta']:.3f}".lstrip("0") if x.get("ta") else "–"
    return [{"label": "Kills", "value": x.get("k")}, {"label": "Kills/set", "value": per(x.get("k"))}, {"label": "Hitting %", "value": hit}, {"label": "Blocks", "value": (x.get("bs") or 0) + (x.get("ba") or 0)}]


def detect_milestones(p, prev_stats, new_stats, today, prev_highs, seeding=False):
    out, highs = [], dict(prev_highs or {})
    results = [m for m in (p.get("recent_matches") or []) if m.get("result") and m.get("date")]
    newest = results[0]["date"] if results else today.isoformat()          # firsts happen in matches, not on run days
    earliest = min((m["date"] for m in results), default=today.isoformat())
    if prev_stats and new_stats:
        for key, (name, thresholds) in MILESTONES.items():
            before, after = prev_stats.get(key) or 0, new_stats.get(key) or 0
            for t in thresholds:
                if before < t <= after:
                    text = f"First college {name}" if t == 1 else f"{t} college {name}s"
                    if seeding: out.append({"date": earliest, "player_id": p["id"], "text": text + " (earlier this season)", "approx": True, "v": 2})
                    else: out.append({"date": newest, "player_id": p["id"], "text": text, "v": 2})
    last_seen = highs.get("_last", "")                                      # career highs: walk matches oldest -> newest, only new ones
    for m in sorted(results, key=lambda m: m["date"]):
        for n, word in re.findall(r"(\d+)\s+(kills?|assists?|digs?|aces?|blocks?)", m.get("player_line") or "", re.I):
            w, n = word.lower().rstrip("s"), int(n)
            if n > highs.get(w, 0):
                if not seeding and m["date"] > last_seen and highs.get(w, 0) > 0 and n >= 3:
                    out.append({"date": m["date"], "player_id": p["id"], "text": f"Career high {n} {w}s vs {m.get('opponent', '')}".strip(), "v": 2})
                highs[w] = n
    if results: highs["_last"] = max(highs.get("_last", ""), results[0]["date"])
    return out, highs


def compute_reunions(players):
    reunions, seen = [], set()
    for a in players:
        if a.get("status") != "playing": continue
        for m in (a.get("upcoming") or []) + (a.get("recent_matches") or []):
            opp = (m.get("opponent") or "").lower()
            for b in players:
                if b["id"] == a["id"] or b.get("status") != "playing" or b["school"] == a["school"]: continue
                if any(t and t in opp for t in (b["school_short"].lower(), (b.get("team_name") or "").lower())):
                    key = (m.get("date"), tuple(sorted((a["id"], b["id"]))))
                    if key in seen: continue
                    seen.add(key)
                    host, guest = (a, b) if m.get("home_away") == "home" else (b, a)
                    label = f"{host['school_short']} hosts {guest['school_short']}" if m.get("home_away") in ("home", "away") else f"{a['school_short']} vs {b['school_short']}"
                    reunions.append({"date": m.get("date"), "a": a["id"], "b": b["id"], "label": label,
                                     "stream_name": m.get("stream_name"), "stream_url": m.get("stream_url")})
    return sorted(reunions, key=lambda r: r["date"] or "")


# ---------------------------------------------------------------- main
def process_player(c, cfg, prev, prev_players, today, now, flags):
    """All the work for one girl. Returns (entry, milestones, buzz_items, reseeded, failed)."""
    first_run, reseed, schedules_day, news_day, no_news_yet = (flags[k] for k in ("first_run", "reseed", "schedules_day", "news_day", "no_news_yet"))
    old = prev_players.get(c["id"], {}) if not first_run else {}
    p = {**old, **c, "college_season": cfg["college_season"]}
    p.pop("fetch_note", None); p.pop("disambiguation", None)
    for k in ("recent_matches", "upcoming", "season_stats"): p.setdefault(k, [])
    p["recent_matches"], p["upcoming"] = clean_matches(p["recent_matches"]), clean_matches(p["upcoming"])
    for k in ("stats", "team_record", "photo_url"): p.setdefault(k, None)
    p.setdefault("socials", {"instagram": None, "x": None}); p["stale"] = False
    miles, items, reseeded, failed = [], [], False, False
    if c.get("status") != "playing":
        p.update({"recent_matches": [], "upcoming": [], "stats": None, "season_stats": []})
        return p, miles, items, reseeded, failed

    log(f"{p['name']} ({p['school_short']})")
    p.pop("error", None)
    try:
        optional = spent() < 0.7            # past 70% of budget: stats only
        if (not old.get("jersey") or not old.get("photo_url")) and (not old.get("profile_tried") or today.weekday() == 0) and optional and (first_run or today.weekday() == 0 or not old.get("profile_tried")):
            p["profile_tried"] = True
            log("  profile"); prof = research_profile(c)
            p.update({k: v for k, v in prof.items() if v and not (k == "position" and c.get("position"))})
        if c.get("position"): p["position"] = c["position"]
        need = [(m["date"], m["opponent"]) for m in old.get("recent_matches", []) if m.get("result") and not m.get("player_line")]
        log("  daily"); d = research_daily({**p, **c}, today, need)
        if d.get("_notes"): log("    " + "; ".join(d["_notes"])); p["_notes"] = d["_notes"]
        prev_stats = old.get("stats")
        if isinstance(d.get("team_record"), dict): p["team_record"] = {**(p.get("team_record") or {}), **{k: as_text(v) for k, v in d["team_record"].items() if v}}
        if isinstance(d.get("stats"), dict):
            new_stats = {k: (int(float(d["stats"][k])) if str(d["stats"].get(k, "")).replace(".", "").isdigit() else None) for k in STAT_KEYS}
            if any(v is not None for v in new_stats.values()): p["stats"] = new_stats; p["stats_source"] = d.get("stats_source")
            elif old.get("stats"): log(f"  stats came back empty — keeping last good line for {p['name']}")
        new_results = clean_matches(d.get("results"))
        if d.get("results_authoritative"):
            for m in new_results:
                o = next((c_ for c_ in old.get("recent_matches", []) if c_.get("date") == m.get("date") and same_opp(c_.get("opponent"), m.get("opponent"))), None)
                if o:
                    if not m.get("player_line") and o.get("player_line"): m["player_line"] = o["player_line"]
                    if not m.get("box_url") and o.get("box_url"): m["box_url"] = o["box_url"]
            p["recent_matches"] = sorted(new_results, key=lambda m: m.get("date") or "", reverse=True)
        else:
            p["recent_matches"] = merge_matches(old.get("recent_matches"), new_results)
        if d.get("blurb") and not (isinstance(d.get("stats"), dict) and not any(v is not None for v in new_stats.values()) and old.get("blurb")):
            p["blurb"] = as_text(d["blurb"])
        played = {(m.get("date"), opp_key(m.get("opponent"))) for m in p["recent_matches"] if m.get("result")}
        p["upcoming"] = [m for m in p["upcoming"] if m.get("date") and m["date"] >= today.isoformat() and (m["date"], opp_key(m.get("opponent"))) not in played]
        has_firsts = any(m.get("player_id") == p["id"] and "First college" in m.get("text", "") for m in prev.get("milestones", []))
        seeding = (first_run or reseed or not has_firsts) and bool(p["stats"])
        if seeding: prev_stats = {k: 0 for k in STAT_KEYS}; reseeded = True
        miles, highs = detect_milestones(p, prev_stats, p["stats"], today, old.get("_highs") if not seeding else None, seeding)
        p["_highs"] = highs
        if (schedules_day or not p["upcoming"]) and spent() < 0.7:
            log("  weekly"); w = research_weekly({**c, **p}, today)
            if w.get("schedule"):
                p["upcoming"] = sorted([m for m in clean_matches(w["schedule"]) if m.get("date") and (m["date"], (m.get("opponent") or "").lower()) not in played], key=lambda m: m["date"])
            if w.get("standing"): p["team_record"] = {**(p.get("team_record") or {}), "standing": as_text(w["standing"]), "standings_url": as_text(w.get("standings_url"))}
            if w.get("socials") and any((w["socials"] or {}).values()): p["socials"] = w["socials"]
        if news_day and spent() < 0.7:
            log("  news"); n = research_news(c, today, first_run or no_news_yet)
            junk = re.compile(r"roster|\bcommit|schedule\b|/sports/womens-volleyball/?$|sportsrecruits|topflightvbc", re.I)
            items = [{**it, "player_id": p["id"]} for it in n.get("items", [])
                     if it.get("url") and it.get("title") and not junk.search(it["title"] + " " + it["url"])]
        p["fetched_at"] = now.isoformat(timespec="minutes")
    except Exception as e:
        failed = True; log(f"  FAILED {p['name']}: {str(e)[:160]}"); p["stale"] = True; p["error"] = str(e)[:200]
    if any("no schedule source answered" in n or "fetch failed" in n for n in (p.get("_notes") or [])) and not p.get("stats_source"):
        # the school's site refused every route (CAPTCHA / bot wall): show unknown, never stale zeros, and say why
        p["stats"] = None; p["stats_source"] = "unavailable"
        link = p.get("profile_url") or p.get("site") or ""
        p["blurb"] = (f"{p['name'].split()[0]}'s school site doesn't allow automated stat updates, so her numbers aren't tracked here yet. "
                      + (f"Her player page has the latest: {link}" if link else ""))
    p["matches_played"], p["sets_played"] = (p["stats"] or {}).get("mp"), (p["stats"] or {}).get("sp")
    p["season_stats"] = season_stats_list(p["stats"])
    return p, miles, items, reseeded, failed


def assemble(cfg, prev, players_by_id, milestones, reunions, buzz, summary, now, today, failures, note):
    ordered = [players_by_id.get(c["id"]) or {**c, "stale": True, "recent_matches": [], "upcoming": [], "season_stats": [], "stats": None, "team_record": None}
               for c in cfg["players"]]
    for p in ordered: p.pop("fetch_note", None); p.pop("disambiguation", None)
    return {"team": cfg["team"], "club": cfg["club"], "club_season": cfg["club_season"], "college_season": cfg["college_season"],
            "updated_at": now.isoformat(timespec="minutes"), "updated_label": now.strftime("%A %-I:%M %p CT"),
            "summary": summary, "players": ordered, "milestones": milestones, "reunions": reunions, "buzz": buzz,
            "run": {"date": today.isoformat(), "failures": failures, "model": MODEL, "note": note,
                    "input_tokens": USAGE["in"], "output_tokens": USAGE["out"], "calls": USAGE["calls"]}}


def dry_report(data):
    """Per-girl verification: did code get her stats, do parsed results reconcile with the posted record, is her schedule there, which sources answered."""
    rows = []
    for p in data["players"]:
        if p.get("status") != "playing": continue
        rm = p.get("recent_matches") or []; st = p.get("stats") or {}
        wl = [m["result"][0] for m in rm if m.get("result")]
        rec = (p.get("team_record") or {}).get("overall")
        rows.append({"player": p["name"], "record": rec, "parsed_wl": f"{wl.count('W')}-{wl.count('L')}", "reconciles": rec == f"{wl.count('W')}-{wl.count('L')}",
                     "results": len(rm), "box_links": sum(1 for m in rm if m.get("box_url")), "upcoming": len(p.get("upcoming") or []),
                     "next": (p.get("upcoming") or [{}])[0].get("date"), "stats_source": p.get("stats_source"),
                     "sp": st.get("sp"), "k": st.get("k"), "a": st.get("a"), "dig": st.get("dig"), "photo": bool(p.get("photo_url")),
                     "error": p.get("error"), "notes": p.get("_notes")})
    ok = all(r["reconciles"] and r["stats_source"] for r in rows)
    return {"generated": data["updated_at"], "all_green": ok, "players": rows}


def save(data):
    if DRY_RUN:
        rep_ = dry_report(data)
        json.dump(rep_, open(os.path.join(HERE, "dry-run-report.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
        return
    tmp = DATA_FILE + ".tmp"
    json.dump(data, open(tmp, "w", encoding="utf-8"), ensure_ascii=False, indent=1); os.replace(tmp, DATA_FILE)


def main():
    if not API_KEY and not DRY_RUN: sys.exit("ANTHROPIC_API_KEY is not set")
    cfg = json.load(open(PLAYERS_FILE, encoding="utf-8"))
    tz = ZoneInfo(cfg.get("home_tz", "America/Chicago"))
    now = datetime.now(tz); today = now.date()
    prev = {}
    if os.path.exists(DATA_FILE):
        try: prev = json.load(open(DATA_FILE, encoding="utf-8"))
        except Exception: prev = {}
    prev_players = {p["id"]: p for p in prev.get("players", [])}
    first_run = not prev_players or prev.get("run", {}).get("model") in (None, "bootstrap", "sample")
    no_news_yet = not any(b.get("kind") in ("news", "coach") for b in prev.get("buzz", []))
    forced = os.environ.get("TRACKER_FULL") == "1" or DRY_RUN
    flags = {"first_run": first_run,
             "reseed": not any(m.get("v") == 2 for m in prev.get("milestones", [])),
             "schedules_day": forced or first_run or today.weekday() == 0,                    # schedules/standings: Monday (plus any girl missing hers)
             "news_day": forced or first_run or no_news_yet or today.weekday() == 0,         # news: Monday, or whenever Buzz has none yet
             "no_news_yet": no_news_yet}
    summary_age = (today - datetime.fromisoformat(prev["run"]["date"]).date()).days if prev.get("run", {}).get("date") and prev.get("summary") else 99
    writing_day = forced or first_run or no_news_yet or today.weekday() in (0, 3) or summary_age > 4       # recap Monday, preview Thursday, or the note is stale
    piece_kind = "recap" if today.weekday() in (0, 1, 5, 6) else "preview"
    log(f"Run {today} · {flags} · writing={writing_day} · budget={TOKEN_BUDGET:,} tokens · workers={WORKERS} · model={MODEL}")

    players_by_id = {pid: p for pid, p in prev_players.items()}          # start from last good data; replace as girls finish
    milestones = list(prev.get("milestones") or [])
    buzz = [b for b in (prev.get("buzz") or []) if not (b.get("kind") == "recap" and (b.get("date") or "") < "2026-09-16")]   # pre-rebuild recaps had bad numbers
    summary = prev.get("summary")
    failures, done = 0, 0
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        futs = {pool.submit(process_player, c, cfg, prev, prev_players, today, now, flags): c for c in cfg["players"]}
        for fut in as_completed(futs):
            c = futs[fut]
            try: p, miles, items, reseeded, failed = fut.result()
            except Exception as e:
                failed, p, miles, items, reseeded = True, {**prev_players.get(c["id"], c), "stale": True, "error": str(e)[:200]}, [], [], False
                log(f"  FAILED {c['name']}: {str(e)[:160]}")
            failures += int(failed); done += 1
            players_by_id[c["id"]] = p
            if reseeded: milestones = [m for m in milestones if m.get("player_id") != c["id"]]
            milestones += miles
            have = {b.get("url") for b in buzz if b.get("url")}
            buzz += [b for b in items if b["url"] not in have]
            snapshot = assemble(cfg, prev, players_by_id, sorted(milestones, key=lambda m: m["date"], reverse=True)[:80],
                                prev.get("reunions") or [], buzz, summary, now, today, failures, f"in progress · {done}/{len(cfg['players'])} done")
            save(snapshot)                                                   # a timeout or a billing stop keeps everything finished so far
            log(f"  saved · {done}/{len(cfg['players'])} · {spent():.0%} of budget")

    players = [players_by_id[c["id"]] for c in cfg["players"] if c["id"] in players_by_id]
    milestones = sorted(milestones, key=lambda m: m["date"], reverse=True)[:80]
    reunions = compute_reunions(players)
    if writing_day and not DRY_RUN:
        try:
            log(f"Writing the {piece_kind}")
            week_miles = [m for m in milestones if m["date"] >= (today - timedelta(days=7)).isoformat()]
            piece = write_piece(piece_kind, players, today, week_miles, [r for r in reunions if (r["date"] or "") >= today.isoformat()])
            post = {"date": today.isoformat(), "kind": "recap", "piece": piece_kind, "title": piece["title"], "body": piece["body"], "spotlight": piece.get("spotlight")}
            buzz = [b for b in buzz if not (b.get("kind") == "recap" and b.get("date") == today.isoformat())] + [post]
            summary = {"headline": piece["title"], "body": piece["body"], "spotlight": piece.get("spotlight"), "piece": piece_kind}
        except Exception as e:
            failures += 1; log(f"  writing FAILED: {e}")
    cutoff = (today - timedelta(days=75)).isoformat()
    buzz = sorted([b for b in buzz if (b.get("date") or "") >= cutoff], key=lambda b: b["date"], reverse=True)
    save(assemble(cfg, prev, players_by_id, milestones, reunions, buzz, summary, now, today, failures, "complete"))
    log(f"Done · {len(players)} players · {failures} failure(s) · {USAGE['calls']} calls · {USAGE['in']:,} in / {USAGE['out']:,} out tokens ({spent():.0%} of budget)")
    if DRY_RUN:
        r = dry_report(assemble(cfg, prev, players_by_id, milestones, reunions, buzz, summary, now, today, failures, "dry run"))
        log("DRY RUN REPORT · all_green=%s" % r["all_green"])
        for x in r["players"]: log(f"  {x['player']:20} record={x['record']} parsed={x['parsed_wl']} {'OK ' if x['reconciles'] else 'XX '} box={x['box_links']}/{x['results']} upcoming={x['upcoming']} stats={x['stats_source']} sp={x['sp']} k={x['k']} a={x['a']} dig={x['dig']} {('ERR '+str(x['error'])[:60]) if x['error'] else ''}")


if __name__ == "__main__":
    main()
