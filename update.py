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

import json, os, re, sys, time, threading, urllib.request, urllib.error
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

API_URL = "https://api.anthropic.com/v1/messages"
API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
MODEL = os.environ.get("TRACKER_MODEL", "claude-sonnet-5")      # research calls (Haiku proved too sloppy: merged rows, missed box scores)
WRITER_MODEL = os.environ.get("TRACKER_WRITER_MODEL", "claude-sonnet-5")   # Monday recap / Thursday preview only
HERE = os.path.dirname(os.path.abspath(__file__))
PLAYERS_FILE, DATA_FILE = os.path.join(HERE, "players.json"), os.path.join(HERE, "data.json")

TOKEN_BUDGET = int(os.environ.get("TRACKER_TOKEN_BUDGET", "1200000"))   # input tokens per run; optional work stops at 70%, everything at 100%
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


def tools_for(model, fetch_cap=8000):
    """Haiku lacks the newer fetch tool's dynamic filtering, so it gets the basic fetch with a tighter cap."""
    if "haiku" in (model or MODEL):
        return [{"type": "web_fetch_20250910", "name": "web_fetch", "max_uses": 6, "max_content_tokens": min(fetch_cap, 10000)}, SEARCH]
    return [{"type": "web_fetch_20260318", "name": "web_fetch", "max_uses": 6, "max_content_tokens": fetch_cap, "use_cache": False}, SEARCH]
SYSTEM = ("You maintain a small, family-friendly tracker of college volleyball players for their parents. "
          "Research only from the official pages you are given (and web_search when told to). Return ONLY valid JSON "
          "matching the requested structure — no prose, no code fences. Accuracy beats completeness: never invent a "
          "number, score, date, or link; use null when something is not on the page. Dates are ISO YYYY-MM-DD.")


def log(m): print(f"[{datetime.now().strftime('%H:%M:%S')}] {m}", flush=True)


def spent(): return USAGE["in"] / TOKEN_BUDGET


class BudgetExceeded(RuntimeError): pass


# ---------------------------------------------------------------- API
def call_claude(prompt, tools=None, max_tokens=5000, system=SYSTEM, model=None):
    """One retry on any failure (network, HTTP, bad JSON). Refuses to start once the run budget is spent."""
    if spent() >= 1.0: raise BudgetExceeded(f"run token budget spent ({USAGE['in']:,} input tokens)")
    for attempt in (1, 2):
        try:
            return _call(prompt, tools, max_tokens, system, model)
        except Exception as e:
            if attempt == 2: raise
            log(f"    retrying after: {str(e)[:120]}"); time.sleep(6)


def _call(prompt, tools, max_tokens, system, model):
    convo = [{"role": "user", "content": prompt}]; asked_again = False
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


# ---------------------------------------------------------------- research calls
def research_daily(p, today, need_lines=()):
    since = (today - timedelta(days=14)).isoformat()
    need = ", ".join(f"{d} vs {o}" for d, o in need_lines[:2]) or "none yet"
    prompt = f"""Today is {today.isoformat()}. Fall {p['college_season']} season. Player: {p['name']}, freshman at {p['school']} ({p['division']}). Position: {p.get('position') or p['club_position']}.
{pages(p)}

Identify her row by jersey number{(' #' + str(p['jersey'])) if p.get('jersey') else ''}, class (freshman), and hometown ({p.get('hometown_hs') or 'see roster'}) — not by name alone.
{p.get('disambiguation', '')}
Never add two rows together. If two rows could be her and you cannot tell which, return null for every stat and say so in the blurb.

Collect:
1. team_record: overall and conference W-L from the schedule page.
2. stats: HER season totals, as integers (null if the column is not published). The stats page has a "View PDF" link to the season cumulative PDF —
   open that PDF; it is the authoritative table (the HTML table sometimes mislabels rows). Match her row by jersey number AND last name.
   mp matches played, sp sets played, k kills, e attack errors, ta total attacks, a assists, bhe ball-handling errors,
   sa service aces, se service errors, srv serve attempts, dig digs, re reception errors, bs block solos, ba block assists, be block errors.
   If she does not appear in the table at all, set every field to 0 except srv/re/ta which may be null — but first confirm by searching the page text for her last name; do not report zeros because a table was truncated.
3. results: team matches from {since} through {today.isoformat()} with a final score, most recent first. For each: date, opponent, home_away (home/away/neutral),
   result ("W 3-1" / "L 0-3"), box_url (the box score link from the schedule page), player_line.
   player_line = her numbers from the box score in plain words ("7 kills, 3 blocks, 2 digs" / "24 assists, 6 digs" / "did not play"). The individual lines are on the
   box score page under the "Individual" tab, listed by team with jersey numbers — find the row with her number. Open the box score for the matches that still need a
   line (at most 2 fetches, newest first): {need} plus any newer match. Leave null only if you truly could not open it.
4. blurb: 2-3 sentences for her parents: what she and the team did lately, whether she is getting court time, what is next. Warm, plain, factual.

Return ONLY:
{{"team_record": {{"overall": "W-L", "conference": "W-L or null"}},
 "stats": {{"mp":0,"sp":0,"k":0,"e":0,"ta":0,"a":0,"bhe":0,"sa":0,"se":0,"srv":null,"dig":0,"re":null,"bs":0,"ba":0,"be":0}},
 "results": [{{"date":"YYYY-MM-DD","opponent":"","home_away":"home","result":"W 3-1","box_url":null,"player_line":null}}],
 "blurb": ""}}"""
    return call_claude(prompt, tools_for(p.get("model")), model=p.get("model"))


def research_weekly(p, today):
    prompt = f"""Today is {today.isoformat()}. Fall {p['college_season']} season. Team: {p['school']} {p.get('team_name', '')} volleyball ({p['division']}, {p['conference']}). Player of interest: {p['name']}.
Schedule page: {p['schedule_url']}
{('Note: ' + p['fetch_note']) if p.get('fetch_note') else ''}

If the schedule page is long and gets truncated, work from what you received plus web_search for the rest; do not refetch the same URL more than twice.

Collect:
1. schedule: EVERY remaining match from {today.isoformat()} to the end of the regular season (and conference tournament dates if listed). For each: date, time exactly as listed,
   time_ct — the same time converted to US Central (the school is in {p['tz']}; Central is one hour behind Eastern), opponent, home_away, location (city, ST or venue),
   stream_name (ESPN+, FloSports, NSIC Network, Hudl, school stream, etc. as labeled) and stream_url (the actual link on the schedule page, else null).
2. standing: the team's current place in its conference standings, formatted like "3rd of 11 OVC" (find the standings page from the schedule page or the conference site; web_search if needed), and standings_url.
3. socials: the official program Instagram and X handles (no @), null if not found.

Return ONLY:
{{"schedule": [{{"date":"YYYY-MM-DD","time":"6:00 PM ET","time_ct":"5:00 PM CT","opponent":"","home_away":"home","location":null,"stream_name":null,"stream_url":null}}],
 "standing": "3rd of 11 OVC", "standings_url": null, "socials": {{"instagram": null, "x": null}}}}"""
    m = p.get("model") or STRONG_MODEL
    return call_claude(prompt, tools_for(m, fetch_cap=10000), max_tokens=6000, model=m)


def research_profile(p):
    prompt = f"""Player: {p['name']}, {p['school']} volleyball, freshman. Roster/bio page: {p['roster_url']}
{('Note: ' + p['fetch_note']) if p.get('fetch_note') else ''}
From the roster page (open her bio if linked): jersey, position as listed, class_year, height, hometown_hs ("Hometown, ST / High School"), bio_url,
and photo_url — the direct URL of her roster headshot image (an https link to a .jpg/.jpeg/.png/.webp or a Sidearm image URL); null if none.
Return ONLY: {{"jersey":null,"position":null,"class_year":null,"height":null,"hometown_hs":null,"bio_url":null,"photo_url":null}}"""
    return call_claude(prompt, tools_for(p.get("model")), max_tokens=1200, model=p.get("model"))


def research_news(p, today, first_run=False):
    since = (today - timedelta(days=21 if first_run else 8)).isoformat()
    news_url = p["site"].rstrip("/") + "/news"
    prompt = f"""Today is {today.isoformat()}. Use web_search only (2 searches, e.g. "{p['name']} volleyball", "{p['school_short']} volleyball {p['name'].split()[-1]}", "{p['school_short']} volleyball coach"). Find up to 5 items published since {since} that either
(a) mention {p['name']} by name — school match recaps and features, signing/roster announcements, local papers (Daily Herald, Kane County Reporter, Northwest Herald, Elgin Courier-News), conference weekly honors — or
(b) concern the {p['school']} volleyball coaching staff — hires, departures, contract extensions, awards, suspensions — head coach or assistants.
Skip recaps that do not mention her and skip anything older than {since}. Return an empty list if nothing qualifies.
For a recap that mentions her, put her line or the quote in "note" (one sentence). Return up to 6 items.
Return ONLY: {{"items": [{{"date":"YYYY-MM-DD","kind":"news" or "coach","title":"","source":"publication name","url":"https://...","note":null}}]}}"""
    return call_claude(prompt, [{**SEARCH, "max_uses": 2}], max_tokens=1500, model=STRONG_MODEL)


def write_piece(kind, players, today, milestones, reunions):
    wk_ago, wk_ahead = (today - timedelta(days=7)).isoformat(), (today + timedelta(days=7)).isoformat()
    compact = []
    for p in players:
        if p.get("status") != "playing": continue
        compact.append({k: p.get(k) for k in ("name", "school_short", "division", "position", "team_record", "stats", "blurb", "stale")}
                       | {"results_last_7": [m for m in p.get("recent_matches", []) if (m.get("date") or "") >= wk_ago],
                          "next_7": [m for m in p.get("upcoming", []) if today.isoformat() <= (m.get("date") or "") <= wk_ahead]})
    if kind == "recap":
        ask = ("Write the MONDAY WEEKEND RECAP. Paragraph 1: the weekend in two or three sentences — who stood out, any milestone, any big team result. "
               "Then ONE short paragraph (1-2 sentences) for EACH girl who had a match this week, in this form: her first name, the results, her line, one human note "
               "(e.g. 'Anna — Morehead State split at Marshall (L 1-3, W 3-2); 5 kills and 4 blocks Saturday, her best block night yet.'). "
               "Skip girls with no match this week. Say plainly if a girl did not see the court.")
    else:
        ask = ("Write the THURSDAY WEEKEND PREVIEW. Paragraph 1: the weekend ahead in two or three sentences — the biggest matches, conference openers, anything at stake. "
               "Then ONE short paragraph (1-2 sentences) for EACH girl with a match this weekend: first name, opponent(s), day and Central time, stream, and why it matters "
               "(e.g. 'Kylie — Arkansas Tech at Southern Nazarene, Fri 6 PM CT on FloSports; a win keeps the Suns alone atop the GAC.').")
    prompt = f"""Today is {today.isoformat()}. Data for the girls (JSON): {json.dumps(compact, ensure_ascii=False)}
Milestones this week: {json.dumps(milestones)}
Reunions coming up: {json.dumps(reunions[:3])}

{ask}
Rules: title under 12 words, no "Recap:" prefix. body = the paragraphs described above (the intro plus one per girl), each under 60 words, plain warm language, no hype, no bullet points, no jargon.
Ignore anyone marked stale. spotlight = one girl with the best week and a one-sentence reason, or null.
Return ONLY: {{"title":"", "body":["",""], "spotlight": {{"name":"First Last","note":""}} }}"""
    system = "You write short, warm notes for a group of volleyball moms whose daughters played club together and are now college freshmen. Return ONLY valid JSON."
    return call_claude(prompt, None, max_tokens=2500, system=system, model=WRITER_MODEL)


# ---------------------------------------------------------------- normalize model output
def as_text(v):
    if v is None or isinstance(v, str): return v
    if isinstance(v, (int, float)): return str(v)
    if isinstance(v, dict): return ", ".join(str(x) for x in v.values() if x not in (None, "", [])) or None
    if isinstance(v, list): return ", ".join(as_text(x) or "" for x in v).strip(", ") or None
    return str(v)


def clean_match(m):
    if not isinstance(m, dict): return None
    out = {k: as_text(m.get(k)) for k in ("date", "time", "time_ct", "opponent", "home_away", "location", "result", "box_url", "player_line", "stream_name", "stream_url")}
    if out["home_away"]: out["home_away"] = out["home_away"].lower().strip()
    if out["date"]: out["date"] = out["date"][:10]
    return out if out["date"] and out["opponent"] else None


def clean_matches(lst):
    return [x for x in (clean_match(m) for m in (lst or [])) if x]


# ---------------------------------------------------------------- derived data
def merge_matches(old, new):
    seen = {}
    for m in (old or []) + (new or []):
        k = (m.get("date"), (m.get("opponent") or "").lower())
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
    p.pop("fetch_note", None); p.pop("disambiguation", None); p.pop("position_note", None)
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
        log("  daily"); d = research_daily({**c, **p}, today, need)
        prev_stats = old.get("stats")
        if isinstance(d.get("team_record"), dict): p["team_record"] = {**(p.get("team_record") or {}), **{k: as_text(v) for k, v in d["team_record"].items() if v}}
        if isinstance(d.get("stats"), dict): p["stats"] = {k: (int(float(d["stats"][k])) if str(d["stats"].get(k, "")).replace(".", "").isdigit() else None) for k in STAT_KEYS}
        p["recent_matches"] = merge_matches(old.get("recent_matches"), clean_matches(d.get("results")))
        if d.get("blurb"): p["blurb"] = as_text(d["blurb"])
        played = {(m.get("date"), (m.get("opponent") or "").lower()) for m in p["recent_matches"] if m.get("result")}
        p["upcoming"] = [m for m in p["upcoming"] if m.get("date") and m["date"] >= today.isoformat() and (m["date"], (m.get("opponent") or "").lower()) not in played]
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
            items = [{**it, "player_id": p["id"]} for it in n.get("items", []) if it.get("url") and it.get("title")]
        p["fetched_at"] = now.isoformat(timespec="minutes")
    except Exception as e:
        failed = True; log(f"  FAILED {p['name']}: {str(e)[:160]}"); p["stale"] = True; p["error"] = str(e)[:200]
    p["matches_played"], p["sets_played"] = (p["stats"] or {}).get("mp"), (p["stats"] or {}).get("sp")
    p["season_stats"] = season_stats_list(p["stats"])
    return p, miles, items, reseeded, failed


def assemble(cfg, prev, players_by_id, milestones, reunions, buzz, summary, now, today, failures, note):
    ordered = [players_by_id.get(c["id"]) or {**c, "stale": True, "recent_matches": [], "upcoming": [], "season_stats": [], "stats": None, "team_record": None}
               for c in cfg["players"]]
    for p in ordered: p.pop("fetch_note", None); p.pop("disambiguation", None); p.pop("position_note", None)
    return {"team": cfg["team"], "club": cfg["club"], "club_season": cfg["club_season"], "college_season": cfg["college_season"],
            "updated_at": now.isoformat(timespec="minutes"), "updated_label": now.strftime("%A %-I:%M %p CT"),
            "summary": summary, "players": ordered, "milestones": milestones, "reunions": reunions, "buzz": buzz,
            "run": {"date": today.isoformat(), "failures": failures, "model": MODEL, "note": note,
                    "input_tokens": USAGE["in"], "output_tokens": USAGE["out"], "calls": USAGE["calls"]}}


def save(data):
    tmp = DATA_FILE + ".tmp"
    json.dump(data, open(tmp, "w", encoding="utf-8"), ensure_ascii=False, indent=1); os.replace(tmp, DATA_FILE)


def main():
    if not API_KEY: sys.exit("ANTHROPIC_API_KEY is not set")
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
    forced = os.environ.get("TRACKER_FULL") == "1"
    flags = {"first_run": first_run,
             "reseed": not any(m.get("v") == 2 for m in prev.get("milestones", [])),
             "schedules_day": forced or first_run or today.weekday() == 0,                    # schedules/standings: Monday (plus any girl missing hers)
             "news_day": forced or first_run or today.weekday() == 0,                        # news: Monday only (a mid-week manual run skips it)
             "no_news_yet": no_news_yet}
    summary_age = (today - datetime.fromisoformat(prev["run"]["date"]).date()).days if prev.get("run", {}).get("date") and prev.get("summary") else 99
    writing_day = forced or first_run or today.weekday() in (0, 3) or summary_age > 4       # recap Monday, preview Thursday, or the note is stale
    piece_kind = "recap" if today.weekday() in (0, 1, 5, 6) else "preview"
    log(f"Run {today} · {flags} · writing={writing_day} · budget={TOKEN_BUDGET:,} tokens · workers={WORKERS} · model={MODEL}")

    players_by_id = {pid: p for pid, p in prev_players.items()}          # start from last good data; replace as girls finish
    milestones = list(prev.get("milestones") or [])
    buzz = list(prev.get("buzz") or [])
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
    if writing_day and spent() < 1.0:
        try:
            log(f"Writing the {piece_kind}")
            week_miles = [m for m in milestones if m["date"] >= (today - timedelta(days=7)).isoformat()]
            piece = write_piece(piece_kind, players, today, week_miles, [r for r in reunions if (r["date"] or "") >= today.isoformat()])
            post = {"date": today.isoformat(), "kind": "recap", "title": piece["title"], "body": piece["body"], "spotlight": piece.get("spotlight")}
            buzz = [b for b in buzz if not (b.get("kind") == "recap" and b.get("date") == today.isoformat())] + [post]
            summary = {"headline": piece["title"], "body": piece["body"], "spotlight": piece.get("spotlight")}
        except Exception as e:
            failures += 1; log(f"  writing FAILED: {e}")
    cutoff = (today - timedelta(days=75)).isoformat()
    buzz = sorted([b for b in buzz if (b.get("date") or "") >= cutoff], key=lambda b: b["date"], reverse=True)
    save(assemble(cfg, prev, players_by_id, milestones, reunions, buzz, summary, now, today, failures, "complete"))
    log(f"Done · {len(players)} players · {failures} failure(s) · {USAGE['calls']} calls · {USAGE['in']:,} in / {USAGE['out']:,} out tokens ({spent():.0%} of budget)")


if __name__ == "__main__":
    main()
