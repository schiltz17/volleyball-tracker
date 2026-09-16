#!/usr/bin/env python3
"""
Top Flight 18 Elite — Where Are They Now · updater v2

Runs every morning on GitHub Actions. Reads players.json + the previous data.json,
asks Claude (web fetch + web search) for what changed, writes data.json for the app.

Cadence
  Mon, Thu, Sat stats, team record, results (with box score and her line) for each girl
  Mon + Thu     full-season schedules (times in Central), standings, program socials,
                and the written piece: Monday recap / Thursday preview
  Mon           news + coach items
  as needed     profile facts (photo, jersey, class, height, hometown)

Env:  ANTHROPIC_API_KEY (required) · TRACKER_MODEL (default claude-haiku-4-5-20251001) · TRACKER_WRITER_MODEL (default claude-sonnet-5)
      TRACKER_FULL=1 forces the Mon/Thu work to run today (the first run does this automatically)
"""

import json, os, re, sys, time, urllib.request, urllib.error
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

API_URL = "https://api.anthropic.com/v1/messages"
API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
MODEL = os.environ.get("TRACKER_MODEL", "claude-haiku-4-5-20251001")      # research calls
WRITER_MODEL = os.environ.get("TRACKER_WRITER_MODEL", "claude-sonnet-5")   # Monday recap / Thursday preview only
HERE = os.path.dirname(os.path.abspath(__file__))
PLAYERS_FILE, DATA_FILE = os.path.join(HERE, "players.json"), os.path.join(HERE, "data.json")

STAT_KEYS = ["mp", "sp", "k", "e", "ta", "a", "bhe", "sa", "se", "srv", "dig", "re", "bs", "ba", "be"]
MILESTONES = {"k": ("kill", [1, 25, 50, 100, 150, 200, 300]), "a": ("assist", [1, 50, 100, 200, 300, 500, 750]),
              "dig": ("dig", [1, 50, 100, 150, 200, 300, 400]), "sa": ("ace", [1, 10, 25, 50]),
              "bs": ("solo block", [1]), "ba": ("block assist", [1])}

# Haiku does not support the newer fetch tool's dynamic filtering, so it gets the basic fetch with a tighter cap.
FETCH = ({"type": "web_fetch_20250910", "name": "web_fetch", "max_uses": 6, "max_content_tokens": 10000} if "haiku" in MODEL
         else {"type": "web_fetch_20260318", "name": "web_fetch", "max_uses": 8, "max_content_tokens": 14000, "use_cache": False})
TOOLS = [FETCH, {"type": "web_search_20250305", "name": "web_search", "max_uses": 3}]
SYSTEM = ("You maintain a small, family-friendly tracker of college volleyball players for their parents. "
          "Research only from the official pages you are given (and web_search when told to). Return ONLY valid JSON "
          "matching the requested structure — no prose, no code fences. Accuracy beats completeness: never invent a "
          "number, score, date, or link; use null when something is not on the page. Dates are ISO YYYY-MM-DD.")


def log(m): print(f"[{datetime.now().strftime('%H:%M:%S')}] {m}", flush=True)


# ---------------------------------------------------------------- API
def call_claude(prompt, tools=None, max_tokens=5000, system=SYSTEM, model=None):
    convo = [{"role": "user", "content": prompt}]
    for _ in range(6):
        body = {"model": model or MODEL, "max_tokens": max_tokens, "system": system, "messages": convo}
        if tools: body["tools"] = tools
        req = urllib.request.Request(API_URL, data=json.dumps(body).encode(), method="POST", headers={
            "x-api-key": API_KEY, "anthropic-version": "2023-06-01", "content-type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=300) as r:
                data = json.loads(r.read())
        except urllib.error.HTTPError as e:
            raise RuntimeError(f"API HTTP {e.code}: {e.read().decode('utf-8', 'replace')[:600]}") from None
        if data.get("stop_reason") == "pause_turn":
            convo.append({"role": "assistant", "content": data["content"]}); continue
        text = "".join(b.get("text", "") for b in data.get("content", []) if b.get("type") == "text")
        u = data.get("usage", {}); st = u.get("server_tool_use", {})
        log(f"    tokens {u.get('input_tokens')}/{u.get('output_tokens')} fetch={st.get('web_fetch_requests', 0)} search={st.get('web_search_requests', 0)}")
        return parse_json(text)
    raise RuntimeError("too many pause_turn continuations")


def parse_json(text):
    text = text.strip().strip("`")
    if text.lower().startswith("json"): text = text[4:]
    s, e = text.find("{"), text.rfind("}")
    if s < 0: raise ValueError("no JSON in response: " + text[:200])
    return json.loads(text[s:e + 1])


def pages(p):
    note = ("\n" + p["fetch_note"]) if p.get("fetch_note") else ""
    return (f"Official pages:\n- Roster/bio: {p['roster_url']}\n- Schedule: {p['schedule_url']}\n- Stats: {p['stats_url']}"
            f"{note}\nIf a page fails or has no data, fall back to web_search (\"{p['name']} {p['school_short']} volleyball\").")


# ---------------------------------------------------------------- research calls
def research_daily(p, today):
    since = (today - timedelta(days=14)).isoformat()
    prompt = f"""Today is {today.isoformat()}. Fall {p['college_season']} season. Player: {p['name']}, freshman at {p['school']} ({p['division']}). Position: {p.get('position') or p['club_position']}.
{pages(p)}

Collect:
1. team_record: overall and conference W-L from the schedule page.
2. stats: HER season totals from the individual stats table, as integers (null if the column is not published):
   mp matches played, sp sets played, k kills, e attack errors, ta total attacks, a assists, bhe ball-handling errors,
   sa service aces, se service errors, srv serve attempts, dig digs, re reception errors, bs block solos, ba block assists, be block errors.
   If she does not appear in the table at all, set every field to 0 except srv/re/ta which may be null.
3. results: team matches from {since} through {today.isoformat()} with a final score, most recent first. For each: date, opponent, home_away (home/away/neutral),
   result ("W 3-1" / "L 0-3"), box_url (the box score link from the schedule page), player_line (her numbers from that box score in plain words,
   e.g. "7 kills, 3 blocks" or "24 assists, 6 digs" or "did not play"; null if you could not open the box score). Open only the newest box score (one fetch); leave player_line null for older matches you did not open.
4. blurb: 2-3 sentences for her parents: what she and the team did lately, whether she is getting court time, what is next. Warm, plain, factual.

Return ONLY:
{{"team_record": {{"overall": "W-L", "conference": "W-L or null"}},
 "stats": {{"mp":0,"sp":0,"k":0,"e":0,"ta":0,"a":0,"bhe":0,"sa":0,"se":0,"srv":null,"dig":0,"re":null,"bs":0,"ba":0,"be":0}},
 "results": [{{"date":"YYYY-MM-DD","opponent":"","home_away":"home","result":"W 3-1","box_url":null,"player_line":null}}],
 "blurb": ""}}"""
    return call_claude(prompt, TOOLS)


def research_weekly(p, today):
    prompt = f"""Today is {today.isoformat()}. Fall {p['college_season']} season. Team: {p['school']} {p.get('team_name', '')} volleyball ({p['division']}, {p['conference']}). Player of interest: {p['name']}.
Schedule page: {p['schedule_url']}
{('Note: ' + p['fetch_note']) if p.get('fetch_note') else ''}

Collect:
1. schedule: EVERY remaining match from {today.isoformat()} to the end of the regular season (and conference tournament dates if listed). For each: date, time exactly as listed,
   time_ct — the same time converted to US Central (the school is in {p['tz']}; Central is one hour behind Eastern), opponent, home_away, location (city, ST or venue),
   stream_name (ESPN+, FloSports, NSIC Network, Hudl, school stream, etc. as labeled) and stream_url (the actual link on the schedule page, else null).
2. standing: the team's current place in its conference standings, formatted like "3rd of 11 OVC" (find the standings page from the schedule page or the conference site; web_search if needed), and standings_url.
3. socials: the official program Instagram and X handles (no @), null if not found.

Return ONLY:
{{"schedule": [{{"date":"YYYY-MM-DD","time":"6:00 PM ET","time_ct":"5:00 PM CT","opponent":"","home_away":"home","location":null,"stream_name":null,"stream_url":null}}],
 "standing": "3rd of 11 OVC", "standings_url": null, "socials": {{"instagram": null, "x": null}}}}"""
    return call_claude(prompt, TOOLS, max_tokens=6000)


def research_profile(p):
    prompt = f"""Player: {p['name']}, {p['school']} volleyball, freshman. Roster/bio page: {p['roster_url']}
{('Note: ' + p['fetch_note']) if p.get('fetch_note') else ''}
From the roster page (open her bio if linked): jersey, position as listed, class_year, height, hometown_hs ("Hometown, ST / High School"), bio_url,
and photo_url — the direct URL of her roster headshot image (an https link to a .jpg/.jpeg/.png/.webp or a Sidearm image URL); null if none.
Return ONLY: {{"jersey":null,"position":null,"class_year":null,"height":null,"hometown_hs":null,"bio_url":null,"photo_url":null}}"""
    return call_claude(prompt, TOOLS, max_tokens=1200)


def research_news(p, today):
    since = (today - timedelta(days=8)).isoformat()
    prompt = f"""Today is {today.isoformat()}. Use web_search (2-4 searches). Find items published since {since} that either
(a) mention {p['name']} by name (school news, local papers such as the Daily Herald / Kane County Reporter / Northwest Herald, conference honors), or
(b) concern the {p['school']} volleyball coaching staff — hires, departures, contract extensions, awards, suspensions — head coach or assistants.
Skip generic game recaps that do not mention her and skip anything older than {since}. Return an empty list if nothing qualifies.
Return ONLY: {{"items": [{{"date":"YYYY-MM-DD","kind":"news" or "coach","title":"","source":"publication name","url":"https://..."}}]}}"""
    return call_claude(prompt, [TOOLS[1]], max_tokens=1500)


def write_piece(kind, players, today, milestones, reunions):
    wk_ago, wk_ahead = (today - timedelta(days=7)).isoformat(), (today + timedelta(days=7)).isoformat()
    compact = []
    for p in players:
        if p.get("status") != "playing": continue
        compact.append({k: p.get(k) for k in ("name", "school_short", "division", "position", "team_record", "stats", "blurb", "stale")}
                       | {"results_last_7": [m for m in p.get("recent_matches", []) if (m.get("date") or "") >= wk_ago],
                          "next_7": [m for m in p.get("upcoming", []) if today.isoformat() <= (m.get("date") or "") <= wk_ahead]})
    if kind == "recap":
        ask = ("Write the MONDAY WEEKEND RECAP: what happened Thursday–Sunday across the group. Who played, who stood out, notable team results, "
               "any milestone. Mention girls by first name. Do not list everyone. Say plainly if someone did not see the court.")
    else:
        ask = ("Write the THURSDAY WEEKEND PREVIEW: what is coming Thursday–Sunday. Big matches, home openers, conference play, any reunion where two of "
               "the girls face each other, and which streams to have ready (all times Central). Mention girls by first name. Do not list everyone.")
    prompt = f"""Today is {today.isoformat()}. Data for the girls (JSON): {json.dumps(compact, ensure_ascii=False)}
Milestones this week: {json.dumps(milestones)}
Reunions coming up: {json.dumps(reunions[:3])}

{ask}
Rules: title under 12 words, no "Recap:" prefix. body = 2-3 paragraphs, each under 70 words, plain warm language, no hype, no bullet points, no jargon.
Ignore anyone marked stale. spotlight = one girl with the best week and a one-sentence reason, or null.
Return ONLY: {{"title":"", "body":["",""], "spotlight": {{"name":"First Last","note":""}} }}"""
    system = "You write short, warm notes for a group of volleyball moms whose daughters played club together and are now college freshmen. Return ONLY valid JSON."
    return call_claude(prompt, None, max_tokens=1200, system=system, model=WRITER_MODEL)


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


def detect_milestones(p, prev_stats, new_stats, today, prev_highs):
    out, highs = [], dict(prev_highs or {})
    if prev_stats and new_stats:
        for key, (name, thresholds) in MILESTONES.items():
            before, after = prev_stats.get(key) or 0, new_stats.get(key) or 0
            for t in thresholds:
                if before < t <= after:
                    out.append({"date": today.isoformat(), "player_id": p["id"],
                                "text": f"First college {name}" if t == 1 else f"{t} college {name}s"})
    for m in (p.get("recent_matches") or [])[:2]:
        for n, word in re.findall(r"(\d+)\s+(kills?|assists?|digs?|aces?|blocks?)", m.get("player_line") or "", re.I):
            w, n = word.lower().rstrip("s"), int(n)
            if n > highs.get(w, 0):
                if highs.get(w, 0) > 0 and n >= 3:
                    out.append({"date": m.get("date") or today.isoformat(), "player_id": p["id"], "text": f"Career high {n} {w}s vs {m.get('opponent', '')}".strip()})
                highs[w] = n
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
    full = os.environ.get("TRACKER_FULL") == "1" or first_run or today.weekday() in (0, 3)   # Mon=0, Thu=3
    piece_kind = "recap" if today.weekday() in (0, 1, 5, 6) else "preview"
    log(f"Run {today} · full={full} · first_run={first_run} · model={MODEL}")

    players, failures, new_miles, new_buzz = [], 0, [], []
    for c in cfg["players"]:
        old = prev_players.get(c["id"], {}) if not first_run else {}
        p = {**old, **c, "college_season": cfg["college_season"]}
        p.pop("fetch_note", None)
        for k in ("recent_matches", "upcoming", "season_stats"): p.setdefault(k, [])
        for k in ("stats", "team_record", "photo_url"): p.setdefault(k, None)
        p.setdefault("socials", {"instagram": None, "x": None}); p["stale"] = False
        if c.get("status") != "playing":
            p.update({"recent_matches": [], "upcoming": [], "stats": None, "season_stats": []}); players.append(p); continue

        log(f"{p['name']} ({p['school_short']})")
        try:
            if not old.get("jersey") or not old.get("photo_url"):
                log("  profile"); prof = research_profile(c)
                p.update({k: v for k, v in prof.items() if v})
            log("  daily"); d = research_daily({**c, **p}, today)
            prev_stats = old.get("stats")
            if d.get("team_record"): p["team_record"] = {**(p.get("team_record") or {}), **{k: v for k, v in d["team_record"].items() if v}}
            if d.get("stats"): p["stats"] = {k: d["stats"].get(k) for k in STAT_KEYS}
            p["recent_matches"] = merge_matches(old.get("recent_matches"), d.get("results"))
            if d.get("blurb"): p["blurb"] = d["blurb"]
            played = {(m.get("date"), (m.get("opponent") or "").lower()) for m in p["recent_matches"] if m.get("result")}
            p["upcoming"] = [m for m in p["upcoming"] if m.get("date") and m["date"] >= today.isoformat() and (m["date"], (m.get("opponent") or "").lower()) not in played]
            miles, highs = detect_milestones(p, prev_stats, p["stats"], today, old.get("_highs"))
            p["_highs"] = highs; new_miles += miles
            if full:
                log("  weekly"); w = research_weekly({**c, **p}, today)
                if w.get("schedule"):
                    p["upcoming"] = sorted([m for m in w["schedule"] if m.get("date") and (m["date"], (m.get("opponent") or "").lower()) not in played], key=lambda m: m["date"])
                if w.get("standing"): p["team_record"] = {**(p.get("team_record") or {}), "standing": w["standing"], "standings_url": w.get("standings_url")}
                if w.get("socials") and any((w["socials"] or {}).values()): p["socials"] = w["socials"]
                if first_run or today.weekday() == 0:
                    log("  news"); n = research_news(c, today)
                    new_buzz += [{**it, "player_id": p["id"]} for it in n.get("items", []) if it.get("url") and it.get("title")]
            p["fetched_at"] = now.isoformat(timespec="minutes")
        except Exception as e:
            failures += 1; log(f"  FAILED: {e}"); p["stale"] = True; p["error"] = str(e)[:200]
        p["matches_played"], p["sets_played"] = (p["stats"] or {}).get("mp"), (p["stats"] or {}).get("sp")
        p["season_stats"] = season_stats_list(p["stats"])
        players.append(p); time.sleep(1.5)

    milestones = sorted((prev.get("milestones") or []) + new_miles, key=lambda m: m["date"], reverse=True)[:60]
    reunions = compute_reunions(players)
    buzz = list(prev.get("buzz") or [])
    have = {b.get("url") for b in buzz if b.get("url")}
    buzz += [b for b in new_buzz if b["url"] not in have]
    summary = prev.get("summary")
    if full:
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

    data = {"team": cfg["team"], "club": cfg["club"], "club_season": cfg["club_season"], "college_season": cfg["college_season"],
            "updated_at": now.isoformat(timespec="minutes"), "updated_label": now.strftime("%A %-I:%M %p CT"),
            "summary": summary, "players": players, "milestones": milestones, "reunions": reunions, "buzz": buzz,
            "run": {"date": today.isoformat(), "full": full, "failures": failures, "model": MODEL}}
    json.dump(data, open(DATA_FILE, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    log(f"Wrote data.json · {len(players)} players · {failures} failure(s) · {len(new_miles)} milestone(s) · {len(reunions)} reunion(s)")


if __name__ == "__main__":
    main()
