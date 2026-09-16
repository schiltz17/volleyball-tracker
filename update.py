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

import base64, json, os, re, sys, time, threading, urllib.request, urllib.error
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
    pat = re.compile(rf"(?:^|\n)\s*(\d{{1,2}})?\s*{re.escape(last)},\s*{re.escape(first)}\b[^\n\d-]*((?:-?\.?\d+(?:\.\d+)?\s+){{20,22}})", re.I)
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
    """Overwrite the model's W/L and score with the sheet's, matched by date (+ first word of opponent when a date has two matches)."""
    for r in results or []:
        if not isinstance(r, dict) or not r.get("date"): continue
        cands = [x for x in pdf_results if x["date"] == r["date"]]
        if len(cands) > 1:
            w = (r.get("opponent") or "").lower().split()[:1]
            cands = [x for x in cands if w and w[0] in x["opponent"].lower()] or cands
        if len(cands) == 1: r["result"] = cands[0]["result"]
    return results


def pdf_text(b):
    """Text of a PDF via pypdf (installed by the workflow). Returns "" if unavailable."""
    try:
        import io
        from pypdf import PdfReader
        return "\n".join((pg.extract_text() or "") for pg in PdfReader(io.BytesIO(b)).pages)
    except Exception as e:
        log(f"    pypdf failed: {str(e)[:60]}"); return ""


# ---------------------------------------------------------------- page fetching (Python browses, Claude only reads)
UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/128.0 Safari/537.36"


def http_get(url, timeout=30, binary=False):
    req = urllib.request.Request(url, headers={"User-Agent": UA, "Accept": "*/*", "Accept-Language": "en-US,en;q=0.9"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        data = r.read()
        return data if binary else data.decode(r.headers.get_content_charset() or "utf-8", "replace")


class Blocks(HTMLParser):
    """Turns HTML into text lines. Rows/list items/divs become lines; cells become tab-separated; hrefs and img srcs are kept as [href] / [img] tags."""
    BLOCK = {"tr", "li", "p", "h1", "h2", "h3", "h4", "section", "article", "table", "thead", "tbody", "ul", "ol"}   # these end a line
    SEP = {"div", "br", "dt", "dd", "span"}                                                                         # these just separate parts within a line
    SKIP = {"script", "style", "noscript", "svg", "head", "nav", "footer", "iframe"}
    def __init__(self, base):
        super().__init__(convert_charrefs=True); self.out, self.cur, self.skip, self.base = [], [], 0, base
    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        if tag in self.SKIP: self.skip += 1
        if tag in ("td", "th"): self.cur.append("\t")
        elif tag in self.SEP: self.cur.append(" | ")
        if tag in self.BLOCK: self.flush()
        if tag == "a" and a.get("href"): self.cur.append(f" [href {urllib.request.urljoin(self.base, a['href'])}] ")
        if tag == "img" and (a.get("data-src") or a.get("src")): self.cur.append(f" [img {urllib.request.urljoin(self.base, a.get('data-src') or a['src'])}] ")
    def handle_endtag(self, tag):
        if tag in self.SKIP: self.skip = max(0, self.skip - 1)
        if tag in self.BLOCK: self.flush()
    def handle_data(self, d):
        if not self.skip:
            self.cur.append(d)
            if self.handle_data_len() > 1200: self.flush()      # pages without li/tr structure still get line breaks
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


def keep_lines(text, patterns, context=0, max_chars=24000):
    """Keep lines matching any pattern (plus neighbours); fall back to the head of the page if nothing matches."""
    lines = text.split("\n"); rx = [re.compile(p, re.I) for p in patterns]; keep = set()
    for i, ln in enumerate(lines):
        if any(r.search(ln) for r in rx):
            for j in range(max(0, i - context), min(len(lines), i + context + 1)): keep.add(j)
    out = "\n".join(lines[i] for i in sorted(keep)) if keep else text
    return out[:max_chars]


def find_pdf(raw_html, base):
    """The season cumulative stats sheet, if the page links one. Prefers links that look like it; ignores media guides etc."""
    links = [urllib.request.urljoin(base, m) for m in re.findall(r"""href=["']([^"']+\.pdf[^"']*)["']""", raw_html, re.I)]
    for pat in (r"cume", r"stats?/\d{4}", r"season", r"overall"):
        for u in links:
            if re.search(pat, u, re.I): return u
    return None


def fetch_pdf(url, max_bytes=3_000_000):
    b = http_get(url, binary=True)
    if not b.startswith(b"%PDF") or len(b) > max_bytes: raise ValueError("not a usable PDF")
    return b


def record_from(text):
    """'Overall 4-4 · Conf 0-0' style record text from a Sidearm schedule page, if present."""
    o = re.search(r"Overall\D{0,12}(\d{1,2}-\d{1,2})", text, re.I)
    c = re.search(r"Conf(?:erence)?\D{0,12}(\d{1,2}-\d{1,2})", text, re.I)
    return {"overall": o.group(1) if o else None, "conference": c.group(1) if c else None}


def gather_daily(p, today):
    """Fetch stats (PDF preferred), schedule, and the box scores she still needs. Returns compact text + attachments, or raises."""
    parts, docs, notes, pdf_results = [], [], [], []
    surname = p["name"].split()[-1]
    # schedule page: game blocks with box score links
    sched_text, _ = page_text(p["schedule_url"])
    rec = record_from(sched_text)
    if rec["overall"]: notes.append(f"record from page: {rec['overall']}")
    parts.append("=== SCHEDULE / RESULTS PAGE (each line is one game block; [href ...] are the links in it) ===\n" +
                 keep_lines(sched_text, [r"\b(Aug|Sep|Oct|Nov|Dec)\b|\d{1,2}/\d{1,2}", r"\[href [^\]]*box", r"\b[WL]\b,?\s*\d-\d"], 0, 16000))
    # stats: cumulative PDF if the page links one, else the trimmed HTML table
    stats_text, stats_raw = page_text(p["stats_url"])
    pdf = find_pdf(stats_raw, p["stats_url"])
    parsed = None
    if pdf:
        try:
            pdf_bytes = fetch_pdf(pdf); txt = pdf_text(pdf_bytes)
            parsed = stats_from_cume(txt, p["name"], p.get("jersey")) if txt else None
            pdf_results = results_from_cume(txt) if txt else []
            if pdf_results: notes.append(f"{len(pdf_results)} results on the sheet")
            if parsed:
                parsed["mp"] = mp_from_html_rows(stats_text.split("\n"), parsed)
                notes.append(f"stats parsed from PDF: sp={parsed['sp']} k={parsed['k']} a={parsed['a']} dig={parsed['dig']}")
            elif txt and re.search(rf"\b{re.escape(surname)},", txt, re.I) is None:
                parsed = {k: (0 if k not in ("srv", "re", "mp") else None) for k in STAT_KEYS}; notes.append("not on the stats sheet — zeros")
            else:
                docs.append(("Season cumulative stats PDF", pdf_bytes)); notes.append("PDF row not parsed — PDF attached for reading")
        except Exception as e:
            notes.append(f"pdf skipped ({str(e)[:50]})")
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
    return "\n\n".join(parts), docs, notes, rec, parsed, pdf_results


# ---------------------------------------------------------------- research calls
def research_daily(p, today, need_lines=()):
    since = (today - timedelta(days=14)).isoformat()
    ident = f"#{p['jersey']} " if p.get("jersey") else ""
    who = (f"{p['name']} ({ident}{p.get('position') or p['club_position']}, freshman, {p.get('hometown_hs') or 'hometown per roster'}). "
           f"{p.get('disambiguation', '')} {p.get('position_note', '')}")
    schema = ('{"team_record": {"overall": "W-L", "conference": "W-L or null"},\n'
              ' "stats": {"mp":0,"sp":0,"k":0,"e":0,"ta":0,"a":0,"bhe":0,"sa":0,"se":0,"srv":null,"dig":0,"re":null,"bs":0,"ba":0,"be":0},\n'
              ' "results": [{"date":"YYYY-MM-DD","opponent":"","home_away":"home","result":"W 3-1","box_url":null,"player_line":null}],\n'
              ' "blurb": ""}')
    rules = f"""Rules:
- stats: if a section "HER SEASON STAT LINE" is present, copy it exactly. Otherwise take HER single row from the stats table (match by jersey number AND last name; never add rows together; if two rows could be her, return null stats). Integers; null where a column is not published.
- team_record = the record printed on the schedule/results page (e.g. "Overall 4-4"); copy it, do not tally matches yourself.
- result is written W/L then HER team's sets first: "W 3-1", "L 0-3" — never "L 3-0".
- results = the team's matches from {since} through {today.isoformat()} that have a final score, most recent first, with box_url = the box-score link from that game's block. player_line = her numbers from a BOX SCORE section below if one is present for that match ("7 kills, 3 blocks, 2 digs" / "24 assists, 6 digs" / "did not play"); otherwise null.
- blurb = 2-3 sentences for her parents: what she and the team did lately, whether she is getting court time, what is next. Warm, plain, factual. Treat any position note above as fact and never mention where it came from (no "per the family", no "listed as").
  The stats sheet decides court time: if she is not in it or has 0 sets played, say plainly that she has not appeared in a match yet and move on to the team. Never write about the data itself — no mention of pages, PDFs, box scores, tables, or what could or could not be found. Write only about her and the team.
Return ONLY: {schema}"""
    text, docs, notes, rec, parsed, pdf_results = None, [], [], {}, None, []
    if not p.get("fetch_note"):
        try: text, docs, notes, rec, parsed, pdf_results = gather_daily(p, today)
        except Exception as e: notes, rec, parsed, pdf_results = [f"fetch failed ({str(e)[:80]}) — used search"], {}, None, []
    if text is None or len(text) < 200:      # blocked or empty site: one search-only call
        prompt = f"""Today is {today.isoformat()}. Player: {who}
Her school's site blocks automated reading. Use web_search (up to 2 searches: "{p['school']} volleyball {p['name'].split()[-1]}", "{p['school']} volleyball results 2026", "{p['school']} volleyball stats") and read the result snippets only.
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
    out["stats_source"] = "sheet-parsed" if parsed else "model-read"
    if parsed: out["stats"] = parsed                     # code-parsed line overrides whatever the model wrote
    if rec.get("overall"):          # the page's own record beats anything the model tallied
        out["team_record"] = {**(out.get("team_record") or {}), "overall": rec["overall"], **({"conference": rec["conference"]} if rec.get("conference") else {})}
    if pdf_results: apply_pdf_results(out.get("results"), pdf_results)   # the sheet's scores beat the model's
    for m in out.get("results") or []:
        if isinstance(m, dict) and m.get("result"): m["result"] = fix_result(m["result"])
    return out


def research_weekly(p, today):
    """Full remaining schedule (Python-fetched) + standings (one search)."""
    shape = ('{"schedule": [{"date":"YYYY-MM-DD","time":"6:00 PM ET","time_ct":"5:00 PM CT","opponent":"","home_away":"home","location":null,"stream_name":null,"stream_url":null}], '
             '"standing": "3rd of 11 OVC or null", "standings_url": null}')
    sched_text, raw = None, ""
    if not p.get("fetch_note"):
        try: sched_text, raw = page_text(p["schedule_url"])
        except Exception as e: log(f"    schedule fetch failed ({str(e)[:60]}) — using search")
    if not sched_text or len(sched_text) < 200:
        prompt = f"""Today is {today.isoformat()}. Team: {p['school']} volleyball ({p['conference']}). Its site blocks automated reading; use web_search (2 searches) for the remaining 2026 schedule and the conference standings. Times in Central.
Return ONLY: {shape}"""
        return call_claude(prompt, [{**SEARCH, "max_uses": 2}], max_tokens=4000, model=STRONG_MODEL)
    trimmed = keep_lines(sched_text, [r"\b(Sep|Oct|Nov|Dec)\b|\d{1,2}/\d{1,2}", r"\[href", r"ESPN|Flo|Network|Stream|Watch|Video|Live"], 0, 22000)
    ig = re.search(r"instagram\.com/([A-Za-z0-9_.]+)", raw); xx = re.search(r"(?:twitter|x)\.com/([A-Za-z0-9_]+)", raw)
    prompt = f"""Today is {today.isoformat()}. Team: {p['school']} {p.get('team_name', '')} volleyball ({p['division']}, {p['conference']}); the school is in the {p['tz']} time zone.
Below is the schedule page as text (one game per line; [href ...] are that game's links, including streaming/TV links).

{trimmed}

Return every match from {today.isoformat()} through the end of the season (conference tournament too if listed). time = as listed; time_ct = converted to US Central. stream_name/stream_url = the streaming/TV label and link in that game's block, else null.
Then use web_search ONCE for "{p['conference']} volleyball standings 2026" and report the team's place as "3rd of 11 OVC" with the standings page URL (null if not found).
Return ONLY: {shape}"""
    out = call_claude(prompt, [{**SEARCH, "max_uses": 1}], max_tokens=5000, model=STRONG_MODEL)
    out["socials"] = {"instagram": ig.group(1) if ig else None, "x": xx.group(1) if xx else None}
    return out


def research_profile(p):
    shape = '{"jersey":null,"position":null,"class_year":null,"height":null,"hometown_hs":"Hometown, ST / High School","bio_url":null,"photo_url":null}'
    text = None
    if not p.get("fetch_note"):
        try: text, _ = page_text(p["roster_url"])
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
        compact.append({k: p.get(k) for k in ("name", "school_short", "division", "position", "position_note", "team_record", "stats", "blurb", "stale")}
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
    return call_claude(prompt, None, max_tokens=2500, system=system, model=WRITER_MODEL, exempt=True)


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
        if d.get("_notes"): log("    " + "; ".join(d["_notes"]))
        prev_stats = old.get("stats")
        if isinstance(d.get("team_record"), dict): p["team_record"] = {**(p.get("team_record") or {}), **{k: as_text(v) for k, v in d["team_record"].items() if v}}
        if isinstance(d.get("stats"), dict):
            new_stats = {k: (int(float(d["stats"][k])) if str(d["stats"].get(k, "")).replace(".", "").isdigit() else None) for k in STAT_KEYS}
            if any(v is not None for v in new_stats.values()): p["stats"] = new_stats; p["stats_source"] = d.get("stats_source")
            elif old.get("stats"): log(f"  stats came back empty — keeping last good line for {p['name']}")
        p["recent_matches"] = merge_matches(old.get("recent_matches"), clean_matches(d.get("results")))
        if d.get("blurb") and not (isinstance(d.get("stats"), dict) and not any(v is not None for v in new_stats.values()) and old.get("blurb")):
            p["blurb"] = as_text(d["blurb"])
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
            junk = re.compile(r"roster|\bcommit|schedule\b|/sports/womens-volleyball/?$|sportsrecruits|topflightvbc", re.I)
            items = [{**it, "player_id": p["id"]} for it in n.get("items", [])
                     if it.get("url") and it.get("title") and not junk.search(it["title"] + " " + it["url"])]
        p["fetched_at"] = now.isoformat(timespec="minutes")
    except Exception as e:
        failed = True; log(f"  FAILED {p['name']}: {str(e)[:160]}"); p["stale"] = True; p["error"] = str(e)[:200]
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
    if writing_day:
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
