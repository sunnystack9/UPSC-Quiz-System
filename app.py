"""
Algorithmic Quiz System — UPSC / HCS Prelims GS Paper I
A practice-mode Streamlit app. Not for use during the live exam --
this tool is strictly for home practice.
"""

import streamlit as st
import json
import random
import re
import base64
import contextlib
import hashlib
import os
import secrets
import tempfile
import time
from pathlib import Path
from datetime import datetime, timedelta
from collections import defaultdict

try:
    import requests
except ImportError:
    requests = None

try:
    from google import genai
    from google.genai import types as genai_types
except ImportError:
    genai = None
    genai_types = None

try:
    import fcntl
except ImportError:
    fcntl = None  # Windows fallback: writes still work, just without lock protection

DATA_DIR = Path(__file__).parent / "data"
RESPONSES_FILE = Path(__file__).parent / "responses.json"
SCHEDULE_FILE = Path(__file__).parent / "schedule.json"
USERS_FILE = Path(__file__).parent / "users.json"
NOTES_FILE = Path(__file__).parent / "community_notes.json"
RESOLVED_FILE = Path(__file__).parent / "resolved.json"
BOOKMARKS_FILE = Path(__file__).parent / "bookmarks.json"
EDITS_FILE = Path(__file__).parent / "edit_log.json"
ACTIVE_SESSIONS_FILE = Path(__file__).parent / "active_sessions.json"

# Official UPSC CSE Prelims and HCS Prelims GS Paper I marking schemes.
# HCS in particular revises its pattern periodically (most recently
# Jan 2026) -- re-check this against the current-year official notification
# (hpsc.gov.in / upsc.gov.in) before relying on it for real practice.
# Only paper "1" (GS Paper I) is modeled; this app doesn't cover CSAT Paper II.
MARKING_CONFIG = {
    ("UPSC", "1"): {"marks_correct": 2.0, "marks_wrong": -0.66, "label": "UPSC GS Paper I"},
    ("HCS", "1"): {"marks_correct": 1.0, "marks_wrong": -0.25, "label": "HCS GS Paper I"},
}


def marking_for(exam, paper):
    """Returns the marking-scheme dict for (exam, paper), or None if this
    question's exam/paper isn't in MARKING_CONFIG. Callers must handle None
    rather than assume every question has a known marking scheme -- this
    keeps the door open to adding more exams/papers to the data later
    without a matching MARKING_CONFIG entry crashing scoring."""
    return MARKING_CONFIG.get((exam, str(paper)))


try:
    st.set_page_config(
        page_title="Algorithmic Quiz System — UPSC / HCS GS Paper I",
        layout="centered",
        initial_sidebar_state="collapsed",
    )
except Exception:
    # set_page_config() needs an active Streamlit script-run context and isn't
    # available (or needed) when app.py is merely imported -- e.g. by the
    # tests in tests/test_core.py, which exercise the pure logic below
    # without a live `streamlit run`.
    pass

WEAKNESS_BASELINE = 3.0
WEAKNESS_FLOOR = 0.5
# How far a fully-weak theme's sampling weight sits above WEAKNESS_FLOOR --
# see theme_weight(). weakness_scores are normalized to roughly [0, 1] by
# compute_theme_weakness(), so this is what maps that back onto a usable
# sampling-weight range: weight = WEAKNESS_FLOOR + weakness * WEAKNESS_SPAN.
WEAKNESS_SPAN = 6.0
# Bayesian-smoothing prior for compute_theme_weakness() and the new
# compute_question_weakness() below (the same per-user mastery concept,
# just grouped by theme vs. by individual question_id), as well as
# compute_question_stats() (a DIFFERENT, cohort-wide-not-per-user notion of
# difficulty, hence its own QUESTION_DIFFICULTY_PRIOR pair): before any
# evidence, assume a neutral 0.6 mastery/success rate rather than 0 --
# otherwise a theme, question, or user-question pair with only one or two
# attempts swings straight to "totally mastered" or "totally failed" on a
# single data point, the same way two coin flips don't prove a coin is rigged.
# *_PRIOR_STRENGTH is in attempt-weight units: how many "phantom" neutral
# attempts the prior is worth -- higher means slower to react to early attempts.
MASTERY_PRIOR = 0.6
MASTERY_PRIOR_STRENGTH = 2.0
QUESTION_DIFFICULTY_PRIOR = 0.6
QUESTION_DIFFICULTY_PRIOR_STRENGTH = 2.0
# Floor for a theme's "how unexplored is this" weight when picking NEW
# (never-attempted) questions -- see theme_coverage_weights(). Never zero:
# even a nearly-fully-covered theme should still occasionally get its
# remaining new questions picked, not be shut out entirely.
NEW_ITEM_COVERAGE_FLOOR = 0.1
# hours: 0h, 4h, 12h, 1d, 2d, 4d, 7d, 10d -- rescaled from a days-based ladder
# for a ~30-day exam runway. Fast early touches (same day) for initial
# retention, tapering to a 10-day ceiling rather than the old 240-day one --
# a ceiling anywhere near 30 days would mean "fully mastered" content might
# only get reviewed once, or not at all, before the exam. update_schedule_
# entry() reads its "max stage" from len(STAGE_INTERVALS_HOURS), so nothing
# else needs to change to add more stages.
STAGE_INTERVALS_HOURS = [0, 4, 12, 24, 48, 96, 168, 240]
RECENCY_HALF_LIFE_DAYS = 30  # a response's weight in the weakness score halves every N days
MAX_USERS = 50  # registered-profile cap for this deployment; existing profiles can always log in
# Review-priority scoring for overdue questions, in "hours of overdue-age"
# units so it composes directly with actual overdue age (see
# review_priority_bonus() / split_pool_by_schedule()). This is a BLENDED
# score, not a hard priority tier: overdue age keeps accumulating without a
# ceiling, so a genuinely stale backlog item still eventually outranks any
# capped bonus below -- a couple of recent misses shifts a question up the
# queue, it doesn't let it starve everything older.
# A consecutive wrong-answer streak (resets the moment the learner gets it
# right again -- see wrong_streak_counts()) is objective evidence of a
# CURRENT trouble spot. QUESTION_WEAKNESS_BONUS_SCALE is a complementary,
# longer-memory signal on top of the streak (see compute_question_weakness()):
# a Bayesian-smoothed, recency-weighted measure of how this learner has done
# on this exact question across their WHOLE history with it, so a question
# missed repeatedly over weeks -- with the odd lucky correct guess in between
# resetting the streak to 0 -- still keeps some priority instead of being
# fully forgotten each time.
# NOTE: bookmarking a question intentionally has NO effect on this score, or
# on regular Smart Quiz session composition at all -- a star is purely a
# personal flag for the dedicated Bookmarks-tab session (see render_bookmarks
# / "Start practice with my bookmarks"). There used to be a BOOKMARK_BONUS
# constant and a bookmark-based exclusion from new/not_due pools here; both
# were removed by request so bookmarking a question can't change its odds
# of appearing in -- or being prioritized within -- a freshly built quiz.
WRONG_STREAK_BONUS_PER_MISS = 24  # hours-equivalent per consecutive wrong answer, starting at streak 2
WRONG_STREAK_BONUS_CAP = 72       # caps the streak bonus at 3 misses' worth, so it can't dominate everything
QUESTION_WEAKNESS_BONUS_SCALE = 48  # hours-equivalent at a fully-weak (1.0) personal history with this question
# Fallback fraction of an adaptive session reserved for due re-reviews when
# there's no completed-session history yet to scale from (a brand-new
# profile's very first session) -- see dynamic_review_share() below, which
# is what actually sets the share for every session after that, scaled
# directly to the learner's most recent quiz score. Before that dynamic
# version existed, this WAS the share for every session, regardless of
# performance; a strict tier order (either due-first or new-first) starves
# whichever pool is smaller: due-first swamped new material entirely
# whenever the overdue backlog was >= the session size (the original bug),
# and new-first later swamped due-reviews just as completely once the
# not-attempted pool got large (937 vs. a 10-question session, say). A fixed
# share sidesteps both failure modes -- due-reviews always get roughly this
# fraction of a session (ranked by review_priority_bonus, highest first),
# not-yet-attempted material fills the rest, and whichever pool comes up
# short lets the other backfill so a session is never short a question.
REVIEW_SLOT_SHARE = 0.4



# Constants (avoids typo-induced bugs breaking comparisons in the adaptive-weighting math)
OPT_SKIPPED = "SKIPPED"
CONF_GUESSED = "Guessed"
CONF_SOMEWHAT = "Somewhat sure"
CONF_CONFIDENT = "Confident"

# Per-response evidence weight by confidence, used in compute_theme_weakness()
# and compute_question_stats(): a guess (right or wrong) carries weaker
# evidence of mastery either way than an answer given with actual conviction.
CONFIDENCE_ATTEMPT_WEIGHT = {CONF_GUESSED: 0.5, CONF_SOMEWHAT: 1.0, CONF_CONFIDENT: 1.0}
# Of that evidence, how much of a *correct* answer counts as proof of mastery
# rather than luck -- confident-correct is strong evidence, a correct guess is weak.
CONFIDENCE_SUCCESS_CREDIT = {CONF_GUESSED: 0.5, CONF_SOMEWHAT: 0.9, CONF_CONFIDENT: 1.0}

_ROMAN_ORDER = ["i", "ii", "iii", "iv", "v", "vi", "vii", "viii", "ix", "x"]
_ROMAN_ORDER_UPPER = [r.upper() for r in _ROMAN_ORDER]
# Opening "(" is optional -- source data uses both "(i)" and bare "i)" styles.
# Lookahead accepts an uppercase letter OR a digit: most enumerated sub-statements
# open with a capitalized word ("Any person..."), but plenty of tax-law
# sub-statements open directly with a section number instead, e.g.
# "i) 194BA(1)-Notwithstanding...", which a letter-only lookahead would miss.
_PATTERN_ROMAN = re.compile(r'(\*\*)?\(?(i{1,3}|iv|v|vi{0,3}|ix|x)\)(\*\*)?\s+(?=[A-Z0-9])')
# Uppercase Roman-numeral-with-period style, e.g. "I. ... II. ... III. ...IV. ..."
# -- distinct from the "(i)/i)" style above and from single-letter A./B./C.
_PATTERN_ROMAN_UPPER = re.compile(r'\b(I{1,3}|IV|V|VI{0,3}|IX|X)\.\s+(?=[A-Z])')
_PATTERN_ALPHA = re.compile(r'\b([A-J])\.\s+(?=[A-Z])')
_PATTERN_NUM = re.compile(r'\b([1-9][0-9]?)\.\s+(?=[A-Z])')


def _line_starts_with_pipe(text, pos):
    """True if `pos` falls on a markdown table row -- i.e. the line
    containing it, once leading spaces are stripped, starts with '|'.

    format_question_stem() below inserts a paragraph break (\\n\\n)
    immediately before an enumerated marker like '(i)' to split run-on
    list items onto their own lines -- but a 'Match List I with List II'
    table stores its rows as '| (i) Section 69 | a. ... |', with the
    marker sitting right after the row's OWN leading pipe, not at the
    start of a sentence. Inserting a paragraph break there orphans that
    leading '|' on its own line and opens a blank line in the middle of
    the table, which breaks Markdown table parsing entirely: the header
    still renders (it has no enumerated marker to trip this), but every
    body row after the first split falls out of the table and shows up as
    plain text with literal '|' characters -- exactly the "why doesn't
    this look like a table" bug this check exists to prevent."""
    line_start = text.rfind("\n", 0, pos) + 1
    line_end = text.find("\n", pos)
    if line_end == -1:
        line_end = len(text)
    return text[line_start:line_end].lstrip(" ").startswith("|")


def format_question_stem(text):
    """Inserts paragraph breaks before enumerated sub-statements (A./B./C.., 
    1./2./3.., I./II./III.., or (i)/(ii)/(iii)../i)/ii)/iii)..), but only when
    they form a genuine strict sequence. Never splits inside a markdown table
    row -- see _line_starts_with_pipe()."""
    valid_positions = set()
    matches = list(_PATTERN_ROMAN.finditer(text))
    if matches:
        seq_idx = 0
        run_positions = []
        for m in matches:
            roman = m.group(2)
            expected = _ROMAN_ORDER[seq_idx] if seq_idx < len(_ROMAN_ORDER) else None
            if roman == expected:
                run_positions.append(m.start())
                seq_idx += 1
            elif roman == "i":
                run_positions = [m.start()]
                seq_idx = 1
        if len(run_positions) >= 2:
            valid_positions.update(run_positions)

    ru_matches = list(_PATTERN_ROMAN_UPPER.finditer(text))
    if ru_matches:
        seq_idx = 0
        run_positions = []
        for m in ru_matches:
            roman = m.group(1)
            expected = _ROMAN_ORDER_UPPER[seq_idx] if seq_idx < len(_ROMAN_ORDER_UPPER) else None
            if roman == expected:
                run_positions.append(m.start())
                seq_idx += 1
            elif roman == "I":
                run_positions = [m.start()]
                seq_idx = 1
        if len(run_positions) >= 2:
            valid_positions.update(run_positions)

    a_matches = list(_PATTERN_ALPHA.finditer(text))
    if a_matches:
        run, best_run, prev_letter = [], [], None
        for m in a_matches:
            letter = m.group(1)
            if prev_letter is not None and ord(letter) == ord(prev_letter) + 1:
                run.append(m.start())
            else:
                run = [m.start()]
            if len(run) > len(best_run):
                best_run = run
            prev_letter = letter
        if len(best_run) >= 2:
            valid_positions.update(best_run)

    n_matches = list(_PATTERN_NUM.finditer(text))
    if n_matches:
        run, best_run, prev_num = [], [], None
        for m in n_matches:
            num = int(m.group(1))
            if prev_num is not None and num == prev_num + 1:
                run.append(m.start())
            else:
                run = [m.start()] if num == 1 else []
            if len(run) > len(best_run):
                best_run = run
            prev_num = num
        if len(best_run) >= 2:
            valid_positions.update(best_run)

    result = text
    if valid_positions:
        # Only skip a position if it already has a genuine paragraph break (\n\n)
        # right before it. A single \n (common in the source data) is exactly the
        # case this function needs to fix — Markdown collapses a lone \n into a
        # space, which is why some questions render as one run-on paragraph
        # despite the source text having newlines between statements.
        valid_positions = {
            pos for pos in valid_positions
            if not text[:pos].rstrip(" ").endswith("\n\n")
            and not _line_starts_with_pipe(text, pos)
        }
        for pos in sorted(valid_positions, reverse=True):
            head = result[:pos].rstrip(" \n")
            tail = result[pos:].lstrip("\n")
            result = head + "\n\n" + tail

    # Break out a trailing instruction phrase ("Choose the correct option:",
    # "Which of the above are true?", etc.) that's stuck onto the last list item.
    # Uses the LAST match, not the first — "Which of the following" is also how
    # many of these questions legitimately OPEN, so only treat it as a trailing
    # instruction when it's not at/near the very start of the text.
    _trailer_pattern = re.compile(
        r'Choose the correct (?:option|answer)\b|'
        r'Select the correct (?:option|answer)\b|'
        r'The correct answer is\b|'
        r'Which of the above\b|Which of the following\b',
        re.I,
    )
    # "Choose/Select the correct ..." and "The correct answer is" are unambiguous
    # instructional closers -- unlike "Which of the following", they never
    # legitimately appear mid-sentence, so they can start a new line even without
    # preceding punctuation. This is exactly what's needed when a list item has
    # no terminal punctuation of its own, e.g. "...IV. Domestic company Choose
    # the correct option:" -- item IV already gets its own line from the roman-
    # numeral splitting above, but without this, the trailing instruction would
    # still be glued onto the end of it.
    _unambiguous_trailer = re.compile(
        r'Choose the correct (?:option|answer)\b|'
        r'Select the correct (?:option|answer)\b|'
        r'The correct answer is\b',
        re.I,
    )
    def _is_sentence_boundary(pos, end, unambiguous):
        before = result[:pos].rstrip(" \n")
        if not before or before[-1] in '.:;)':
            return True
        if not unambiguous:
            return False
        # An unambiguous closer only skips the punctuation requirement when
        # it's actually acting as a short, final closer -- little or
        # nothing left to say once "choose/select the correct option" has
        # been said. Without this check, the exact same wording appearing
        # naturally mid-question (e.g. "...Please select the correct
        # option from the options given below, of those persons who are
        # required to...") gets mistaken for a trailer too, splitting the
        # sentence in the middle even though a long remainder follows --
        # a real trailing instruction never has that much left to say.
        remainder = result[end:].strip()
        return len(remainder) <= 60

    trailer_matches = [
        m for m in _trailer_pattern.finditer(result)
        if m.start() > 20
        and _is_sentence_boundary(m.start(), m.end(), bool(_unambiguous_trailer.match(result, m.start())))
    ]
    if trailer_matches:
        pos = trailer_matches[-1].start()
        result = result[:pos].rstrip(" \n") + '\n\n' + result[pos:].lstrip(" \n")
    return result.lstrip("\n")


def render_justified(text, container_key, extra_style=""):
    """Renders question text with justified alignment.

    Previously this escaped the text, hand-converted **bold** via regex,
    and turned every \\n into <br> inside a raw HTML <div>. That worked for
    bold and line breaks, but silently broke two things: any $$...$$ LaTeX
    math shows up as literal text, and any bullet list (* item) shows up
    as a literal asterisk rather than a real bullet — because CommonMark
    treats a raw HTML block like that <div> as opaque and stops parsing
    Markdown/math inside it entirely.

    This version uses Streamlit's own native st.markdown() instead, which
    handles bold, bullet lists, and LaTeX correctly out of the box.
    Justification is applied via a small CSS rule scoped to a keyed
    container (container_key) rather than the <div> wrapper, so parsing
    still happens normally. container_key must be unique per call within a
    single script run — when rendering a list of stems (e.g. one per
    bookmarked question), pass a key that includes the question_id.

    format_question_stem() already inserts real paragraph breaks (\\n\\n)
    before detected enumerated markers like (i)/(ii)/A./B. — those are left
    untouched here. Any OTHER lone \\n in the text (freeform line breaks in
    the source data that aren't part of a detected sequence) still needs
    converting, since native Markdown otherwise collapses a lone newline
    into a space — the old code avoided this by blanket-converting every
    \\n to <br>, which the new native-markdown approach can't do the same
    way, so it's handled explicitly below instead.

    extra_style appends additional inline CSS (e.g. spacing between entries
    when rendering a list of options one call at a time)."""
    st.markdown(
        f"<style>[class*='st-key-{container_key}'] p {{ text-align: justify; {extra_style} }}</style>",
        unsafe_allow_html=True,
    )
    display_text = re.sub(r'(?<!\n)\n(?!\n)', '  \n', text)
    with st.container(key=container_key):
        st.markdown(display_text)


def inject_option_justify_css():
    """Applies the same justified-alignment + 1rem font-size treatment used
    for the question stem (see render_justified()) to the OTHER text
    surfaces a question's content appears in: option rows rendered via
    st.success/st.error/st.write (plain <p> tags inside stAlert or
    stMarkdownContainer), and explanation text rendered the same way.

    render_question() (the live quiz screen) has always injected this
    rule inline as part of a larger CSS block that also fixes radio-circle
    alignment -- but that inline injection only ever ran on that one
    screen. render_full_question() (Bookmarks tab, and all three Flagged
    Qns sections -- flagged/missing_key/disputed) and
    render_reviewed_question() (the Prev/Next in-session history view)
    both call render_justified() for the stem, but never injected this
    rule for anything else on the page -- so their options and explanation
    silently fell back to Streamlit's default left-aligned, smaller font,
    inconsistent with the live quiz view right next to them. This call
    fixes that gap; render_question()'s own inline block is left as-is
    (already verified working) rather than refactored to share this, to
    avoid any risk of regressing something already confirmed correct.

    Safe to call more than once per page render -- re-injecting the same
    CSS rule has no visible effect beyond the harmless duplication."""
    st.markdown(
        """
        <style>
        div[data-testid="stAlert"] p,
        div[data-testid="stMarkdownContainer"] p {
            text-align: justify;
            font-size: 1rem !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def inject_metric_card_css():
    """CSS for render_metric_card()'s header+table styling. Split out as its
    own injector, same convention as inject_option_justify_css() -- safe to
    call more than once per page render, harmless duplication.

    Includes a max-480px media query that collapses the 3-column Metric/
    Value/Details table into a stacked block per row -- the reference this
    was modeled on is a wide desktop mockup, and a literal 3-column table at
    that width would either overflow or get uncomfortably cramped on a
    phone, which is this app's primary usage context (see project notes on
    mobile Chrome). The table headers hide in that stacked mode since
    "Metric"/"Value"/"Details" read as redundant labels once each row is
    already showing icon + label + value + detail stacked in that order."""
    st.markdown(
        """
        <style>
        .metric-card { border: 1px solid #e2e8f0; border-radius: 12px; overflow: hidden; margin-bottom: 20px; }
        .metric-card-header { padding: 14px 18px; display: flex; justify-content: space-between;
                               align-items: center; flex-wrap: wrap; gap: 4px; }
        .metric-card-header .mc-title { font-size: 1.15em; font-weight: 700; color: #1e293b; }
        .metric-card-header .mc-subtitle { color: #64748b; font-size: 0.85em; }
        .metric-card table { width: 100%; border-collapse: collapse; background: white; }
        .metric-card th { text-align: left; padding: 8px 14px; font-size: 0.78em; color: #64748b;
                           font-weight: 600; background: #f8fafc; }
        .metric-card td { padding: 12px 14px; border-top: 1px solid #eef2f6; vertical-align: top; }
        .metric-card .mc-label { font-weight: 600; color: #1e293b; white-space: nowrap; }
        .metric-card .mc-value { font-weight: 700; font-size: 1.05em; }
        .metric-card .mc-detail { color: #475569; font-size: 0.85em; }
        .metric-card-footer { margin: 0 18px 16px 18px; padding: 10px 14px; border-radius: 8px;
                               font-size: 0.85em; color: #334155; }
        @media (max-width: 480px) {
            .metric-card thead { display: none; }
            .metric-card table, .metric-card tbody, .metric-card tr, .metric-card td { display: block; width: 100%; }
            .metric-card tr { padding: 10px 14px; border-top: 1px solid #eef2f6; }
            .metric-card td { padding: 2px 0; border-top: none; }
            .metric-card .mc-detail { margin-top: 2px; }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def inject_button_tint_css(box_key, bg, border, text, hover_bg=None, hover_border=None, hover_text=None,
                            press_bg=None, press_border=None, press_text=None,
                            disabled_bg=None, disabled_border=None, disabled_text=None):
    """Injects this app's standard light-tint button styling for a single
    keyed st.container(key=box_key) wrapping a button -- background/border/
    text color, plus a :hover variant, and optional :active (press) and
    :disabled variants. Consolidates what used to be ~9 separately
    hand-written <style> blocks scattered across render_practice/
    render_question/render_report/render_bookmarks/main() (Submit/Skip/
    Restart, Next question, Start a new session, resume/discard session,
    Start / Restart Session, Back up my progress, Start practice with my
    bookmarks, and the top-nav tabs) -- all of them this exact same shape,
    just with different colors and box keys. A tweak to one hover shade
    used to mean finding and editing every copy by hand; this is now one
    place.

    Purely a refactor of how the CSS gets generated -- every call site
    below passes the same literal color values the old inline blocks used,
    so none of the actual rendered styling changes.

    box_key must match the key= given to the wrapping st.container() --
    Streamlit renders that as a `st-key-{box_key}` class on the container,
    which is what the `[class*="st-key-{box_key}"] button` selector below
    targets.

    hover_border/hover_text default to the base border/text (no change on
    hover) when omitted; hover_bg has no default since every real button
    here wants a visibly different hover fill. press_bg/press_border/
    press_text and disabled_bg/disabled_border/disabled_text are each
    optional -- omitted entirely (no :active / :disabled rule at all,
    rather than one that just repeats the base state) unless press_bg /
    disabled_bg is given, since only the top-nav tabs use a press state and
    only the Smart Quiz setup screen's "Start / Restart Session" button
    (the one button in the app that's ever actually disabled) needs a
    disabled state."""
    hover_border = hover_border if hover_border is not None else border
    hover_text = hover_text if hover_text is not None else text
    press_rule = ""
    if press_bg is not None:
        press_border = press_border if press_border is not None else press_bg
        press_text = press_text if press_text is not None else text
        press_rule = f"""
        [class*="st-key-{box_key}"] button:active {{
            background-color: {press_bg} !important;
            border-color: {press_border} !important;
            color: {press_text} !important;
        }}"""
    disabled_rule = ""
    if disabled_bg is not None:
        # opacity forced back to 1 -- Streamlit's own default disabled-dimming
        # would otherwise fade these explicit colors, defeating the point of
        # setting them at all.
        disabled_rule = f"""
        [class*="st-key-{box_key}"] button:disabled {{
            background-color: {disabled_bg} !important;
            border-color: {disabled_border} !important;
            color: {disabled_text} !important;
            opacity: 1 !important;
        }}"""
    st.markdown(
        f"""
        <style>
        [class*="st-key-{box_key}"] button {{
            background-color: {bg} !important;
            border-color: {border} !important;
            color: {text} !important;
        }}
        [class*="st-key-{box_key}"] button:hover {{
            background-color: {hover_bg} !important;
            border-color: {hover_border} !important;
            color: {hover_text} !important;
        }}{press_rule}{disabled_rule}
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_metric_card(icon, title, subtitle, header_bg, accent, rows, footer_lines=None):
    """Renders one styled Metric/Value/Details table -- Streamlit has no
    built-in widget for "colored header banner + a small table with a
    value pill and a comparison note per row", so this builds it directly
    via st.markdown(unsafe_allow_html=True) rather than forcing st.metric/
    st.dataframe into a shape they don't support.

    rows: list of dicts, each {"icon": emoji, "label": str, "value": str,
    "value_color": optional CSS color (defaults to dark slate), "detail":
    str shown in the third column, "—" if omitted}.

    footer_lines: optional list of strings, each rendered as its own tinted
    note below the table (header_bg reused as the note's background, so it
    reads as part of the same card rather than a separate element)."""
    inject_metric_card_css()
    rows_html = "".join(
        f'<tr><td class="mc-label">{r["icon"]} {r["label"]}</td>'
        f'<td class="mc-value" style="color:{r.get("value_color", "#1e293b")};">{r["value"]}</td>'
        f'<td class="mc-detail">{r.get("detail", "—")}</td></tr>'
        for r in rows
    )
    footer_html = "".join(
        f'<div class="metric-card-footer" style="background:{header_bg};">ℹ️ {line}</div>'
        for line in (footer_lines or [])
    )
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-card-header" style="background:{header_bg}; border-bottom:2px solid {accent};">
                <div class="mc-title">{icon} {title}</div>
                <div class="mc-subtitle">{subtitle}</div>
            </div>
            <table>
                <thead><tr><th>Metric</th><th>Value</th><th>Details / Comparison</th></tr></thead>
                <tbody>{rows_html}</tbody>
            </table>
            {footer_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_explanation_text(text):
    """Renders explanation text (question bank data or an admin's edit via
    render_explanation_editor()) with unsafe_allow_html=True, so a literal
    <br> tag -- a common leftover from pasting an AI tool's HTML/table
    output, e.g. "Section 16(ia):<br>" -- actually executes as a line
    break instead of showing as inert text (Streamlit's markdown escapes
    raw HTML by default).

    Deliberately still just st.markdown() on the ORIGINAL text, not routed
    through any external HTML pipeline first. An earlier version of this
    function pre-rendered the text via the `markdown` library (with nl2br/
    sane_lists extensions) before handing the resulting HTML to st.markdown
    -- that fixed the <br> issue but broke two things Streamlit's own
    native renderer already handled correctly: GFM-style pipe tables
    (python-markdown doesn't support those without a separate `tables`
    extension -- without it, a table silently degrades to one literal
    paragraph of "| a | b |" text) and inline/block LaTeX via $.../$$...$$
    (bypassed entirely once the text is pre-rendered to raw HTML, since
    Streamlit's KaTeX handling only runs on markdown it parses itself, not
    on HTML handed to it directly). Simply allowing HTML through
    Streamlit's OWN renderer, rather than swapping renderers, keeps both
    of those working exactly as before while still fixing <br>.

    Security note: unsafe_allow_html is scoped to this one function, used
    only for explanation text, which only admins can edit (see
    render_explanation_editor()'s _is_admin gate) -- same trust level this
    app already extends admins for answer-key and question-stem edits."""
    st.markdown(text, unsafe_allow_html=True)


def render_copy_button(label, text, key):
    """Renders a single-click 'copy to clipboard' button via a small HTML/
    JS component -- unlike st.code()'s built-in copy icon (used elsewhere
    in this app for "Copy question text"/"Copy this question"), this
    copies immediately on click with no expander to open first and no
    small icon to find inside a code block afterward.

    Re-added after being tried once before and briefly removed -- the
    first attempt wasn't found simply because of where it was placed
    (inside the explanation editor), not because it failed to render or
    work; confirmed rendering correctly via screenshot. Now used for
    "Copy question text for AI" in render_full_question() (Bookmarks tab).

    Tries the modern navigator.clipboard API first, and falls back to the
    older document.execCommand('copy') approach (via a temporary hidden
    textarea) if that throws -- the modern API can be blocked inside a
    sandboxed component iframe (which is exactly what st.components.v1.html
    renders into) depending on the browser, while execCommand works
    reliably in that same sandboxed context since it doesn't go through
    the Permissions API at all. Both attempts happen synchronously inside
    the same click handler, so the browser still treats it as a direct
    user gesture either way.

    text is JSON-encoded before being embedded in the <script> tag --
    that's what safely escapes quotes, newlines, and non-ASCII characters
    (₹, LaTeX backslashes, etc.) into a valid JS string literal. Never
    interpolate raw text directly into a <script> tag like this.

    key must be unique per call on the page (used as the button's DOM id).
    height is a little more generous than the button visually needs (60px
    vs. a roughly ~45-50px button+margin) since st.components.v1.html()
    fixes the iframe height and clips rather than reflows any overflow --
    better to have a few px of empty space below the button than risk
    clipping it on a browser/font combination that renders it slightly
    taller than expected."""
    import streamlit.components.v1 as components
    js_text = json.dumps(text)
    js_label = json.dumps(label)
    html = f"""
    <div style="margin: 0.25rem 0 0.5rem 0;">
        <button id="{key}" type="button" style="
            width: 100%;
            padding: 0.5rem 1rem;
            background-color: #eff6ff;
            border: 1px solid #93c5fd;
            border-radius: 0.5rem;
            color: #1d4ed8;
            font-size: 0.95rem;
            font-family: inherit;
            cursor: pointer;
        ">{label}</button>
    </div>
    <script>
    (function() {{
        const btn = document.getElementById({json.dumps(key)});
        const originalLabel = {js_label};
        const textToCopy = {js_text};
        btn.addEventListener('click', async function() {{
            let success = false;
            try {{
                await navigator.clipboard.writeText(textToCopy);
                success = true;
            }} catch (err) {{
                try {{
                    const ta = document.createElement('textarea');
                    ta.value = textToCopy;
                    ta.style.position = 'fixed';
                    ta.style.opacity = '0';
                    document.body.appendChild(ta);
                    ta.focus();
                    ta.select();
                    success = document.execCommand('copy');
                    document.body.removeChild(ta);
                }} catch (err2) {{
                    success = false;
                }}
            }}
            btn.innerText = success ? '✅ Copied!' : '⚠️ Copy failed — select and copy manually';
            setTimeout(function() {{ btn.innerText = originalLabel; }}, 1800);
        }});
    }})();
    </script>
    """
    components.html(html, height=60)


# ---------- User profiles (dropdown + optional PIN) ----------

def render_centered_header(caption_text):
    st.markdown(
        "<h1 style='text-align: center; margin-bottom: 0;'>📚 Algorithmic Quiz System</h1>"
        f"<p style='text-align: center; color: gray; font-size: 0.9rem; margin-top: 0.25rem;'>{caption_text}</p>",
        unsafe_allow_html=True,
    )


def canonical_username(name):
    """Case- and whitespace-insensitive identity for duplicate-profile
    checks only -- 'Sunny', 'sunny', and 'Sunny  ' should collide. The
    typed-in name is still what gets stored and displayed; this is never
    written anywhere, only compared."""
    return " ".join(name.strip().split()).casefold()


def _hash_pin(pin):
    """Salted SHA-256. The PIN is explicitly documented elsewhere as a light
    deterrent, not real security -- but there's no reason to keep it sitting
    in users.json (and in every GitHub backup of it) in plain text when
    hashing costs nothing."""
    salt = secrets.token_hex(8)
    digest = hashlib.sha256(f"{salt}:{pin}".encode()).hexdigest()
    return f"{salt}:{digest}"


def _verify_pin(stored, entered):
    if not stored:
        return False
    if ":" in stored:
        salt, _, digest = stored.partition(":")
        if len(salt) == 16:
            return hashlib.sha256(f"{salt}:{entered}".encode()).hexdigest() == digest
    # Legacy plaintext PIN from before hashing was added -- compare directly
    # so existing profiles created before this change still work. It gets
    # hashed automatically the next time that PIN is set.
    return stored == entered


def get_current_user():
    if "current_user" in st.session_state:
        return st.session_state.current_user

    render_centered_header(
        "UPSC / HCS Prelims GS Paper I practice. "
        "Home-practice tool only; not for use during the live exam."
    )

    users = load_users()
    responses = load_responses()
    known_names = sorted(set(list(users.keys()) + [r.get("user") for r in responses if r.get("user")]))
    at_capacity = len(known_names) >= MAX_USERS

    default_param = st.query_params.get("user", "")
    options = ["-- Choose --", "Create new profile..."] + known_names
    default_index = options.index(default_param) if default_param in options else 0

    choice = st.selectbox("Select your profile:", options, index=default_index)

    name = ""
    pin_ok = True
    new_pin = None
    set_pin = False
    can_create = True

    if choice == "Create new profile...":
        if at_capacity:
            st.error(
                f"This app is at its {MAX_USERS}-profile limit ({len(known_names)}/{MAX_USERS} registered). "
                "Ask the admin to free up a slot, or select an existing profile above."
            )
            can_create = False
        else:
            name = st.text_input("Enter your name:").strip()
            known_canonical = {canonical_username(k) for k in known_names}
            if name and canonical_username(name) in known_canonical:
                st.error(
                    f'A profile named "{name}" already exists (names are matched regardless of case '
                    "or spacing). Select it from the dropdown above instead — creating a duplicate "
                    "here would silently take over that profile's data."
                )
                can_create = False
            set_pin = st.checkbox("Protect this profile with a short PIN (optional — a light deterrent against accidental mix-ups, not real security)")
            if set_pin:
                new_pin = st.text_input("Set a PIN:", type="password", max_chars=6)
                pin_ok = bool(new_pin and new_pin.strip())
    elif choice != "-- Choose --":
        name = choice
        stored_pin = users.get(name)
        if stored_pin:
            entered_pin = st.text_input(f"Enter PIN for {name}:", type="password", max_chars=6)
            pin_ok = _verify_pin(stored_pin, entered_pin)
            if entered_pin and not pin_ok:
                st.error("Incorrect PIN.")

    if st.button("Continue", type="primary", disabled=not (can_create and name.strip() and pin_ok)):
        clean = name.strip()
        if choice == "Create new profile..." and set_pin and new_pin:
            save_user_pin(clean, new_pin.strip())
        st.session_state.current_user = clean
        st.query_params["user"] = clean
        st.rerun()

    st.caption("Tip: bookmark this page after selecting your profile — the link will remember you next time.")
    st.stop()


# ---------- GitHub-backed durable storage (best-effort, never breaks the app) ----------

_BACKUP_FILES = [
    (RESPONSES_FILE, "backup/responses.json"),
    (SCHEDULE_FILE, "backup/schedule.json"),
    (USERS_FILE, "backup/users.json"),
    (NOTES_FILE, "backup/community_notes.json"),
    (RESOLVED_FILE, "backup/resolved.json"),
    (BOOKMARKS_FILE, "backup/bookmarks.json"),
    (EDITS_FILE, "backup/edit_log.json"),
    (ACTIVE_SESSIONS_FILE, "backup/active_sessions.json"),
]


def _is_admin(user):
    try:
        admin = st.secrets.get("admin_user")
    except Exception:
        admin = None
    return bool(admin) and user == admin


# How many times _github_commit_file() retries a genuine write conflict
# (HTTP 409 -- another commit landed between its own GET and PUT) before
# giving up. 3 total attempts: the original try plus 2 retries.
_GITHUB_CONFLICT_RETRY_ATTEMPTS = 3


def _github_config():
    if requests is None:
        return None
    try:
        token = st.secrets.get("github_token")
        repo = st.secrets.get("github_repo")
    except Exception:
        return None
    if not token or not repo:
        return None
    return {
        "headers": {"Authorization": f"token {token}", "Accept": "application/vnd.github+json"},
        "repo": repo,
    }


def github_restore_if_needed():
    cfg = _github_config()
    if not cfg:
        return
    for local_path, repo_path in _BACKUP_FILES:
        if local_path.exists():
            continue
        try:
            url = f"https://api.github.com/repos/{cfg['repo']}/contents/{repo_path}"
            resp = requests.get(url, headers=cfg["headers"], timeout=10)
            if resp.status_code == 200:
                content_b64 = resp.json().get("content", "").replace("\n", "")
                content = base64.b64decode(content_b64).decode("utf-8")
                local_path.write_text(content, encoding="utf-8")
        except Exception:
            pass


def _github_commit_file(local_path, repo_path, message):
    """Pushes local_path's current content to repo_path in the repo via the
    GitHub Contents API (create-or-update, using the existing file's sha if
    present so this is a proper update rather than a conflicting create).
    Returns (True, "Success") or (False, "specific error message"). Never
    raises -- shared by github_backup() (state files -> backup/*.json) and
    the in-app typo-fix save (question-bank files -> data/*.json).

    Retries on a 409 specifically, up to _GITHUB_CONFLICT_RETRY_ATTEMPTS
    times: a 409 here means another commit landed between our own GET (to
    read the current sha) and our PUT -- most likely another user's
    session-complete backup firing at nearly the same moment, since
    github_backup() runs on every session completion across every user of
    this deployment. That's a genuine, expected write race under real
    multi-user load, not a real error -- re-fetching a fresh sha and trying
    again resolves it almost every time without the person needing to
    manually retry via "Back up my progress now." Any other non-200/201
    status (auth, permissions, rate limiting, etc.) fails immediately
    without retrying, since a retry wouldn't help those."""
    cfg = _github_config()
    if not cfg:
        return False, "Backup not configured (missing github_token/github_repo secrets)."
    try:
        content = local_path.read_text(encoding="utf-8")
        b64 = base64.b64encode(content.encode("utf-8")).decode("ascii")
        url = f"https://api.github.com/repos/{cfg['repo']}/contents/{repo_path}"
        payload_base = {"message": message, "content": b64}

        for attempt in range(_GITHUB_CONFLICT_RETRY_ATTEMPTS):
            get_resp = requests.get(url, headers=cfg["headers"], timeout=10)
            sha = get_resp.json().get("sha") if get_resp.status_code == 200 else None
            payload = dict(payload_base)
            if sha:
                payload["sha"] = sha
            put_resp = requests.put(url, headers=cfg["headers"], json=payload, timeout=10)
            if put_resp.status_code in (200, 201):
                return True, "Success"
            if put_resp.status_code != 409:
                return False, f"GitHub API returned {put_resp.status_code} while pushing {local_path.name}."
            # 409 -- another commit won the race. Loop and try again with a
            # freshly-fetched sha, unless this was the last attempt.
        return False, (
            f"GitHub API returned 409 while pushing {local_path.name} after "
            f"{_GITHUB_CONFLICT_RETRY_ATTEMPTS} attempts (repeated write conflicts -- "
            "likely heavy concurrent activity right now; a manual retry may still help)."
        )
    except requests.exceptions.Timeout:
        return False, f"Connection timed out while pushing {local_path.name}."
    except requests.exceptions.ConnectionError:
        return False, f"Network error while pushing {local_path.name}."
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"


def github_backup():
    """Returns (True, "Success") or (False, "specific error message(s)"). Never
    raises. Best-effort across every file: one file's push failing (a
    timeout, a network blip) doesn't stop the rest from being attempted --
    they're independent files with independent GitHub API calls, so a
    problem pushing responses.json shouldn't also leave schedule.json,
    users.json, etc. un-backed-up for no reason. If one or more fail, every
    failure is collected and reported together rather than only the first,
    so a retry (or a look at Developer Mode) shows the full picture instead
    of just whichever file happened to be first in _BACKUP_FILES."""
    cfg = _github_config()
    if not cfg:
        return False, "Backup not configured (missing github_token/github_repo secrets)."
    failures = []
    for local_path, repo_path in _BACKUP_FILES:
        if not local_path.exists():
            continue
        message = f"Backup {local_path.name} - {datetime.now().isoformat()}"
        ok, msg = _github_commit_file(local_path, repo_path, message)
        if not ok:
            failures.append(f"{local_path.name}: {msg}")
    if failures:
        return False, "; ".join(failures)
    return True, "Success"


# ---------- Concurrency-safe file access ----------
# Multiple colleagues can submit answers at the same moment. A plain
# read-modify-write on a shared JSON file can silently lose one person's
# write if two happen close together. These helpers hold an exclusive OS
# file lock for the whole read-modify-write, closing that gap.

def _locked_read(path, default):
    if not path.exists():
        return default
    with open(path, "r", encoding="utf-8") as f:
        if fcntl:
            fcntl.flock(f, fcntl.LOCK_SH)
        try:
            raw = f.read().strip()
            return json.loads(raw) if raw else default
        finally:
            if fcntl:
                fcntl.flock(f, fcntl.LOCK_UN)


@contextlib.contextmanager
def _locked_file(path, default):
    """Yields the current contents (list/dict) for in-place mutation.
    Written back to disk automatically -- atomically -- when the 'with'
    block exits normally.

    Locking and writing are deliberately split across two different files:
    - A separate, never-replaced `<name>.lock` file is held exclusively for
      the whole read-modify-write. It's never touched by os.replace(), so
      every process locking `path` is always locking the *same* inode --
      unlike locking `path` itself, which would race the moment one writer
      replaces it with a new inode mid-operation (a second process that
      opened the old inode just before the swap would end up locking a
      now-orphaned file and, on its own write, silently overwrite the first
      writer's change with stale data).
    - The actual data is written to a temp file in the same directory,
      flushed, fsync'd, then moved into place with os.replace(), which is
      atomic on both POSIX and Windows. A process killed mid-write leaves
      either the untouched old file or the complete new one -- never a
      truncated or partially-written one.
    """
    lock_path = path.with_name(path.name + ".lock")
    lock_f = open(lock_path, "a+")
    try:
        if fcntl:
            fcntl.flock(lock_f, fcntl.LOCK_EX)
        if path.exists():
            raw = path.read_text(encoding="utf-8").strip()
            data = json.loads(raw) if raw else default
        else:
            data = default
        yield data

        fd, tmp_name = tempfile.mkstemp(
            dir=str(path.parent), prefix=path.name + ".", suffix=".tmp"
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as tmp_f:
                json.dump(data, tmp_f, indent=2, ensure_ascii=False)
                tmp_f.flush()
                os.fsync(tmp_f.fileno())
            os.replace(tmp_name, path)
        except BaseException:
            with contextlib.suppress(OSError):
                os.remove(tmp_name)
            raise
    finally:
        if fcntl:
            fcntl.flock(lock_f, fcntl.LOCK_UN)
        lock_f.close()


# ---------- Data loading ----------

@st.cache_data
def load_questions():
    """Loads and merges every *.json question-bank file in /data, de-duplicating
    by question_id (the version with an explanation present wins, as before).

    Also detects the case that de-duplication-by-explanation used to paper
    over silently: two copies of the same question_id (e.g. a stale file left
    in /data) that actually *disagree* on the answer. That's not a routine
    duplicate, it's an unresolved answer-key conflict, and per this project's
    own rule this app should never silently pick a side. Any question_id with
    a genuine answer disagreement is tagged "_conflict" (excluded from scored
    practice via is_valid_for_practice) and reported back for Developer Mode
    to surface, the same way every other disputed answer in this bank is
    tracked rather than quietly resolved.

    Also validates that every record carries the fields the rest of the app
    accesses with direct (non-.get) bracket indexing -- question, exam,
    paper, year, theme, and a list-typed options -- not just question_id.
    Without this, a single record missing one of these could crash a whole
    screen for every user (e.g. render_practice's exam/year dropdowns build
    from every practice_questions entry at once), not just fail gracefully
    for that one question. A record failing this check is skipped and
    reported the same way an unparsable file or a missing question_id
    already is -- never silently dropped without a trace.

    Returns (questions, conflicts, skipped_records, source_file_by_id):
      questions        -- every merged question, including ones with no usable
                         answer key — callers building a scored practice
                         session should filter through is_valid_for_practice()
                         first; admin/review views intentionally see the full set.
      conflicts        -- [{"question_id", "sources": [{"file", "answer",
                         "has_explanation"}, ...]}] for every question_id whose
                         duplicate copies disagree on the answer.
      skipped_records  -- [{"file", "error", ...}] for anything excluded because
                         it couldn't be parsed, was missing question_id, or
                         failed the schema check above — never silently
                         dropped without a trace.
      source_file_by_id -- {question_id: filename} for exactly the copy that
                         won the merge into `questions` above -- i.e. whichever
                         file's version is actually shown/edited in the app.
                         apply_typo_fix() uses this instead of independently
                         re-scanning /data (see the now-removed approach in
                         _find_question_source_file()), since re-scanning
                         picks the first file containing the question_id
                         regardless of which copy actually won the merge --
                         when copies disagree on which one has an
                         explanation, that could silently write a typo fix
                         to a DIFFERENT file than the one being displayed,
                         so the fix would never show up.
    """
    REQUIRED_FIELDS = ("question", "exam", "paper", "year", "theme")
    by_id = {}
    sources = defaultdict(list)
    skipped_records = []
    source_file_by_id = {}
    for f in sorted(DATA_DIR.glob("*.json")):
        try:
            raw = json.loads(f.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError) as e:
            skipped_records.append({"file": f.name, "error": f"File failed to parse: {e}"})
            continue
        if not isinstance(raw, list):
            skipped_records.append({
                "file": f.name,
                "error": "Expected a JSON list of questions at the top level; skipped the whole file.",
            })
            continue
        for q in raw:
            qid = q.get("question_id") if isinstance(q, dict) else None
            if not qid:
                skipped_records.append({
                    "file": f.name,
                    "error": "Record missing question_id",
                    "record_preview": str(q)[:200],
                })
                continue
            missing = [
                field for field in REQUIRED_FIELDS
                if field not in q or q[field] in (None, "")
            ]
            if "options" not in q or not isinstance(q["options"], list):
                missing.append("options")
            if missing:
                skipped_records.append({
                    "file": f.name,
                    "error": f"{qid}: missing required field(s): {', '.join(missing)}",
                    "record_preview": str(q)[:200],
                })
                continue
            sources[qid].append((f.name, q))
            existing = by_id.get(qid)
            if existing is None or (not existing.get("explanation") and q.get("explanation")):
                by_id[qid] = q
                source_file_by_id[qid] = f.name

    conflicts = []
    for qid, copies in sources.items():
        if len(copies) < 2:
            continue
        answers = {q.get("answer") for _fname, q in copies if q.get("answer")}
        if len(answers) > 1:
            conflicts.append({
                "question_id": qid,
                "sources": [
                    {"file": fname, "answer": q.get("answer"), "has_explanation": bool(q.get("explanation"))}
                    for fname, q in copies
                ],
            })
            by_id[qid]["_conflict"] = True

    return list(by_id.values()), conflicts, skipped_records, source_file_by_id


# ---------- In-app typo fixes for the question bank (admin only) ----------
# Deliberately narrow: only the question stem and option wording can be edited
# here, never the answer, answer_status, or explanation. Those carry the
# fabrication-rule verification requirement (a primary source, not just a
# quick read of the text) and shouldn't be one tap away on a phone screen.
# This exists for the "that's obviously a typo" cases only.

def log_edit(user, question_id, field, old_value, new_value, source=None):
    """Every typo fix is recorded here regardless of whether the GitHub push
    succeeds — a silent, unlogged edit to the master question bank is exactly
    the kind of silent failure this project has otherwise been careful to
    avoid.

    source is optional and currently only set by apply_explanation_edit()
    when the saved text traces back to an accepted AI Explanation Import
    suggestion (see its ai_model parameter) -- e.g. "AI Explanation Import
    (gemini-2.5-flash)". Omitted (not just blank) for every other edit type,
    so old log entries and manual edits don't gain a stray null field."""
    with _locked_file(EDITS_FILE, []) as log:
        entry = {
            "timestamp": datetime.now().isoformat(),
            "user": user,
            "question_id": question_id,
            "field": field,
            "old_value": old_value,
            "new_value": new_value,
        }
        if source:
            entry["source"] = source
        log.append(entry)


def load_edit_log():
    return _locked_read(EDITS_FILE, [])


def apply_typo_fix(question_id, new_question_text, new_option_texts, user, source_file_by_id):
    """Writes only the fields that actually changed to the one /data file
    question_id lives in, logs each change, clears the load_questions() cache
    so the fix shows up immediately, then best-effort pushes that single file
    to GitHub (data/<filename>, not backup/<filename> -- this updates the
    actual source-of-truth question bank, not a state-file backup copy).

    source_file_by_id (from load_questions()) identifies which file to write
    to -- specifically, whichever copy actually won the merge and is being
    displayed/edited right now. This used to be looked up independently by
    re-scanning /data for the first file containing this question_id, which
    could land on a DIFFERENT file than the one actually shown when two
    duplicate copies disagree on which one has an explanation: the fix would
    write successfully, report success, and then never appear anywhere,
    since the merged view kept loading the untouched copy. Always pass the
    same source_file_by_id that was returned alongside the `questions` list
    this q came from, not a freshly-refetched one, so the two stay in sync.

    Returns a dict:
      {"error": "..."}                                   -- nothing was saved
      {"changed": []}                                     -- saved, but no
                                                               actual change
      {"changed": [...], "github_ok": bool, "github_msg": str} -- saved locally;
                                                               github_ok tells
                                                               you whether the
                                                               push also landed

    IMPORTANT: unlike responses/schedule/etc., /data files are NOT restored
    from GitHub on startup (they're expected to arrive via git deploy). If
    github_ok comes back False, the fix is NOT durable across a redeploy —
    it needs a retry (or a manual re-save) before that happens, whereas the
    other state files would silently recover on their own.
    """
    src_filename = source_file_by_id.get(question_id)
    if src_filename is None:
        return {"error": f"{question_id} not found in any /data file."}
    src_file = DATA_DIR / src_filename

    changed_fields = []
    with _locked_file(src_file, []) as data:
        target = next((q for q in data if q.get("question_id") == question_id), None)
        if target is None:
            return {"error": f"{question_id} disappeared from {src_file.name} mid-edit."}

        old_question = target.get("question", "")
        cleaned_question = (new_question_text or "").strip()
        if cleaned_question and cleaned_question != old_question:
            log_edit(user, question_id, "question stem", old_question, cleaned_question)
            target["question"] = cleaned_question
            changed_fields.append("question stem")

        for opt in target.get("options", []):
            label = opt.get("label")
            if label not in new_option_texts:
                continue
            old_text = opt.get("text", "")
            new_text = (new_option_texts[label] or "").strip()
            if new_text and new_text != old_text:
                log_edit(user, question_id, f"option {label}", old_text, new_text)
                opt["text"] = new_text
                changed_fields.append(f"option {label}")

    if not changed_fields:
        return {"changed": []}

    load_questions.clear()
    github_ok, github_msg = _github_commit_file(
        src_file, f"data/{src_file.name}",
        f"Typo fix: {question_id} ({', '.join(changed_fields)}) — {user}",
    )
    # Best-effort: also push the edit-log entries just written, immediately
    # rather than waiting for the next full github_backup() call. sync_
    # corrections.py (the local reconciliation script) reads this file from
    # GitHub, so a stale copy there would mean a local sync silently misses
    # this fix even though the actual data-file correction went through.
    _github_commit_file(
        EDITS_FILE, "backup/edit_log.json",
        f"Edit log — {question_id} ({', '.join(changed_fields)}) — {user}",
    )
    return {"changed": changed_fields, "github_ok": github_ok, "github_msg": github_msg}


def render_typo_editor(q, user, context, source_file_by_id):
    """Admin-only inline editor for fixing typos in a question's stem or
    option wording. Placed wherever a question already surfaces in the Needs
    Review dashboard, since that's where someone reviewing a flagged/disputed/
    incomplete question is most likely to also spot a typo.

    `context` is a short, call-site-specific tag (e.g. "flagged",
    "missing_key", "disputed") folded into every widget key here. A single
    question can legitimately appear in more than one Needs Review section at
    once (flagged AND missing its answer key, say), which would otherwise
    render this editor twice in the same page with identical keys --
    Streamlit requires every widget key to be unique per run, so without this
    the second render raises StreamlitDuplicateElementKey.

    source_file_by_id is load_questions()'s mapping of question_id -> the
    /data filename that actually won the merge for this question -- passed
    straight through to apply_typo_fix() so a save always lands on the same
    copy that's on screen right now, even when duplicate copies of this
    question_id disagree on which one to display."""
    with st.expander("✏️ Fix a typo"):
        st.caption(
            "Only wording can be fixed here — the answer key has its own "
            "editor below (a separate action, on purpose), and the "
            "answer_status field still goes through the normal "
            "source-verification workflow, not a quick in-app edit."
        )
        new_question = st.text_area(
            "Question text", value=q["question"], height=150,
            key=f"typofix_q_{context}_{q['question_id']}",
        )
        new_options = {}
        for opt in q.get("options", []):
            if not opt.get("text"):
                continue
            new_options[opt["label"]] = st.text_input(
                f"Option {opt['label']}", value=opt["text"],
                key=f"typofix_opt_{context}_{q['question_id']}_{opt['label']}",
            )
        if st.button("Save typo fix", key=f"typofix_save_{context}_{q['question_id']}"):
            result = apply_typo_fix(q["question_id"], new_question, new_options, user, source_file_by_id)
            if "error" in result:
                st.error(result["error"])
            elif not result["changed"]:
                st.info("No changes detected.")
            elif result["github_ok"]:
                st.success(f"Saved and synced to GitHub: {', '.join(result['changed'])}.")
                st.rerun()
            else:
                st.warning(
                    f"Saved locally ({', '.join(result['changed'])}), but the GitHub "
                    f"sync failed: {result['github_msg']}. This fix won't survive a "
                    "redeploy until it syncs — try again in a moment."
                )
                st.rerun()


def apply_explanation_edit(question_id, new_explanation_text, user, source_file_by_id, ai_model=None):
    """Admin-only: overwrites a question's explanation in the one /data file
    it lives in. Same read-modify-write / logging / cache-clear / best-effort
    GitHub push pattern as apply_typo_fix() -- see that docstring for the
    details on source_file_by_id and the "not durable until it syncs" caveat
    -- just scoped to the explanation field instead of the stem/options.

    Kept as its own function rather than folded into apply_typo_fix(): the
    explanation carries the fabrication-rule verification requirement (it
    should reflect a primary source, not just a quick rewrite), so this
    stays a distinct, separately-labeled action in both the UI and the edit
    log rather than blending into "Fix a typo," which is scoped to
    wording-only fixes with no such requirement.

    Like apply_typo_fix()'s text fields, a blank submission is treated as
    "nothing to save" rather than as clearing the explanation -- guards
    against an accidental empty save wiping out an existing explanation
    from a stray tap, at the cost of not being able to deliberately blank
    one out from this editor (edit it to a short placeholder instead, if
    that's ever actually needed).

    ai_model is optional -- when render_explanation_editor() passes one
    through (the resolved model_version from a "Use this" click on an AI
    Explanation Import suggestion), the log_edit() entry below records it as
    the edit's source, so the permanent edit log shows which model touched a
    given explanation even though the app itself always calls a "-latest"
    alias rather than a pinned version.

    Returns the same dict shape as apply_typo_fix():
      {"error": "..."}                                   -- nothing was saved
      {"changed": []}                                     -- saved, but no
                                                               actual change
      {"changed": [...], "github_ok": bool, "github_msg": str} -- saved locally;
                                                               github_ok tells
                                                               you whether the
                                                               push also landed
    """
    src_filename = source_file_by_id.get(question_id)
    if src_filename is None:
        return {"error": f"{question_id} not found in any /data file."}
    src_file = DATA_DIR / src_filename

    changed_fields = []
    with _locked_file(src_file, []) as data:
        target = next((q for q in data if q.get("question_id") == question_id), None)
        if target is None:
            return {"error": f"{question_id} disappeared from {src_file.name} mid-edit."}

        old_explanation = target.get("explanation", "")
        cleaned = (new_explanation_text or "").strip()
        if cleaned and cleaned != old_explanation:
            source = f"AI Explanation Import ({ai_model})" if ai_model else None
            log_edit(user, question_id, "explanation", old_explanation, cleaned, source=source)
            target["explanation"] = cleaned
            changed_fields.append("explanation")

    if not changed_fields:
        return {"changed": []}

    load_questions.clear()
    github_ok, github_msg = _github_commit_file(
        src_file, f"data/{src_file.name}",
        f"Explanation edit: {question_id} — {user}",
    )
    _github_commit_file(
        EDITS_FILE, "backup/edit_log.json",
        f"Edit log — {question_id} (explanation) — {user}",
    )
    return {"changed": changed_fields, "github_ok": github_ok, "github_msg": github_msg}


# ---------- AI Explanation Import (Gemini Flash) ----------
# One feature, two modes, picked automatically from whether the question
# already has an explanation on file: "generate" writes one from scratch,
# "improve" refines an existing draft. Deliberately a single button rather
# than two separate features -- the Question Bank tab's "Has explanation:
# No" filter and its "Yes" counterpart both land an admin on this same
# expander, so the button should just do the right thing either way rather
# than asking the admin to pick a mode themselves.
#
# The suggestion is never saved directly -- "Use this" only copies it into
# the existing editable text area below (see the explfix_{qid} session_state
# write in render_explanation_editor()), so the existing "Save explanation"
# button -- with its own GitHub commit / logging -- stays the one and only
# save path, and a human still has to actually click Save. This matches the
# caption immediately below it: explanation changes should trace back to a
# primary source, not just be accepted from an AI wholesale.

_AI_EXPLANATION_SCHEMA = {
    "type": "object",
    "properties": {
        "revised_explanation": {"type": "string"},
        "issues_found": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["revised_explanation", "issues_found"],
}


@st.cache_resource
def _gemini_client():
    """Cached Gemini API client. Returns None -- rather than raising -- if
    the google-genai package isn't installed or gemini_api_key isn't set in
    secrets, so callers can fail gracefully with a plain warning instead of
    crashing the whole expander. st.cache_resource builds this once per app
    session, not once per rerun."""
    if genai is None:
        return None
    try:
        api_key = st.secrets.get("gemini_api_key")
    except Exception:
        return None
    if not api_key:
        return None
    return genai.Client(api_key=api_key)


def _build_ai_explanation_prompt(q, mode):
    """mode is 'generate' (no explanation on file yet) or 'improve' (one
    already exists) -- same underlying request either way, just different
    framing and whether a CURRENT EXPLANATION line is included, so the two
    modes share one prompt-builder instead of drifting apart as separate copies."""
    options_text = "\n".join(f"{o.get('label')}) {o.get('text', '')}" for o in q.get("options", []))
    header = "Write a new explanation" if mode == "generate" else "Improve the existing explanation"
    current_line = "" if mode == "generate" else f"CURRENT EXPLANATION: {q.get('explanation', '')}\n"
    return (
        f"{header} for this MCQ from a {q.get('exam')} General Studies exam "
        f"(Paper {q.get('paper')}, theme: {q.get('theme')}).\n\n"
        f"QUESTION: {q.get('question', '')}\n"
        f"OPTIONS:\n{options_text}\n"
        f"CORRECT ANSWER: {q.get('answer')}\n"
        f"{current_line}\n"
        "Do NOT change or contradict: the correct answer, dates, names, "
        "places, article/section numbers, statistics, or other factual "
        "details implied by the question. Do not invent facts, dates, "
        "article numbers, or figures not implied by the question itself. "
        "If you are not confident a factual detail is correct, note it in "
        "issues_found instead of stating it as fact. Otherwise make the "
        "explanation clear, exam-oriented, and easy to remember. Return "
        "only the requested JSON."
    )


_GEMINI_RETRY_ATTEMPTS = 3
_GEMINI_RETRY_BACKOFF_SECONDS = (2, 4)  # waited before attempt 2 and attempt 3 respectively


def _is_transient_gemini_error(e):
    """True for a busy-server condition worth retrying (Google's "503
    UNAVAILABLE ... currently experiencing high demand" response, seen
    fairly often on a "-latest" Flash alias at peak times). Checked by
    substring on str(e) rather than a specific exception type/attribute --
    different google-genai SDK versions have surfaced this as different
    exception shapes, and a plain substring check is more resilient to that
    than depending on one exact class. Anything else (a bad API key, a
    malformed request, a quota error) is NOT retried -- see the loop below --
    since a retry wouldn't fix those and would only make the admin wait
    longer to see the real error."""
    msg = str(e).lower()
    return any(s in msg for s in ("503", "unavailable", "overloaded"))


def suggest_ai_explanation(q, mode):
    """Calls Gemini Flash for a suggested explanation. Returns
    {"revised_explanation": ..., "issues_found": [...], "model_version": ...}
    on success, or {"error": "..."} on any failure (missing config, network,
    bad response) -- callers show the error as a plain warning rather than
    crashing the admin-only explanation editor over a third-party API hiccup.

    Retries up to _GEMINI_RETRY_ATTEMPTS times, with a short backoff, but
    ONLY for a transient "server busy" condition (see
    _is_transient_gemini_error()) -- same philosophy as the GitHub 409
    write-conflict retry in _github_commit_file(): retry the specific,
    expected-to-clear-up failure automatically so the admin doesn't have to
    notice an error and manually re-click, but fail immediately on anything
    else rather than making a real problem take three times as long to report.

    model_version comes straight from the API response, not from the model=
    string we sent -- since that string is the "gemini-flash-latest" alias
    (see the model= argument below), this is the only way to know which
    actual model version answered a given call, both for display right after
    the call and for the provenance tag apply_explanation_edit() logs on save."""
    client = _gemini_client()
    if client is None:
        return {"error": "Gemini isn't configured — check gemini_api_key in secrets and that google-genai is installed."}
    last_error = None
    for attempt in range(_GEMINI_RETRY_ATTEMPTS):
        if attempt > 0:
            time.sleep(_GEMINI_RETRY_BACKOFF_SECONDS[attempt - 1])
        try:
            response = client.models.generate_content(
                model="gemini-flash-latest",
                contents=_build_ai_explanation_prompt(q, mode),
                config=genai_types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=_AI_EXPLANATION_SCHEMA,
                ),
            )
            data = json.loads(response.text)
            return {
                "revised_explanation": data.get("revised_explanation", ""),
                "issues_found": data.get("issues_found", []),
                "model_version": getattr(response, "model_version", None) or "unknown",
            }
        except Exception as e:
            last_error = e
            if not _is_transient_gemini_error(e):
                break
    return {"error": f"Gemini request failed: {last_error}"}


def render_explanation_editor(q, user, source_file_by_id):
    """Admin-only inline editor for correcting or filling in a question's
    explanation. Placed in the Bookmarks tab and the Question Bank tab,
    since reviewing a starred or flagged/missing-explanation question is
    exactly when a missing or wrong explanation is most likely to get
    noticed.

    Includes the AI Explanation Import button (see suggest_ai_explanation()
    above) -- generates a fresh explanation when there isn't one yet, or a
    suggested revision when there already is. Either way, "Use this" only
    populates the text area below; the existing "Save explanation" button
    is still the only thing that actually commits anything.

    source_file_by_id is load_questions()'s mapping of question_id -> the
    /data filename that actually won the merge for this question -- passed
    straight through to apply_explanation_edit() so a save always lands on
    the same copy that's on screen right now, same as render_typo_editor()."""
    qid = q["question_id"]
    with st.expander("💬 Edit Explanation"):
        st.caption(
            "Explanation changes should be based on a primary source, same "
            "as any answer-key correction — not just a quick rewrite."
        )

        mode = "improve" if (q.get("explanation") or "").strip() else "generate"
        if st.button("✨ AI Explanation Import", key=f"ai_import_go_{qid}"):
            with st.spinner("Generating explanation..." if mode == "generate" else "Improving explanation..."):
                st.session_state[f"ai_import_result_{qid}"] = suggest_ai_explanation(q, mode)

        ai_result = st.session_state.get(f"ai_import_result_{qid}")
        if ai_result:
            if "error" in ai_result:
                st.warning(ai_result["error"])
            else:
                label = "AI-generated explanation" if mode == "generate" else "AI-suggested revision"
                st.info(f"**{label}:**\n\n{ai_result['revised_explanation']}")
                st.caption(f"Model: {ai_result['model_version']}")
                if ai_result["issues_found"]:
                    st.warning("⚠️ Flagged for human review: " + "; ".join(ai_result["issues_found"]))
                ai_col1, ai_col2 = st.columns(2)
                with ai_col1:
                    if st.button("Use this", key=f"ai_import_use_{qid}", use_container_width=True):
                        st.session_state[f"explfix_{qid}"] = ai_result["revised_explanation"]
                        # Stashed separately from ai_import_result_{qid} (cleared
                        # below) so the model tag survives until the next Save
                        # click, even across the rerun "Use this" triggers.
                        st.session_state[f"ai_import_model_{qid}"] = ai_result["model_version"]
                        del st.session_state[f"ai_import_result_{qid}"]
                        st.rerun()
                with ai_col2:
                    if st.button("Discard", key=f"ai_import_discard_{qid}", use_container_width=True):
                        del st.session_state[f"ai_import_result_{qid}"]
                        st.rerun()

        new_explanation = st.text_area(
            "Explanation", value=q.get("explanation", ""), height=150,
            key=f"explfix_{qid}",
        )
        if st.button("Save explanation", key=f"explfix_save_{qid}"):
            # Popped (not just read) so this tag only ever attaches to the
            # very next save after a "Use this" -- a later, unrelated save on
            # this same question won't inherit a stale AI-assisted label.
            ai_model = st.session_state.pop(f"ai_import_model_{qid}", None)
            save_result = apply_explanation_edit(qid, new_explanation, user, source_file_by_id, ai_model=ai_model)
            if "error" in save_result:
                st.error(save_result["error"])
            elif not save_result["changed"]:
                st.info("No changes detected.")
            elif save_result["github_ok"]:
                st.success("Saved and synced to GitHub.")
                st.rerun()
            else:
                st.warning(
                    f"Saved locally, but the GitHub sync failed: {save_result['github_msg']}. "
                    "This won't survive a redeploy until it syncs — try again in a moment."
                )
                st.rerun()


def apply_answer_key_edit(question_id, new_answer_label, user, source_file_by_id):
    """Admin-only: overwrites a question's recorded answer key (the
    `answer` field -- which option label counts as correct) in the one
    /data file it lives in. Same read-modify-write / logging / cache-clear
    / best-effort GitHub push pattern as apply_typo_fix() and
    apply_explanation_edit() -- see apply_typo_fix()'s docstring for the
    source_file_by_id / "not durable until it syncs" details.

    Kept as its own action, distinct from both the wording-only typo fixer
    and the explanation editor: changing which option is correct is the
    single highest-stakes edit this app allows in-line -- it flips what
    every learner's past and future attempts on this question are scored
    against -- so it gets its own clearly-labeled, separately-logged
    action rather than being folded into either of the others.

    new_answer_label must name one of this question's options that
    actually has text (mirrors is_valid_for_practice()'s own requirement).
    render_answer_key_editor() is expected to only ever offer such labels
    as choices, but it's re-checked here too since an apply_* function
    shouldn't trust a stale or tampered-with caller.

    Returns the same dict shape as apply_typo_fix()/apply_explanation_edit():
      {"error": "..."}                                   -- nothing was saved
      {"changed": []}                                     -- saved, but no
                                                               actual change
      {"changed": [...], "github_ok": bool, "github_msg": str} -- saved locally;
                                                               github_ok tells
                                                               you whether the
                                                               push also landed
    """
    src_filename = source_file_by_id.get(question_id)
    if src_filename is None:
        return {"error": f"{question_id} not found in any /data file."}
    src_file = DATA_DIR / src_filename

    changed_fields = []
    with _locked_file(src_file, []) as data:
        target = next((q for q in data if q.get("question_id") == question_id), None)
        if target is None:
            return {"error": f"{question_id} disappeared from {src_file.name} mid-edit."}

        valid_labels = {o.get("label") for o in target.get("options", []) if o.get("text")}
        if new_answer_label not in valid_labels:
            return {"error": f"{new_answer_label!r} isn't a valid option label for {question_id}."}

        old_answer = target.get("answer")
        if new_answer_label != old_answer:
            log_edit(user, question_id, "answer key", old_answer, new_answer_label)
            target["answer"] = new_answer_label
            changed_fields.append("answer key")

    if not changed_fields:
        return {"changed": []}

    load_questions.clear()
    github_ok, github_msg = _github_commit_file(
        src_file, f"data/{src_file.name}",
        f"Answer key correction: {question_id} — {user}",
    )
    _github_commit_file(
        EDITS_FILE, "backup/edit_log.json",
        f"Edit log — {question_id} (answer key) — {user}",
    )
    return {"changed": changed_fields, "github_ok": github_ok, "github_msg": github_msg}


def render_answer_key_editor(q, user, context, source_file_by_id):
    """Admin-only inline editor for correcting which option is the recorded
    correct answer. Placed alongside render_typo_editor() wherever a
    question surfaces in the Needs Review dashboard -- a flagged, disputed,
    or missing/conflicting-key question is exactly where an answer-key
    correction is most likely to actually be needed, not just a wording
    fix.

    `context` is the same call-site tag render_typo_editor() uses ("flagged",
    "missing_key", "disputed"), folded into every widget key here for the
    same reason: a single question can appear in more than one Needs
    Review section at once, and Streamlit widget keys must stay unique per
    run.

    Kept as its own, more visually distinct action from "Fix a typo" -- see
    apply_answer_key_edit()'s docstring for why."""
    valid_options = [o for o in q.get("options", []) if o.get("text")]
    if not valid_options:
        st.caption(
            "🔑 No options with text exist for this question yet — fix the "
            "option wording above before an answer key can be set here."
        )
        return

    labels = [o["label"] for o in valid_options]
    text_by_label = {o["label"]: o["text"] for o in valid_options}
    current_answer = q.get("answer")
    default_index = labels.index(current_answer) if current_answer in labels else 0

    with st.expander("🔑 Edit answer key"):
        st.caption(
            "⚠️ This changes what every attempt on this question is scored "
            "against, past and future. Only correct this from a primary "
            "source (the Act/Rules text, not just re-reading the question) "
            "— this is the highest-stakes edit available in this app."
        )
        chosen_label = st.selectbox(
            "Correct option",
            labels,
            index=default_index,
            format_func=lambda l: f"{l}) {text_by_label[l]}",
            key=f"answerfix_{context}_{q['question_id']}",
        )
        if st.button("Save answer key", key=f"answerfix_save_{context}_{q['question_id']}"):
            result = apply_answer_key_edit(q["question_id"], chosen_label, user, source_file_by_id)
            if "error" in result:
                st.error(result["error"])
            elif not result["changed"]:
                st.info("No change — that's already the recorded answer.")
            elif result["github_ok"]:
                st.success("Saved and synced to GitHub.")
                st.rerun()
            else:
                st.warning(
                    f"Saved locally, but the GitHub sync failed: {result['github_msg']}. "
                    "This won't survive a redeploy until it syncs — try again in a moment."
                )
                st.rerun()


def is_valid_for_practice(q):
    """A question can only be scored if it has an answer that actually names
    one of its own options, AND that option actually has text. The text
    check matters because render_question/render_full_question/etc. all
    filter out options with empty text before showing them as choices (see
    their `if o["text"]` guards) -- so without this, a record whose
    correct-answer label points at a blank-text option would look "valid"
    here (the label technically matches) while the correct choice never
    actually renders as anything the learner could select, silently making
    the question unanswerable. A missing or defective-flagged answer
    (answer=None) has the same underlying failure mode -- it would
    silently score every response as wrong, no matter what the learner
    picks — see answer_status for *why* a question was excluded.

    Also excludes anything load_questions() flagged as "_conflict" -- two
    source copies disagreeing on the answer is not something to guess through
    silently in a live quiz; see Developer Mode for the details.

    Also excludes anything marked "disputed" -- a later reviewer found
    reason to doubt the recorded answer (see the Needs Review tab's
    Disputed Answers section and each question's dispute_note), which is
    the same "don't silently score against an answer we're not confident
    in" principle as _conflict and a missing/invalid key. Previously
    disputed questions stayed scored while under review; excluded by
    request so scoring and the personal-weakness/priority signals derived
    from it never rest on an answer someone's actively flagged as doubtful.
    Still fully visible in Needs Review either way -- this only affects
    whether it's eligible for scored practice."""
    if q.get("_conflict"):
        return False
    if q.get("disputed"):
        return False
    answer = q.get("answer")
    if not answer:
        return False
    matching_option = next((o for o in q.get("options", []) if o.get("label") == answer), None)
    return bool(matching_option and matching_option.get("text"))


def questions_flagged_by(user):
    """Question ids where `user` specifically has an open (unresolved)
    community flag. Used to keep a question you've personally flagged out
    of your OWN future practice sessions -- see is_valid_for_practice()'s
    docstring on why a flag alone doesn't exclude a question bank-wide (a
    flag is one person's unverified claim; letting it silently pull content
    for everyone would let anyone disable parts of the bank just by
    flagging things). But if you flagged a question yourself, you've
    already said you doubt its recorded answer, so there's no reason you
    personally should keep being served it -- and scored against that same
    answer -- while it's awaiting review. Once an admin resolves the flag
    (Question Bank's "Mark resolved"), it's back in your rotation too, same
    as everyone else's.

    Deliberately keyed off get_open_flags() (not the raw notes file), so a
    flag you raised that's already been resolved doesn't keep excluding the
    question for you forever."""
    return {
        qid for qid, flags in get_open_flags(load_notes(), load_resolved()).items()
        if any(f.get("user") == user for f in flags)
    }


@st.cache_data
def load_responses():
    return _locked_read(RESPONSES_FILE, [])


def save_response(entry):
    with _locked_file(RESPONSES_FILE, []) as responses:
        responses.append(entry)
    load_responses.clear()


@st.cache_data
def load_schedule():
    return _locked_read(SCHEDULE_FILE, {})


def save_schedule_entry(user, question_id, entry):
    with _locked_file(SCHEDULE_FILE, {}) as schedule:
        schedule[schedule_key(user, question_id)] = entry
    load_schedule.clear()


def update_schedule_atomic(user, question_id, correct, skipped, now, confidence=None):
    """Reads, computes, and writes the new spaced-repetition entry within a
    single lock acquisition — avoids the narrow gap of a separate read then
    a separate write racing against another submission for the same user.
    confidence is passed straight through to update_schedule_entry() (#6);
    omitting it preserves the exact previous behavior."""
    key = schedule_key(user, question_id)
    with _locked_file(SCHEDULE_FILE, {}) as schedule:
        schedule[key] = update_schedule_entry(schedule.get(key), correct, skipped, now, confidence=confidence)
    load_schedule.clear()


@st.cache_data
def load_users():
    return _locked_read(USERS_FILE, {})


def save_user_pin(name, pin):
    with _locked_file(USERS_FILE, {}) as users:
        users[name] = _hash_pin(pin)
    load_users.clear()


@st.cache_data
def load_notes():
    """Dict keyed by question_id -> list of {user, note, is_flag, timestamp}."""
    return _locked_read(NOTES_FILE, {})


def add_note(question_id, user, note_text, is_flag):
    with _locked_file(NOTES_FILE, {}) as notes:
        notes.setdefault(question_id, []).append({
            "user": user,
            "note": note_text,
            "is_flag": is_flag,
            "timestamp": datetime.now().isoformat(),
        })
    load_notes.clear()


@st.cache_data
def load_resolved():
    """Dict keyed by question_id -> timestamp it was last marked resolved.
    A flag posted AFTER that timestamp re-opens the question for review."""
    return _locked_read(RESOLVED_FILE, {})


def mark_resolved(question_id):
    with _locked_file(RESOLVED_FILE, {}) as resolved:
        resolved[question_id] = datetime.now().isoformat()
    load_resolved.clear()


def get_open_flags(questions_notes, resolved):
    """Returns {question_id: [flag_notes_posted_since_last_resolution]}"""
    open_flags = {}
    for qid, notes in questions_notes.items():
        flags = [n for n in notes if n.get("is_flag")]
        if not flags:
            continue
        resolved_at = resolved.get(qid, "")
        still_open = [f for f in flags if f["timestamp"] > resolved_at]
        if still_open:
            open_flags[qid] = still_open
    return open_flags


@st.cache_data
def load_bookmarks():
    """Dict keyed by user -> {question_id: timestamp bookmarked}."""
    return _locked_read(BOOKMARKS_FILE, {})


def toggle_bookmark(user, question_id):
    with _locked_file(BOOKMARKS_FILE, {}) as bookmarks:
        user_bookmarks = bookmarks.setdefault(user, {})
        if question_id in user_bookmarks:
            del user_bookmarks[question_id]
        else:
            user_bookmarks[question_id] = datetime.now().isoformat()
    load_bookmarks.clear()


# ---------- Resume-unfinished-session ----------
# st.session_state alone doesn't survive a page refresh or a dropped
# connection -- both real risks on a mobile home-screen shortcut over
# ordinary mobile data. This persists enough of an in-progress session to
# /data-adjacent active_sessions.json to rebuild it: the question_ids (not
# full question snapshots -- looked up fresh against the current bank on
# resume), how far in, the running results tally, and the revealed/
# selected_letter state of whichever question was current at save time.
# That last part used to be deliberately left out, on the theory that a
# resumed session could always safely land on its current question fresh
# and unrevealed -- but that's wrong exactly when a disconnect happens
# right after submitting an answer and before tapping "Next question":
# current_idx doesn't advance until Next is tapped, so without revealed/
# selected_letter a resume would show that already-answered question as
# blank and resubmittable, silently double-counting it if answered again.

@st.cache_data
def load_active_sessions():
    """Dict keyed by user -> {question_ids, current_idx, session_results,
    session_id, started_at}."""
    return _locked_read(ACTIVE_SESSIONS_FILE, {})


def save_active_session(user, session_questions, current_idx, session_results, session_id,
                         revealed=False, selected_letter=None):
    with _locked_file(ACTIVE_SESSIONS_FILE, {}) as sessions:
        started_at = sessions.get(user, {}).get("started_at") or datetime.now().isoformat()
        sessions[user] = {
            "question_ids": [q["question_id"] for q in session_questions],
            "current_idx": current_idx,
            "session_results": session_results,
            "session_id": session_id,
            "started_at": started_at,
            "revealed": revealed,
            "selected_letter": selected_letter,
        }
    load_active_sessions.clear()


def clear_active_session(user):
    with _locked_file(ACTIVE_SESSIONS_FILE, {}) as sessions:
        sessions.pop(user, None)
    load_active_sessions.clear()


# ---------- Spaced repetition ----------

def schedule_key(user, question_id):
    return f"{user}::{question_id}"


def update_schedule_entry(entry, correct, skipped, now, confidence=None):
    """`now` is a datetime, not a date -- stages are hour-based, so exact
    time of day matters for when something becomes due.

    confidence (#6): previously a confident-correct and a guessed-correct
    advanced the schedule identically -- the SRS engine ignored confidence
    entirely even though it's already captured and already used in the
    weakness math (see CONFIDENCE_ATTEMPT_WEIGHT/CONFIDENCE_SUCCESS_CREDIT
    on compute_theme_weakness()/compute_question_weakness()). That was an
    inconsistency worth closing: a lucky guess isn't the same evidence of
    mastery as a confident answer, and the scheduler should agree with the
    rest of the app on that.

    The stage ladder position (`stage`) still always advances by exactly 1
    on any correct answer, regardless of confidence -- this keeps history/
    streak tracking and the stage-based "how far along is this question"
    concept meaningful and comparable across confidence levels. What
    changes is how much of the new stage's longer interval the learner
    actually gets credited with: CONFIDENCE_SUCCESS_CREDIT (already used
    elsewhere for exactly this "how much does a correct answer at this
    confidence level count as real evidence" judgment) blends the due date
    between the OLD stage's interval and the FULL new stage's interval.
    A Confident answer (credit 1.0) gets the full interval -- byte-for-byte
    the same due date this function has always produced, so nothing changes
    for the majority of answers or for any caller that omits confidence
    (defaults to full credit, preserving old behavior exactly). A Guessed
    answer (credit 0.5) only gets roughly halfway there, so it comes back
    for a recheck meaningfully sooner even though the stage number moved on.

    Wrong and skipped answers are unaffected by confidence -- resetting to
    stage 0 (wrong) or holding the current stage due immediately (skip)
    already doesn't distinguish confidence levels in the review's own
    recommendation, and there's no meaningful "how confident were you in
    the wrong answer" signal worth encoding here."""
    stage = entry["stage"] if entry else -1
    if skipped:
        new_stage = stage if stage >= 0 else 0
        due = now
    elif correct:
        new_stage = min(stage + 1, len(STAGE_INTERVALS_HOURS) - 1)
        old_hours = STAGE_INTERVALS_HOURS[stage] if stage >= 0 else 0
        full_new_hours = STAGE_INTERVALS_HOURS[new_stage]
        credit = CONFIDENCE_SUCCESS_CREDIT.get(confidence, 1.0)
        due_hours = old_hours + credit * (full_new_hours - old_hours)
        due = now + timedelta(hours=due_hours)
    else:
        new_stage = 0
        due = now
    return {"stage": new_stage, "next_due": due.isoformat()}


def reconcile_schedule_from_responses(user):
    """Repairs any (user, question_id) pair that has a real response on
    record but no matching schedule entry -- the gap a crash between
    save_response() and update_schedule_atomic() could leave (see their call
    site in render_question(): they're two independent atomic writes, not
    one transaction). Left alone, that question would look "never
    attempted" to split_pool_by_schedule() and resurface as brand-new
    material in future sessions, even though a real response for it
    already exists -- a real but narrow correctness gap: the response
    history itself (and everything derived from it -- scoring, theme/
    question weakness) stays fully correct regardless, since none of that
    reads from schedule.json at all. Only this one question's spaced-
    repetition bucketing would be affected.

    Reconstructs the missing entry by replaying this user's FULL response
    history for that question_id, oldest first, back through update_
    schedule_entry() from scratch -- a single response in isolation isn't
    enough to know the right stage, since that depends on the whole prior
    sequence of hits and misses, which is exactly the history the missing
    entry would otherwise have accumulated.

    Called once per (browser session, user) at login (see main()), not on
    every session build -- this kind of gap should be rare, so a same-
    session fix is the right tradeoff against re-scanning full response
    history on every rerun."""
    responses = load_responses()
    schedule = load_schedule()
    by_question = defaultdict(list)
    for r in responses:
        if r.get("user") == user and r.get("question_id"):
            by_question[r["question_id"]].append(r)

    missing_qids = [qid for qid in by_question if schedule_key(user, qid) not in schedule]
    if not missing_qids:
        return

    for qid in missing_qids:
        # Oldest first -- save_response() always appends chronologically,
        # but sort explicitly rather than trust incoming order, since this
        # only runs for a genuinely exceptional gap and correctness matters
        # more than speed here.
        history = sorted(by_question[qid], key=lambda r: r.get("timestamp", ""))
        entry = None
        for r in history:
            correct = r.get("selected") == r.get("correct_answer")
            skipped = r.get("selected") == OPT_SKIPPED
            try:
                ts = datetime.fromisoformat(r.get("timestamp"))
            except (ValueError, TypeError):
                ts = datetime.now()
            entry = update_schedule_entry(
                entry, correct, skipped, ts,
                confidence=None if skipped else r.get("confidence"),
            )
        save_schedule_entry(user, qid, entry)


def is_due(entry, now):
    """`now` is a datetime. Parses next_due leniently: schedule entries
    written before the hour-based rescale stored a plain date
    ("2026-08-29"); datetime.fromisoformat() parses that as midnight of
    that day, so old entries keep working with no migration step."""
    if entry is None:
        return True
    return datetime.fromisoformat(entry["next_due"]) <= now


def has_full_coverage(responses, user, pool):
    """True once this user has attempted -- correctly or not -- every
    question in `pool` at least once. Used to hold back re-showing a
    correctly-answered question until the whole eligible pool (the current
    exam/year/theme filter selection) has been seen, so a question whose
    spaced-repetition due-date has arrived doesn't resurface known material
    ahead of something the learner hasn't encountered at all yet."""
    attempted_ids = {r["question_id"] for r in responses if r.get("user") == user}
    return all(q["question_id"] in attempted_ids for q in pool)


def last_attempt_correct(responses, user):
    """Dict question_id -> was this user's MOST RECENT attempt on it
    correct? (False for a skip, or if never attempted.) Responses are
    appended in chronological order (save_response() always appends), so
    simply overwriting as we iterate leaves each question_id pointing at
    its last -- i.e. most recent -- attempt."""
    result = {}
    for r in responses:
        if r.get("user") != user:
            continue
        qid = r.get("question_id")
        if qid:
            result[qid] = r.get("selected") == r.get("correct_answer")
    return result


def last_session_accuracy(responses, user):
    """This user's score (correct / total, 0..1) on their most recent
    session -- identified by whichever session_id has the latest response
    timestamp -- or None if they have no recorded attempts at all yet.
    Drives dynamic_review_share() below: a 7/10 (70%) last session means
    the next one reserves about 70% of its slots for due re-reviews.

    A skipped answer counts as wrong here, same as every other scoring
    surface in the app (the "Session complete" screen, Report tab, etc.) --
    0 credit, same as the real exam. Responses missing a correct_answer
    (an answer key that was valid at attempt time but has since been
    pulled under review) are excluded from both the numerator and
    denominator, same as Report's "graded" filter, so a since-invalidated
    question doesn't silently count against this session's score.

    Looks at whatever responses exist for that session_id regardless of
    whether the session was formally completed -- an abandoned session
    still reflects real, current performance, and there's no reason to
    prefer stale data from an older completed session over it."""
    user_responses = [r for r in responses if r.get("user") == user and r.get("session_id")]
    if not user_responses:
        return None
    latest_session_id = max(user_responses, key=lambda r: r.get("timestamp", ""))["session_id"]
    graded = [
        r for r in user_responses
        if r["session_id"] == latest_session_id and r.get("correct_answer")
    ]
    if not graded:
        return None
    correct = sum(1 for r in graded if r.get("selected") == r.get("correct_answer"))
    return correct / len(graded)


def wrong_streak_counts(responses, user):
    """Dict question_id -> this user's CURRENT consecutive wrong-answer
    streak (attempts since the last correct answer, or since the start of
    history if never correct). Resets to 0 the moment they get it right
    again -- a question that was rocky early on but has since been
    answered correctly shouldn't stay permanently flagged as a trouble
    spot. Skips are excluded entirely (neither reset nor extend the streak
    -- skipping isn't evidence of anything either way). Responses are
    appended in chronological order (save_response() always appends), so
    iterating in file order naturally processes each question's attempts
    oldest-to-newest."""
    streaks = defaultdict(int)
    for r in responses:
        if r.get("user") != user:
            continue
        qid = r.get("question_id")
        selected = r.get("selected")
        correct_answer = r.get("correct_answer")
        if not qid or selected == OPT_SKIPPED or not correct_answer:
            continue
        streaks[qid] = 0 if selected == correct_answer else streaks[qid] + 1
    return dict(streaks)


def last_n_sessions_wrong_questions(responses, questions_by_id, user, n=5):
    """This user's n most recent quiz sessions, each paired with the full
    question dicts they got wrong in it. No new tracking needed for this --
    every response already carries a session_id (an ISO timestamp captured
    once at session start, see render_practice()) and a timestamp, so a
    "quiz" is just whatever shares a session_id, and "most recent" is a
    plain sort on that same ISO string.

    "Wrong" uses the same convention as wrong_streak_counts(): selected !=
    correct_answer, with skips excluded entirely (a skip isn't a wrong
    answer, it's a non-answer). Deduplicated per session in case the same
    question is somehow served twice in one sitting.

    Includes every session with at least one logged response, including
    ones abandoned partway through -- responses.json has no separate
    "completed" marker to tell those apart from a finished session, and a
    partial attempt's wrong answers are just as worth reviewing. A session
    where everything was answered correctly still appears, just with an
    empty question list, so "5 most recent sessions" stays an honest count
    rather than silently becoming "5 most recent sessions with a mistake."

    Returns a list of (session_id, [question dicts]) tuples, most recent
    first. Historical question_ids no longer present in the current bank
    (e.g. since removed) are silently dropped from the question list rather
    than erroring."""
    sessions = defaultdict(list)
    for r in responses:
        if r.get("user") == user and r.get("session_id"):
            sessions[r["session_id"]].append(r)

    result = []
    for session_id in sorted(sessions, reverse=True)[:n]:
        wrong_ids = []
        for r in sessions[session_id]:
            qid, selected, correct_answer = r.get("question_id"), r.get("selected"), r.get("correct_answer")
            if not qid or selected == OPT_SKIPPED or not correct_answer:
                continue
            if selected != correct_answer and qid not in wrong_ids:
                wrong_ids.append(qid)
        result.append((session_id, [questions_by_id[qid] for qid in wrong_ids if qid in questions_by_id]))
    return result


def review_priority_bonus(question_id, wrong_streaks, question_weakness=None):
    """Combined review-priority bonus for one question, in hours-equivalent
    units (see the WRONG_STREAK_BONUS_*/QUESTION_WEAKNESS_BONUS_SCALE
    constants). Every consecutive wrong answer contributes, starting from
    the very first one (streak=1 -> WRONG_STREAK_BONUS_PER_MISS, scaling up
    to WRONG_STREAK_BONUS_CAP by streak 3) -- this used to gate out a lone
    miss on the theory that the automatic due-date reset already handled
    it, but that only worked back when due-reviews competed solely against
    each other for leftover slots. Now that due-reviews compete for a
    fixed, limited REVIEW_SLOT_SHARE of every session (see build_session_
    from_filters), a fresh single miss needs a real, immediate signal to
    win one of those few slots against older overdue backlog -- "marked due
    but zero bonus" wasn't enough.

    question_weakness (optional; see compute_question_weakness()) adds a
    second, longer-memory signal on top of the streak: a Bayesian-smoothed,
    recency-weighted measure of how this exact learner has done on this
    exact question across their WHOLE history with it, not just their
    current run. Unlike the streak, it doesn't reset to 0 the instant a
    single correct answer lands, so a question missed repeatedly over weeks
    (with the odd lucky guess in between) still carries some weight even
    mid-streak. Left as None, behavior is unchanged from before this signal
    existed -- kept optional so existing callers/tests built around the
    2-argument form keep working.

    NOTE: bookmarking a question has NO effect on this score (see the
    BOOKMARK_BONUS removal note next to the constants above) -- a star is
    purely a personal flag for the dedicated Bookmarks-tab session.

    Both signals stack: a question on a current wrong streak AND with a
    weak longer-run history is more worth prioritizing than either alone."""
    streak = wrong_streaks.get(question_id, 0)
    bonus = min(streak * WRONG_STREAK_BONUS_PER_MISS, WRONG_STREAK_BONUS_CAP)
    if question_weakness:
        bonus += question_weakness.get(question_id, 0.0) * QUESTION_WEAKNESS_BONUS_SCALE
    return bonus


def split_pool_by_schedule(schedule, user, pool, responses, wrong_streaks, question_weakness=None, now=None):
    """Splits a filtered question pool into three buckets against this user's
    spaced-repetition schedule:
      - new: never attempted -- no schedule entry exists yet.
      - overdue: attempted before, due date has arrived. Sorted by a
        BLENDED priority score, highest first: raw overdue age (in hours)
        plus a bonus for a current wrong-answer streak and/or a weak
        longer-run personal history with this exact question (see
        review_priority_bonus()). This is deliberately a score, not a
        hard "priority items always first" tier -- overdue age keeps
        accumulating without a ceiling, so a genuinely stale backlog item
        still eventually outranks any bonus, however large; a couple of
        recent misses shifts a question up the queue rather than letting
        it starve everything older. (Bookmarking a question has no effect
        here -- see review_priority_bonus().)
      - not_due: attempted before, scheduled for a later date -- successfully
        banked for now. ALSO holds a question whose due-date has arrived but
        was last answered correctly, for as long as this user hasn't yet
        attempted every question in `pool` at least once (see
        has_full_coverage) -- a correctly-known question waits its turn
        behind full coverage of the bank rather than resurfacing ahead of
        material the learner hasn't seen at all. A wrong or skipped answer
        is exempt from this hold-back and always surfaces on schedule,
        since that specifically needs reinforcement regardless of coverage.
    Both build_session_from_filters (session sampling) and render_practice's
    setup screen (the due-count breakdown) use this, so "how many are due"
    and "what a session actually pulls from" always agree.

    question_weakness is optional and simply forwarded to
    review_priority_bonus() -- see there. Omitting it doesn't change which
    bucket anything lands in (only review_priority_bonus's score, which
    only affects sort order within overdue and sampling weight within
    not_due), so existing callers built around the 5-argument form keep
    working unchanged.

    now is optional (defaults to datetime.now()), matching the pattern
    already used by compute_theme_weakness()/compute_question_weakness()/
    compute_question_stats() -- added so this function can be tested
    deterministically (and replayed against historical data, e.g. for a
    before/after comparison when scheduling logic changes) without needing
    to monkeypatch datetime.now() itself."""
    now = now or datetime.now()
    full_coverage = has_full_coverage(responses, user, pool)
    last_correct = last_attempt_correct(responses, user)
    new_items, overdue_pairs, not_due = [], [], []
    for q in pool:
        qid = q["question_id"]
        entry = schedule.get(schedule_key(user, qid))
        if entry is None:
            new_items.append(q)
        elif is_due(entry, now):
            if last_correct.get(qid) and not full_coverage:
                not_due.append(q)
            else:
                overdue_age_hours = (now - datetime.fromisoformat(entry["next_due"])).total_seconds() / 3600
                score = overdue_age_hours + review_priority_bonus(qid, wrong_streaks, question_weakness)
                overdue_pairs.append((score, q))
        else:
            not_due.append(q)
    overdue_pairs.sort(key=lambda pair: pair[0], reverse=True)  # highest priority score first
    overdue = [q for _, q in overdue_pairs]
    return new_items, overdue, not_due


# ---------- Adaptive weighting ----------

def _recency_weight(timestamp_str, now=None):
    """Exponential decay: a response's pull on the weakness score halves every
    RECENCY_HALF_LIFE_DAYS days. Recent mistakes should count more than old
    ones — someone who struggled with TDS three months ago but has since
    drilled it shouldn't keep reading as 'weak in TDS' forever. Missing or
    unparseable timestamps (shouldn't happen for real responses — save_response
    always writes one) fall back to full weight rather than being discounted."""
    if not timestamp_str:
        return 1.0
    try:
        ts = datetime.fromisoformat(timestamp_str)
    except (ValueError, TypeError):
        return 1.0
    now = now or datetime.now()
    days_ago = max((now - ts).total_seconds() / 86400.0, 0.0)
    return 0.5 ** (days_ago / RECENCY_HALF_LIFE_DAYS)


def compute_theme_weakness(responses, user, now=None):
    """Normalized, Bayesian-smoothed weakness per theme, roughly in [0, 1]
    (0 = fully mastered, 1 = fully weak) -- NOT an accumulating error total.

    The earlier version summed a per-response error/bonus score with no
    denominator, so a heavily-drilled theme with a handful of recent slips
    could out-rank a barely-touched theme that's actually weaker just by
    having more history (100 attempts/20 recent errors would outscore 8
    attempts/5 recent errors, even though the second theme's actual error
    *rate* is far worse). This version tracks a recency- and confidence-
    weighted mastery ratio (weighted_correct / weighted_attempts) instead,
    so theme size stops mattering on its own.

    Confidence still matters, just differently: a confident-and-correct
    answer counts as stronger evidence of mastery than a lucky guess, and a
    guess (right or wrong) carries weaker evidence either way than an answer
    given with actual conviction -- see CONFIDENCE_ATTEMPT_WEIGHT/
    CONFIDENCE_SUCCESS_CREDIT. Skips count as a full-weight miss, same as
    the app's exam-mode scoring elsewhere.

    A theme with only one or two attempts is pulled toward the neutral
    MASTERY_PRIOR rather than swinging straight to 0.0 or 1.0 on a tiny
    sample -- the same reason two heads in a row doesn't prove a coin is
    rigged."""
    weighted_correct = defaultdict(float)
    weighted_attempts = defaultdict(float)
    for r in responses:
        if r.get("user") != user:
            continue
        theme = r.get("theme")
        if not theme:
            continue
        w = _recency_weight(r.get("timestamp"), now)

        if r.get("selected") == OPT_SKIPPED:
            weighted_attempts[theme] += w
            continue

        correct_answer = r.get("correct_answer")
        if not correct_answer:
            continue
        confidence = r.get("confidence") or CONF_SOMEWHAT
        attempt_w = w * CONFIDENCE_ATTEMPT_WEIGHT.get(confidence, 1.0)
        weighted_attempts[theme] += attempt_w
        if r.get("selected") == correct_answer:
            weighted_correct[theme] += attempt_w * CONFIDENCE_SUCCESS_CREDIT.get(confidence, 1.0)

    scores = {}
    for theme, attempts in weighted_attempts.items():
        smoothed_mastery = (
            weighted_correct[theme] + MASTERY_PRIOR * MASTERY_PRIOR_STRENGTH
        ) / (attempts + MASTERY_PRIOR_STRENGTH)
        scores[theme] = 1.0 - smoothed_mastery
    return scores


def compute_subtopic_weakness(responses, questions_by_id, user, now=None):
    """Same Bayesian-smoothed, recency- and confidence-weighted mastery math
    as compute_theme_weakness(), one level finer: grouped by the schema's
    `subtopic` field (e.g. "194J" within the "TDS" theme) instead of theme.
    The schema has carried a subtopic field since the CBDT MVP was built,
    but nothing in the app read it until now -- this is the direct answer
    to "overall TDS performance is fine, but 194J-style questions keep
    getting missed," which theme-level weakness alone can't distinguish.

    Unlike theme (which is denormalized onto each response at attempt
    time -- see save_response()), subtopic isn't stored on individual
    responses, so this looks it up live via questions_by_id instead of
    reading it off the response record. That means this always reflects
    each question's CURRENT subtopic classification in the bank, including
    for old responses -- if a question is ever reclassified into a
    different subtopic, its whole response history moves with it, unlike
    theme weakness which stays pinned to whatever theme was recorded at
    attempt time. questions_by_id should map question_id -> the full
    question dict (as returned by load_questions()); a response whose
    question_id isn't found (a question since removed from the bank) or
    whose matched question has no subtopic is simply skipped, same as a
    response with no theme is skipped in compute_theme_weakness()."""
    weighted_correct = defaultdict(float)
    weighted_attempts = defaultdict(float)
    for r in responses:
        if r.get("user") != user:
            continue
        qid = r.get("question_id")
        q = questions_by_id.get(qid) if qid else None
        subtopic = q.get("subtopic") if q else None
        if not subtopic:
            continue
        w = _recency_weight(r.get("timestamp"), now)

        if r.get("selected") == OPT_SKIPPED:
            weighted_attempts[subtopic] += w
            continue

        correct_answer = r.get("correct_answer")
        if not correct_answer:
            continue
        confidence = r.get("confidence") or CONF_SOMEWHAT
        attempt_w = w * CONFIDENCE_ATTEMPT_WEIGHT.get(confidence, 1.0)
        weighted_attempts[subtopic] += attempt_w
        if r.get("selected") == correct_answer:
            weighted_correct[subtopic] += attempt_w * CONFIDENCE_SUCCESS_CREDIT.get(confidence, 1.0)

    scores = {}
    for subtopic, attempts in weighted_attempts.items():
        smoothed_mastery = (
            weighted_correct[subtopic] + MASTERY_PRIOR * MASTERY_PRIOR_STRENGTH
        ) / (attempts + MASTERY_PRIOR_STRENGTH)
        scores[subtopic] = 1.0 - smoothed_mastery
    return scores


def compute_question_weakness(responses, user, now=None):
    """Normalized, Bayesian-smoothed personal weakness per question_id --
    0 = this user has fully mastered this exact question, 1 = fully weak --
    identical math to compute_theme_weakness() just grouped by question_id
    instead of theme. This is the direct answer to "prioritize questions
    I've personally gotten wrong" as opposed to "prioritize weak themes":
    it only reflects THIS user's own history with THIS exact question, not
    a category-level aggregate and not other users' history with it (that's
    compute_question_stats(), a different, cohort-wide notion of difficulty).

    Like theme weakness, recent attempts count more than old ones (see
    _recency_weight) and a confident-correct answer counts as stronger
    evidence of mastery than a lucky guess (see CONFIDENCE_ATTEMPT_WEIGHT/
    CONFIDENCE_SUCCESS_CREDIT). The Bayesian smoothing toward MASTERY_PRIOR
    matters even more at this per-question grain than per-theme, since most
    questions will only ever have a handful of attempts from one learner --
    a single wrong answer shouldn't swing straight to "100% weak."

    Unlike wrong_streak_counts() (which hard-resets to 0 the moment a
    single correct answer lands), this fades out gradually: a question
    missed three times over two months with one lucky correct guess in
    between still reads as meaningfully weak, not freshly mastered."""
    weighted_correct = defaultdict(float)
    weighted_attempts = defaultdict(float)
    for r in responses:
        if r.get("user") != user:
            continue
        qid = r.get("question_id")
        if not qid:
            continue
        w = _recency_weight(r.get("timestamp"), now)

        if r.get("selected") == OPT_SKIPPED:
            weighted_attempts[qid] += w
            continue

        correct_answer = r.get("correct_answer")
        if not correct_answer:
            continue
        confidence = r.get("confidence") or CONF_SOMEWHAT
        attempt_w = w * CONFIDENCE_ATTEMPT_WEIGHT.get(confidence, 1.0)
        weighted_attempts[qid] += attempt_w
        if r.get("selected") == correct_answer:
            weighted_correct[qid] += attempt_w * CONFIDENCE_SUCCESS_CREDIT.get(confidence, 1.0)

    scores = {}
    for qid, attempts in weighted_attempts.items():
        smoothed_mastery = (
            weighted_correct[qid] + MASTERY_PRIOR * MASTERY_PRIOR_STRENGTH
        ) / (attempts + MASTERY_PRIOR_STRENGTH)
        scores[qid] = 1.0 - smoothed_mastery
    return scores


def theme_weight(theme, weakness_scores):
    if theme not in weakness_scores:
        return WEAKNESS_BASELINE
    return WEAKNESS_FLOOR + weakness_scores[theme] * WEAKNESS_SPAN


def compute_question_stats(responses, now=None):
    """Cohort-wide, recency-weighted difficulty per question_id, aggregated
    across every user's response history -- not just one learner. A single
    person's handful of attempts on one question is too sparse to say much
    about that question's intrinsic difficulty; the whole study group's
    combined history says more.

    Unlike compute_theme_weakness(), this deliberately does NOT weight by
    confidence -- confidence is a subjective, per-person signal, and mixing
    several different people's sense of their own confidence into a single
    question's "objective" difficulty adds noise without a clear benefit.
    This is plain recency-weighted, Bayesian-smoothed accuracy.

    Returns {question_id: {"attempts": float, "success_rate": float,
    "difficulty": float}}, difficulty = 1 - success_rate, smoothed toward
    QUESTION_DIFFICULTY_PRIOR the same way theme mastery is."""
    weighted_correct = defaultdict(float)
    weighted_attempts = defaultdict(float)
    for r in responses:
        qid = r.get("question_id")
        if not qid:
            continue
        w = _recency_weight(r.get("timestamp"), now)
        if r.get("selected") == OPT_SKIPPED:
            weighted_attempts[qid] += w
            continue
        correct_answer = r.get("correct_answer")
        if not correct_answer:
            continue
        weighted_attempts[qid] += w
        if r.get("selected") == correct_answer:
            weighted_correct[qid] += w

    stats = {}
    for qid, attempts in weighted_attempts.items():
        smoothed_success = (
            weighted_correct[qid] + QUESTION_DIFFICULTY_PRIOR * QUESTION_DIFFICULTY_PRIOR_STRENGTH
        ) / (attempts + QUESTION_DIFFICULTY_PRIOR_STRENGTH)
        stats[qid] = {
            "attempts": attempts,
            "success_rate": smoothed_success,
            "difficulty": 1.0 - smoothed_success,
        }
    return stats


def question_new_item_factor(q, question_stats, theme_weakness):
    """A gentle nudge, not a hard rule, applied only when selecting among
    NEW (never-attempted-by-this-user) questions: in a theme you're weak
    in, mildly favour easier questions before harder ones in that same
    theme, so brand-new material doesn't immediately compound an existing
    weak spot. In a theme you're not weak in, difficulty barely matters --
    the factor stays close to 1.0 regardless of how hard the question is.

    theme_weakness is compute_theme_weakness()'s 0..1 score for this
    question's theme (0.0 if the theme isn't in that dict, i.e. never
    attempted at all -- nothing to be gentle about yet). Questions with no
    cohort attempt history (can't estimate difficulty) get a neutral 1.0,
    same as if this feature didn't exist."""
    stats = question_stats.get(q["question_id"])
    if not stats:
        return 1.0
    difficulty = stats["difficulty"]  # 0.0 (easy) .. 1.0 (hard)
    return 1.0 + theme_weakness * (0.5 - difficulty)


def theme_coverage_weights(filtered_pool, attempted_ids):
    """For each theme present in filtered_pool, how unexplored is it *within
    this filtered pool* -- 1.0 means nothing in the theme has been attempted
    yet, near 0 means it's nearly fully covered. Used only for selecting
    among NEW (never-attempted) questions, so that whichever themes happen
    to have the most matching questions don't dominate new-question
    selection by sheer numbers -- a 40-question theme and a 10-question
    theme should both get worked through, not just whichever is bigger.
    Scoped to the current filtered pool (not the whole bank), so it reflects
    coverage of what's actually selectable under today's filters."""
    theme_totals = defaultdict(int)
    theme_attempted = defaultdict(int)
    for q in filtered_pool:
        theme_totals[q["theme"]] += 1
        if q["question_id"] in attempted_ids:
            theme_attempted[q["theme"]] += 1
    weights = {}
    for theme, total in theme_totals.items():
        coverage_ratio = theme_attempted[theme] / total if total else 0.0
        weights[theme] = max(1.0 - coverage_ratio, NEW_ITEM_COVERAGE_FLOOR)
    return weights


def weighted_sample(pool, weights, k):
    pool = list(pool)
    weights = list(weights)
    result = []
    for _ in range(min(k, len(pool))):
        total = sum(weights)
        r = random.uniform(0, total)
        upto = 0.0
        picked = False
        for i, w in enumerate(weights):
            upto += w
            if upto >= r:
                result.append(pool.pop(i))
                weights.pop(i)
                picked = True
                break
        if not picked:
            result.append(pool.pop())
            weights.pop()
    return result


# ---------- Main ----------

def main():
    if "_restore_attempted" not in st.session_state:
        github_restore_if_needed()
        st.session_state._restore_attempted = True

    user = get_current_user()

    # Repairs any response left without a matching schedule entry by a past
    # crash between the two independent writes in render_question() -- see
    # reconcile_schedule_from_responses(). Gated per-user (not just once per
    # session) so switching users via the sidebar button below still
    # reconciles the newly-selected profile.
    if st.session_state.get("_schedule_reconciled_for") != user:
        reconcile_schedule_from_responses(user)
        st.session_state._schedule_reconciled_for = user

    with st.sidebar:
        st.write(f"Practicing as **{user}**")
        if st.button("Switch user"):
            del st.session_state.current_user
            if "user" in st.query_params:
                del st.query_params["user"]
            st.rerun()

    render_centered_header("UPSC / HCS Prelims GS Paper I practice.")

    # Centers every expander's header text app-wide. Streamlit's expander
    # header isn't a native <summary> element -- it's a div[role="button"]
    # wrapping a <p> (the same stMarkdownContainer pattern used for a
    # widget's own label elsewhere), so text-align on that <p> is the
    # correct target; a flexbox rule on a plain <summary> selector (the
    # first attempt at this) matched nothing and had no effect. width:100%
    # makes sure the <p> actually spans the row before text-align has
    # anything to center within. Applied once here rather than scoped per
    # call site, by request, so it covers every expander in the app (Fix a
    # typo, Developer Mode, Explanation, Community notes, Review what you
    # missed, etc.) without repeating this block at each of them.
    #
    # Two earlier guesses at Streamlit's internal expander markup (a bare
    # `summary { justify-content: center }`, then a `div[role="button"] p`
    # selector copied from an older Streamlit version) both had zero
    # visible effect -- neither is documented/stable, and this app's
    # deployed version apparently matches neither guess. This version
    # forces `summary` into a flex container explicitly (rather than
    # assuming it already is one -- if it wasn't, justify-content alone
    # does nothing, which is the most likely reason attempt #1 was a
    # no-op) and backs it up with !important text-align on every plausible
    # inner text wrapper, so at least one rule should land regardless of
    # which exact element carries the label text.
    #
    # DevTools inspection (thanks, Sunny) showed why attempts #1-#3 still
    # had no visible effect: the header isn't one flex row, it's THREE
    # nested flex containers stacked inside <summary> (summary -> a
    # wrapping span -> the div actually holding the label). The outer two
    # are both width:100%/flex-grow:1 -- already filling all available
    # space with zero slack left over for justify-content to redistribute,
    # which is exactly why centering `summary` alone did nothing. Rather
    # than guess which of the three levels is the one with real leftover
    # space (almost certainly the innermost one), this forces
    # justify-content on ALL of them via a blanket `summary *` selector --
    # a no-op on any level that isn't a flex container or has no slack,
    # and the fix wherever it actually does.
    st.markdown(
        """
        <style>
        [data-testid="stExpander"] summary,
        [data-testid="stExpander"] summary * {
            justify-content: center !important;
        }
        [data-testid="stExpander"] summary p,
        [data-testid="stExpander"] summary span,
        [data-testid="stExpander"] summary div {
            text-align: center !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    questions, question_conflicts, skipped_records, source_file_by_id = load_questions()
    if not questions:
        st.error("No question bank found in /data. Add a question-bank JSON file first.")
        return

    if _is_admin(user):
        _render_developer_mode(questions, question_conflicts, skipped_records)

    open_flag_count = len(get_open_flags(load_notes(), load_resolved()))
    qbank_label = f"📚 Question Bank ({open_flag_count} 🚩)" if open_flag_count else "📚 Question Bank"
    my_bookmark_count = len(load_bookmarks().get(user, {}))
    bookmark_label = f"⭐ Bookmarks ({my_bookmark_count})" if my_bookmark_count else "⭐ Bookmarks"

    if "active_tab" not in st.session_state:
        st.session_state.active_tab = "practice"

    tab_labels = {
        "practice": "🎯 Smart Quiz",
        "report": "📊 Session Report",
        "bookmarks": bookmark_label,
        "review": qbank_label,
    }

    # ---- Tab-like navigation: a 2x2 grid of buttons, not st.tabs/st.radio ----
    # st.tabs can't be switched from code at all (no index/state argument).
    # st.radio(horizontal=True) *can* be switched from code, but on a narrow
    # (mobile) screen it wraps options onto a second line based on each
    # label's own text width -- so the item that wraps to row 2, column 2
    # doesn't line up under row 1, column 2 above it. Two explicit
    # st.columns(2) rows force both rows to share identical column widths,
    # so column 2 (Session Report / Question Bank) stays aligned regardless
    # of label length or count digits.
    # Buttons also sidestep a real Streamlit restriction the old st.radio
    # version hit: you cannot write to a widget's own session_state key
    # after that widget has already been instantiated in the same run
    # (raises StreamlitAPIException) -- which broke "jump to Practice"
    # from inside render_bookmarks(), called later in the same run. Plain
    # buttons have their own keys, so active_tab itself is just a normal
    # session_state variable now and can be set directly from anywhere.
    row1 = st.columns(2)
    row2 = st.columns(2)
    # Functional color per tab -- a color hint for what each box is for
    # (Smart Quiz=green/"go", Session Report=blue/data, Bookmarks=amber to
    # match the star, Question Bank=red to match the flag and its "needs
    # attention" nature) -- layered on TOP of the existing solid-when-active
    # / light-when-inactive distinction, so which tab is currently selected
    # still reads clearly at a glance rather than getting lost once every
    # box has its own permanent color. Uses the same scoped `st-key-*`
    # container-CSS pattern already proven for the "Start practice with my
    # bookmarks" button in render_bookmarks() (targeting a keyed
    # st.container's class, not the button's own `key=` or a global button
    # selector) -- !important
    # is needed here (unlike that simpler example) to outrank Streamlit's
    # own more specific built-in rules for the primary/secondary button
    # variants themselves.
    tab_colors = {
        "practice":  {"solid": "#16a34a", "solid_hover": "#15803d", "solid_press": "#166534",
                      "tint": "#f0fdf4", "tint_border": "#86efac", "tint_text": "#15803d", "tint_hover": "#dcfce7"},
        "report":    {"solid": "#2563eb", "solid_hover": "#1d4ed8", "solid_press": "#1e40af",
                      "tint": "#eff6ff", "tint_border": "#93c5fd", "tint_text": "#1d4ed8", "tint_hover": "#dbeafe"},
        "bookmarks": {"solid": "#d97706", "solid_hover": "#b45309", "solid_press": "#92400e",
                      "tint": "#fffbeb", "tint_border": "#fcd34d", "tint_text": "#b45309", "tint_hover": "#fef3c7"},
        "review":    {"solid": "#dc2626", "solid_hover": "#b91c1c", "solid_press": "#991b1b",
                      "tint": "#fef2f2", "tint_border": "#fca5a5", "tint_text": "#b91c1c", "tint_hover": "#fee2e2"},
    }
    for col, tab_id in [(row1[0], "practice"), (row1[1], "report"),
                         (row2[0], "bookmarks"), (row2[1], "review")]:
        with col:
            is_active = st.session_state.active_tab == tab_id
            c = tab_colors[tab_id]
            box_key = f"navbox_{tab_id}"
            # Previously Smart Quiz was the one exception that stayed in its
            # lighter tint even while active, since it's the tab you're on
            # for nearly the whole time you're using the app -- going solid
            # meant a heavy block of solid green across the top of the
            # screen for the entire duration of every quiz. Removed by
            # request: the inconsistent feedback (three tabs visibly show
            # "you are here", one doesn't) was the bigger problem in
            # practice. If the solid green ever feels too heavy during long
            # sessions, a lighter middle ground (e.g. a colored underline
            # instead of a full fill) is a quick follow-up rather than a
            # full revert.
            show_solid = is_active
            if show_solid:
                inject_button_tint_css(
                    box_key, c['solid'], c['solid'], 'white',
                    hover_bg=c['solid_hover'], hover_border=c['solid_hover'], hover_text='white',
                    press_bg=c['solid_press'], press_border=c['solid_press'], press_text='white',
                )
            else:
                inject_button_tint_css(
                    box_key, c['tint'], c['tint_border'], c['tint_text'],
                    hover_bg=c['tint_hover'],
                )
            with st.container(key=box_key):
                if st.button(
                    tab_labels[tab_id],
                    key=f"navbtn_{tab_id}",
                    type="primary" if show_solid else "secondary",
                    use_container_width=True,
                ):
                    st.session_state.active_tab = tab_id
                    st.rerun()

    active_tab = st.session_state.active_tab

    # A one-shot notice queued by another section (e.g. bookmarks) before it
    # switched tabs and reran -- shown once here, then cleared.
    if st.session_state.get("pending_toast"):
        st.toast(st.session_state.pop("pending_toast"))

    if active_tab == "practice":
        render_practice(questions, user)
    elif active_tab == "report":
        render_report(questions, user, source_file_by_id)
    elif active_tab == "bookmarks":
        render_bookmarks(questions, user, source_file_by_id)
    elif active_tab == "review":
        render_question_bank(questions, user, source_file_by_id)


def _render_developer_mode(questions, conflicts=None, skipped_records=None):
    """Admin-only shell: consolidates data-source visibility in one place.
    Legal Mapping stays a placeholder until section-level Act-mapping data
    actually exists in the schema — showing zeros or invented coverage
    numbers here would violate the fabrication rule."""
    conflicts = conflicts or []
    skipped_records = skipped_records or []
    with st.expander("🛠️ Developer Mode", expanded=False):
        st.subheader("Data source")
        files = sorted(f.name for f in DATA_DIR.glob("*.json"))
        st.write(f"Files found in /data: {', '.join(files) if files else 'none'}")
        excluded_count = sum(1 for q in questions if not is_valid_for_practice(q))
        st.write(f"Total questions loaded (after de-duplication): {len(questions)}")
        st.write(f"Excluded from practice (no usable/undisputed answer key): {excluded_count} / {len(questions)}")
        with_exp = sum(1 for q in questions if q.get("explanation"))
        st.write(f"Questions with an explanation: {with_exp} / {len(questions)}")
        with_section = sum(1 for q in questions if q.get("section_tags"))
        st.write(f"Questions with a section tag: {with_section} / {len(questions)}")
        st.write(f"Backup storage configured: {'yes' if _github_config() else 'no'}")
        registered = sorted(set(
            list(load_users().keys())
            + [r.get("user") for r in load_responses() if r.get("user")]
        ))
        st.write(f"Registered profiles: {len(registered)} / {MAX_USERS}")

        st.divider()
        st.subheader("Answer-key conflicts")
        st.caption(
            "Two source files disagreeing on a question_id's answer — never "
            "auto-resolved; excluded from scored practice until fixed in /data."
        )
        if not conflicts:
            st.caption("None detected.")
        else:
            st.warning(f"{len(conflicts)} question(s) have conflicting answers between source files.")
            for c in conflicts:
                st.markdown(f"**{c['question_id']}**")
                for s in c["sources"]:
                    exp_note = " (has explanation)" if s["has_explanation"] else ""
                    st.caption(f"- `{s['file']}`: answer = {s['answer']!r}{exp_note}")

        st.divider()
        st.subheader("Skipped / malformed records")
        if not skipped_records:
            st.caption("None — every record in /data parsed and loaded cleanly.")
        else:
            st.warning(f"{len(skipped_records)} record(s) excluded while loading.")
            for s in skipped_records:
                st.caption(f"- `{s['file']}`: {s['error']}")

        st.divider()
        st.subheader("Legal Mapping")
        st.caption(
            "Not available yet. This will show ITA 1961 ↔ 2025 Act coverage, "
            "unmapped sections, and verified mappings once section-level "
            "Act-reference data exists for the question bank — right now only "
            "chapter-level `syllabus_correlation` exists, and only for ITI "
            "2022-2025. Showing numbers here today would mean either all-zero "
            "stats or fabricating coverage that doesn't exist."
        )

        st.divider()
        st.subheader("Recent typo fixes")
        edit_log = sorted(load_edit_log(), key=lambda e: e.get("timestamp", ""), reverse=True)
        if not edit_log:
            st.caption("No in-app edits recorded yet.")
        else:
            for entry in edit_log[:10]:
                st.markdown(
                    f"**{entry.get('question_id')}** · {entry.get('field')} · "
                    f"{entry.get('user')} · {entry.get('timestamp', '')[:16].replace('T', ' ')}"
                )
                with st.expander("Before / after"):
                    st.caption("Before:")
                    st.write(entry.get("old_value", ""))
                    st.caption("After:")
                    st.write(entry.get("new_value", ""))
            if len(edit_log) > 10:
                st.caption(f"...and {len(edit_log) - 10} earlier edit(s) in the full log.")

        st.divider()
        st.subheader("🔍 Why this question? (session diagnostics)")
        st.caption(
            "The priority-score components split_pool_by_schedule()/review_priority_bonus() "
            "actually use to rank a chosen user's currently-overdue questions -- read-only, "
            "doesn't affect any real session."
        )
        if not registered:
            st.caption("No registered profiles yet.")
        else:
            diag_user = st.selectbox("User", registered, key="diag_user_select")
            practiceable_diag = [q for q in questions if is_valid_for_practice(q)]
            if not practiceable_diag:
                st.caption("No practiceable questions in the bank.")
            else:
                diag_responses = load_responses()
                diag_schedule = load_schedule()
                diag_wrong_streaks = wrong_streak_counts(diag_responses, diag_user)
                diag_question_weakness = compute_question_weakness(diag_responses, diag_user)
                diag_theme_weakness = compute_theme_weakness(diag_responses, diag_user)
                _new, diag_overdue, _not_due = split_pool_by_schedule(
                    diag_schedule, diag_user, practiceable_diag, diag_responses,
                    diag_wrong_streaks, diag_question_weakness,
                )
                if not diag_overdue:
                    st.caption(f"{diag_user} has no overdue questions right now.")
                else:
                    now_diag = datetime.now()
                    for q in diag_overdue[:15]:
                        qid = q["question_id"]
                        entry = diag_schedule.get(schedule_key(diag_user, qid))
                        overdue_hours = (
                            (now_diag - datetime.fromisoformat(entry["next_due"])).total_seconds() / 3600
                            if entry else 0.0
                        )
                        streak = diag_wrong_streaks.get(qid, 0)
                        # Decompose review_priority_bonus()'s combined bonus into its two
                        # components by calling it once with and once without the
                        # weakness term, rather than re-deriving the formula here --
                        # keeps this diagnostic from silently drifting out of sync with
                        # the real scoring function.
                        combined_bonus = review_priority_bonus(qid, diag_wrong_streaks, diag_question_weakness)
                        streak_bonus = review_priority_bonus(qid, diag_wrong_streaks)
                        weakness_bonus = combined_bonus - streak_bonus
                        final_score = overdue_hours + combined_bonus
                        with st.expander(f"{qid} — final priority {final_score:.1f}"):
                            st.write("Pool: Overdue")
                            st.write(f"Theme: {q.get('theme', '—')}")
                            st.write(f"Overdue by: {overdue_hours:.1f}h")
                            st.write(f"Wrong streak: {streak}  (+{streak_bonus:.0f}h-equiv)")
                            st.write(
                                f"Personal question weakness: {diag_question_weakness.get(qid, 0.0) * 100:.0f}%  "
                                f"(+{weakness_bonus:.0f}h-equiv)"
                            )
                            st.write(f"Theme weakness: {diag_theme_weakness.get(q.get('theme'), 0.0) * 100:.0f}%")
                            st.write(f"**Final priority score: {final_score:.1f}**")
                    if len(diag_overdue) > 15:
                        st.caption(f"...and {len(diag_overdue) - 15} more overdue question(s) not shown.")


# ---------- Practice / Smart Quiz tab ----------

def start_session(session_questions, user):
    st.session_state.session_questions = session_questions
    st.session_state.current_idx = 0
    st.session_state.session_id = datetime.now().isoformat()
    st.session_state.revealed = False
    st.session_state.selected_letter = None
    st.session_state.session_results = []
    st.session_state.nav_idx = 0
    save_active_session(
        user, session_questions, 0, [], st.session_state.session_id,
        revealed=False, selected_letter=None,
    )


def dynamic_review_share(responses, user):
    """How much of an adaptive session's guaranteed review quota (see
    REVIEW_SLOT_SHARE / build_session_from_filters) goes to due re-reviews,
    scaled DIRECTLY to this user's most recent quiz score -- a 7/10 (70%)
    last session means the next one reserves about 70% of its slots for
    the wrongly-answered/due-review pool; a 3/10 (30%) last session means
    about 30%. This is a deliberate design choice, not a smoothed or
    inverted one: the share moves exactly with the raw score, with no
    floor/ceiling beyond the natural 0..1 range a ratio already has, and no
    damping across sessions -- each session's share reflects only the one
    immediately before it. Falls back to the static REVIEW_SLOT_SHARE
    default when there's no completed-session history yet to scale from
    (a brand-new profile's very first session), via
    last_session_accuracy()."""
    accuracy = last_session_accuracy(responses, user)
    if accuracy is None:
        return REVIEW_SLOT_SHARE
    return accuracy


def build_session_from_filters(practice_questions, user, params):
    """Applies a stored filter-params dict (see render_practice's 'Start /
    Restart Session' button) to build a freshly-sampled session list, without
    showing the setup screen. Used both by that button and by 'Start a new
    session' at the end of a session, so the two always sample the same way.
    Responses/schedule/weakness are reloaded fresh each call (not passed in)
    since they can change between sessions -- most notably, the review just
    finished updates due-dates and weakness scores that should feed the next
    sample. Returns [] if nothing currently matches the stored filters.

    When params["adaptive"] is True, a session fills in this order: (1) a
    GUARANTEED quota of due re-reviews, ranked by a blended priority score
    (see dynamic_review_share() / review_priority_bonus) -- a share of the
    session that scales directly with the learner's most recent quiz score,
    capped by however many are actually due; (2) not-yet-attempted material
    fills the rest, weighted toward weak/under-explored themes; (3) if
    not-yet-attempted material runs out before the session is full, any due
    re-reviews beyond the guaranteed quota backfill the remainder; (4)
    not-yet-due material, as a last-resort fallback if both of the above are
    exhausted. The guaranteed quota exists because a strict either-first
    order starves whichever pool is smaller: due-first swamped new material
    whenever the overdue backlog was large, new-first later swamped due-
    reviews just as completely once the not-attempted pool got large -- a
    guaranteed share avoids both failure modes at once.

    Bookmarking a question has NO effect on session composition at all --
    not on eligibility, not on priority within any pool. It used to carve
    bookmarked questions out of the not-yet-attempted and not-yet-due pools
    (on the theory that a bookmark alone shouldn't justify pulling in extra
    material, since the Bookmarks tab already has its own dedicated,
    unfiltered "Start practice with my bookmarks" session for that) and add
    a small priority bump to the review-quota ranking. Both were removed by
    request: a bookmarked question is now treated exactly like any other
    question everywhere in this function, and starring something is purely
    a personal flag with no bearing on what a freshly built quiz pulls in.
    render_practice's setup screen no longer applies any bookmark carve-out
    either, so its preview counts and this function's actual build still
    agree."""
    filtered = [
        q for q in practice_questions
        if q["exam"] in params["selected_exams"] and q["year"] in params["selected_years"]
    ]
    if params["selected_themes"]:
        filtered = [q for q in filtered if q["theme"] in params["selected_themes"]]

    if not filtered:
        return []

    k = min(params["num_questions"], len(filtered))

    if params["adaptive"]:
        responses = load_responses()
        schedule = load_schedule()
        weakness_scores = compute_theme_weakness(responses, user)
        wrong_streaks = wrong_streak_counts(responses, user)
        question_weakness = compute_question_weakness(responses, user)
        new_items, overdue, not_due = split_pool_by_schedule(
            schedule, user, filtered, responses, wrong_streaks, question_weakness
        )

        # Guaranteed review quota: due re-reviews (already sorted by
        # split_pool_by_schedule's blended priority score, highest first)
        # claim a share of the session that scales directly with how the
        # learner's last quiz went (see dynamic_review_share()) -- no matter
        # how much not-yet-attempted material is sitting in the pool. This
        # is what actually answers "prioritize wrongly-answered questions
        # alongside not-attempted ones": a share of every session goes to
        # review regardless of which pool happens to be bigger today, and
        # that share grows or shrinks with recent performance rather than
        # sitting fixed.
        review_share = dynamic_review_share(responses, user)
        review_quota = min(round(k * review_share), len(overdue))
        session = overdue[:review_quota]
        remaining_needed = k - len(session)

        if remaining_needed > 0:
            # Not-yet-attempted material fills the rest, weighted by the
            # same weakness score reviews use (so a theme you're already
            # struggling in gets introduced faster), combined with how
            # little of that theme's pool you've touched at all (so a theme
            # with more matching questions doesn't dominate purely by
            # outnumbering everything else), and with question-level
            # difficulty (so a weak theme's easier questions tend to come up
            # before its harder ones, rather than sampling the theme
            # uniformly). This is deliberately still theme-based (not
            # personal-weakness-based like the review quota above) -- a
            # never-attempted question has no personal history yet for
            # compute_question_weakness() to measure, so theme weakness is
            # the only signal available here. Multiplying weakness and
            # coverage means a theme that's both weak AND under-explored
            # gets pulled in fastest, while a theme you've already covered
            # heavily gets naturally throttled even if it once scored weak.
            attempted_ids = {r["question_id"] for r in responses if r.get("user") == user}
            coverage_weights = theme_coverage_weights(filtered, attempted_ids)
            question_stats = compute_question_stats(responses)
            new_weights = [
                theme_weight(q["theme"], weakness_scores)
                * coverage_weights.get(q["theme"], 1.0)
                * question_new_item_factor(q, question_stats, weakness_scores.get(q["theme"], 0.0))
                for q in new_items
            ]
            session.extend(weighted_sample(new_items, new_weights, remaining_needed))
            remaining_needed = k - len(session)

        if remaining_needed > 0:
            # Not enough not-yet-attempted material to fill the rest -- any
            # due re-reviews beyond the guaranteed quota backfill next,
            # still highest-priority-first.
            leftover_overdue = overdue[review_quota:]
            session.extend(leftover_overdue[:remaining_needed])
            remaining_needed = k - len(session)

        if remaining_needed > 0:
            chosen_ids = {q["question_id"] for q in session}
            rest_pool = [q for q in not_due if q["question_id"] not in chosen_ids]
            # Same bonus concept as the overdue sort, converted into a smooth
            # weight multiplier (24 hours-equivalent == +1.0x) rather than a
            # sort score, since this pool is drawn from probabilistically.
            # This pool has already been attempted at least once, so
            # review_priority_bonus's personal question_weakness term
            # applies -- a specific question you've personally struggled
            # with gets pulled forward here even if its theme overall is one
            # you're otherwise fine in.
            weights = [
                theme_weight(q["theme"], weakness_scores)
                * (1.0 + review_priority_bonus(q["question_id"], wrong_streaks, question_weakness) / 24.0)
                for q in rest_pool
            ]
            session.extend(weighted_sample(rest_pool, weights, remaining_needed))
        random.shuffle(session)
    else:
        # Plain random mode: no bookmark carve-out here either -- every
        # question matching the filters is equally eligible.
        session = random.sample(filtered, min(k, len(filtered)))

    return session


def render_practice(questions, user):
    valid_questions = [q for q in questions if is_valid_for_practice(q)]
    excluded_count = len(questions) - len(valid_questions)

    my_flagged_ids = questions_flagged_by(user)
    practice_questions = (
        [q for q in valid_questions if q["question_id"] not in my_flagged_ids]
        if my_flagged_ids else valid_questions
    )
    my_flag_excluded_count = len(valid_questions) - len(practice_questions)

    # ---- Active session: skip straight to the question, no setup clutter ----
    if st.session_state.get("session_questions"):
        render_question(practice_questions, user)
        return

    # ---- Resume an in-progress session from before a refresh or a dropped
    # connection -- real risks on a mobile home-screen shortcut. Only reached
    # when nothing is already live in this run's session_state above.
    persisted = load_active_sessions().get(user)
    if persisted and persisted.get("question_ids"):
        questions_by_id = {q["question_id"]: q for q in practice_questions}
        ids = persisted["question_ids"]
        if all(qid in questions_by_id for qid in ids):
            resumed_qs = [questions_by_id[qid] for qid in ids]
            total = len(resumed_qs)
            done = min(persisted.get("current_idx", 0), total)
            if done < total:
                started = persisted.get("started_at", "")[:16].replace("T", " ")
                # If the question at `done` was already answered but not yet
                # advanced past (revealed=True -- see save_active_session),
                # it's genuinely finished even though current_idx hasn't
                # moved on to the next one yet; reflect that in the count.
                completed_count = done + (1 if persisted.get("revealed") else 0)
                st.info(
                    f"You have an unfinished session: {completed_count}/{total} completed"
                    + (f", started {started}" if started else "") + "."
                )
                # Same light-tint density as everywhere else in the app --
                # green for the "go/continue" action, red for the
                # "discards your progress" one, matching that same
                # green=go / red=warning language used throughout.
                inject_button_tint_css("resume_session_box", "#f0fdf4", "#86efac", "#15803d", hover_bg="#dcfce7")
                inject_button_tint_css("discard_session_box", "#fef2f2", "#fca5a5", "#b91c1c", hover_bg="#fee2e2")
                col_resume, col_discard = st.columns(2)
                with col_resume:
                    with st.container(key="resume_session_box"):
                        resume_clicked = st.button(
                            "▶️ Resume session", use_container_width=True,
                        )
                if resume_clicked:
                    st.session_state.session_questions = resumed_qs
                    st.session_state.current_idx = done
                    st.session_state.session_id = persisted.get("session_id") or datetime.now().isoformat()
                    # Restore exactly the revealed/selected state that was
                    # persisted at the last submit -- this used to always
                    # reset to unrevealed, which meant a disconnect right
                    # after submitting an answer (before tapping "Next
                    # question") came back as a blank, resubmittable
                    # question and could double-count that response.
                    st.session_state.revealed = persisted.get("revealed", False)
                    st.session_state.selected_letter = persisted.get("selected_letter")
                    st.session_state.session_results = persisted.get("session_results", [])
                    st.session_state.nav_idx = done
                    st.rerun()
                with col_discard:
                    with st.container(key="discard_session_box"):
                        discard_clicked = st.button(
                            "Discard and start fresh", use_container_width=True,
                        )
                if discard_clicked:
                    clear_active_session(user)
                    st.rerun()
                return
            # current_idx already reached the end (e.g. the app restarted
            # mid the completion screen, before it could clear this record)
            # -- nothing left to resume.
            clear_active_session(user)
        else:
            # The bank changed since this session was recorded (a question
            # removed, or newly excluded via is_valid_for_practice) -- the
            # saved position would no longer line up safely against the
            # current question list, so don't offer a resume that could
            # silently skip or misalign questions.
            clear_active_session(user)

    # ---- Setup screen (only shown when no session is active) ----

    exams = sorted(set(q["exam"] for q in practice_questions))
    years = sorted(set(q["year"] for q in practice_questions))
    themes = sorted(set(q["theme"] for q in practice_questions))

    # Sky blue tags for the Exams/Years multiselect pills, by request
    # (previously yellow) -- same bg/border/text "density" formula as every
    # other tint elsewhere in the app (very light bg, a few shades darker
    # border, a readable darker text), using Tailwind's sky-50/300/700 the
    # same way the yellow version used yellow-50/300/700. Theme's
    # multiselect below is deliberately NOT given a key/styled here, since
    # only Exams/Years were asked for.
    #
    # DevTools inspection (thanks again, Sunny) showed the first version of
    # this rule targeted the wrong markup entirely: this deployed
    # Streamlit's multiselect isn't built on BaseWeb anymore (no
    # [data-baseweb="tag"] anywhere) -- it's react-aria based, and each
    # pill is a <span data-tag ...> that sets background-color AND a white
    # text color directly on itself (the visible label text is a child
    # span that just inherits that color rather than setting its own).
    # Overriding color on [data-tag] itself, not a nested span, is what
    # actually reaches the text. Kept the old [data-baseweb="tag"] selector
    # alongside the new [data-tag] one too -- harmless if it matches
    # nothing, a safety net if some environment still renders the older
    # markup.
    st.markdown(
        """
        <style>
        [class*="st-key-exams_filter"] [data-baseweb="tag"],
        [class*="st-key-years_filter"] [data-baseweb="tag"],
        [class*="st-key-exams_filter"] [data-tag],
        [class*="st-key-years_filter"] [data-tag] {
            background-color: #f0f9ff !important;
            color: #0369a1 !important;
            border-color: #7dd3fc !important;
        }
        [class*="st-key-exams_filter"] [data-baseweb="tag"] span,
        [class*="st-key-years_filter"] [data-baseweb="tag"] span,
        [class*="st-key-exams_filter"] [data-tag] span,
        [class*="st-key-years_filter"] [data-tag] span {
            color: #0369a1 !important;
        }
        [class*="st-key-exams_filter"] [data-baseweb="tag"] svg,
        [class*="st-key-years_filter"] [data-baseweb="tag"] svg,
        [class*="st-key-exams_filter"] [data-tag] svg,
        [class*="st-key-years_filter"] [data-tag] svg {
            fill: #0369a1 !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    col1, col2 = st.columns([1, 2])
    with col1:
        with st.container(key="exams_filter"):
            selected_exams = st.multiselect("Exams", exams, default=exams)
    with col2:
        with st.container(key="years_filter"):
            selected_years = st.multiselect("Years", years, default=years)

    selected_themes = st.multiselect("Theme (leave empty for all themes)", themes, default=[])

    num_questions = st.slider("Number of questions this session", 10, 50, 10, step=10)

    responses = load_responses()
    schedule = load_schedule()
    weakness_scores = compute_theme_weakness(responses, user)
    question_weakness = compute_question_weakness(responses, user)
    adaptive = st.checkbox(
        "Prioritize weak spots and due reviews (adaptive)", value=True,
        help="Reserves a share of the session for due re-reviews, scaled directly to your "
             "last quiz score (a 70% score reserves about 70%; defaults to 40% until you've "
             "completed one) and ranked toward specific questions you've personally gotten "
             "wrong. Fills the rest with not-yet-attempted material (weighted toward themes "
             "you've done worse in), and lets either side backfill the other if it runs "
             "short. Turn off for plain random practice. (Bookmarking a question has no "
             "effect here -- star it and visit the Bookmarks tab to study it specifically.)",
    )

    if adaptive and weakness_scores:
        ranked = sorted(weakness_scores.items(), key=lambda x: -x[1])[:5]
        with st.expander("Currently prioritizing (weakest first)"):
            for theme, score in ranked:
                st.write(f"- {theme}  ({score * 100:.0f}% weak)")

    questions_by_id_preview = {q["question_id"]: q for q in practice_questions}

    if adaptive and question_weakness:
        ranked_questions = sorted(question_weakness.items(), key=lambda x: -x[1])[:5]
        with st.expander("Individually weak questions (weakest first)"):
            st.caption("Your own track record on these exact questions -- separate from theme.")
            for qid, score in ranked_questions:
                qobj = questions_by_id_preview.get(qid)
                stem = qobj["question"] if qobj else ""
                preview = (stem[:70] + "…") if len(stem) > 70 else stem
                st.write(f"- **{qid}** {preview}  ({score * 100:.0f}% weak)")

    if adaptive:
        subtopic_weakness = compute_subtopic_weakness(responses, questions_by_id_preview, user)
        if subtopic_weakness:
            ranked_subtopics = sorted(subtopic_weakness.items(), key=lambda x: -x[1])[:5]
            with st.expander("Weak subtopics (weakest first)"):
                st.caption(
                    "One level finer than theme -- e.g. overall TDS may look fine while a "
                    "specific subtopic within it (like 194J) keeps tripping you up. "
                    "Informational only for now -- doesn't yet affect which questions a "
                    "session picks."
                )
                for subtopic, score in ranked_subtopics:
                    st.write(f"- {subtopic}  ({score * 100:.0f}% weak)")

    filtered = [
        q for q in practice_questions
        if q["exam"] in selected_exams and q["year"] in selected_years
    ]
    if selected_themes:
        filtered = [q for q in filtered if q["theme"] in selected_themes]

    # Same light green tint as Submit/Next question/Smart Quiz elsewhere in
    # the app -- this is the setup screen's own "go" button (separate from
    # any of those; this one only exists before a session starts), so it
    # gets its own scoped st-key container-CSS block. Adds an explicit
    # :disabled style, unlike the other tinted buttons -- this is the one
    # button in the app that's ever actually disabled (when no questions
    # match the chosen filters), and the !important color overrides above
    # would otherwise mask Streamlit's own default disabled-dimming, making
    # a non-clickable button look identical to a clickable one.
    inject_button_tint_css(
        "startbox", "#f0fdf4", "#86efac", "#15803d", hover_bg="#dcfce7",
        disabled_bg="#f3f4f6", disabled_border="#d1d5db", disabled_text="#9ca3af",
    )
    with st.container(key="startbox"):
        if st.button("Start / Restart Session", disabled=not filtered, use_container_width=True):
            st.session_state.last_session_params = {
                "selected_exams": selected_exams,
                "selected_years": selected_years,
                "selected_themes": selected_themes,
                "num_questions": num_questions,
                "adaptive": adaptive,
            }
            session = build_session_from_filters(
                practice_questions, user, st.session_state.last_session_params
            )
            start_session(session, user)
            st.rerun()

    st.info("Set your filters and click **Start / Restart Session** to begin.")

    # ---- Supplementary details, tucked below the main call-to-action so the
    # filters and Start button aren't crowded by numbers most sessions don't
    # need to weigh. See build_session_from_filters() for the algorithm these
    # counts describe.
    if excluded_count:
        st.caption(
            f"ℹ️ {excluded_count} question(s) are temporarily excluded from practice "
            "sessions — missing, disputed, or under-review answer keys. See Question Bank for details."
        )
    if my_flag_excluded_count:
        st.caption(
            f"🚩 {my_flag_excluded_count} question(s) you've personally flagged are excluded "
            "from your own sessions until reviewed — other users may still see them."
        )
    if filtered:
        wrong_streaks = wrong_streak_counts(responses, user)
        new_items, overdue, not_due = split_pool_by_schedule(
            schedule, user, filtered, responses, wrong_streaks, question_weakness
        )
        # No bookmark carve-out here anymore -- this mirrors
        # build_session_from_filters() exactly, so the preview's counts
        # match what a session will actually pull.
        filtered_ids = {q["question_id"] for q in filtered}
        attempted_in_filter = len({
            r["question_id"] for r in responses
            if r.get("user") == user and r["question_id"] in filtered_ids
        })
        st.caption(
            f"{len(filtered)} questions match your filters — {len(new_items)} new, "
            f"{len(overdue)} due for re-review, {len(not_due)} scheduled for later. "
            f"You've attempted {attempted_in_filter} so far."
        )
        if adaptive:
            last_accuracy = last_session_accuracy(responses, user)
            if last_accuracy is None:
                st.caption(
                    f"📊 No completed quiz yet — this session reserves the default "
                    f"{round(REVIEW_SLOT_SHARE * 100)}% of slots for due re-reviews."
                )
            else:
                st.caption(
                    f"📊 Your last quiz: {round(last_accuracy * 100)}% — this session "
                    f"reserves about {round(last_accuracy * 100)}% of slots for due re-reviews."
                )
    else:
        st.caption("0 questions match your filters.")


def render_full_question(q, key_prefix, note=None, include_answer_in_copy=False, show_explanation=True, copy_for_ai=False):
    """Full reading-style rendering of one question: formatted stem, every
    option with the correct one highlighted (when a usable answer exists),
    explanation, syllabus correlation, and a copy-text expander -- the same
    presentation Reader mode uses. Shared so Bookmarks and Question Bank can
    offer the same read-through experience instead of a truncated preview.

    key_prefix must be unique per call SITE (not per question) -- e.g.
    "reader", "bookmark", "flagged", "missing_key", "disputed" -- since a
    single question_id can legitimately appear in more than one place at
    once (bookmarked AND flagged, say), and widget/container keys must stay
    unique across the whole page for that run.

    note, if given, is shown as a caution caption ABOVE the options -- for
    any context where the highlighted answer isn't a settled, undisputed
    one (flagged, disputed, or missing/conflicting), callers should pass
    something here rather than letting a green highlight imply more
    confidence than the data actually supports.

    include_answer_in_copy controls whether the plain-copy expander's text
    also includes an "Answer: ..." line -- see format_question_plaintext()'s
    docstring. Ignored when copy_for_ai=True (see below), since that path
    always includes the answer via format_question_for_ai_copy().

    copy_for_ai swaps the plain "📋 Copy question text" expander (st.code(),
    format_question_plaintext(), meant for pasting into WhatsApp/notes/etc.)
    for a single-click "📋 Copy question text for AI" button (render_copy_
    button(), format_question_for_ai_copy() -- adds the correct answer and
    a short instruction preamble asking for real Markdown tables and LaTeX
    math, see that function's docstring for why). Only the Bookmarks tab
    call site opts in; Question Bank' three call sites (flagged/missing_key/
    disputed) keep the plain version, since those questions' answers
    aren't reliably confirmed yet and format_question_for_ai_copy() would
    present an unconfirmed answer as settled fact to whatever AI tool it's
    pasted into.

    show_explanation defaults to True (Bookmarks tab keeps showing it --
    that's for personal study, where the explanation is exactly what's
    useful). Question Bank' three call sites (flagged/missing_key/disputed)
    pass False -- by request, that tab is a fast review/triage surface, not
    a study one, and the explanation was just adding scroll length there."""
    inject_option_justify_css()
    render_justified(
        format_question_stem(q["question"]),
        container_key=f"{key_prefix}_stem_{q['question_id']}",
    )
    if note:
        st.caption(note)

    correct_letter = q.get("answer")
    for o in q.get("options", []):
        if not o.get("text"):
            continue
        label = f"{o['label']}) {o['text']}"
        if o["label"] == correct_letter:
            st.success(label)
        else:
            st.write(label)

    if show_explanation:
        st.markdown("**Explanation**")
        explanation = q.get("explanation")
        if explanation:
            render_explanation_text(explanation)
        else:
            st.caption("No explanation available for this question yet.")

    correlation = q.get("syllabus_correlation")
    if correlation:
        st.caption(
            f"📖 Syllabus: {correlation['topic']} — Chapter {correlation['chapter_2025_act']} of the "
            f"Income-tax Act, 2025 (Chapter {correlation['chapter_1961_act']} of the 1961 Act)"
        )

    if copy_for_ai:
        render_copy_button(
            "📋 Copy question text for AI",
            format_question_for_ai_copy(q),
            key=f"copyai_{key_prefix}_{q['question_id']}",
        )
    else:
        with st.expander("📋 Copy question text"):
            st.code(format_question_plaintext(q, include_answer=include_answer_in_copy), language=None, wrap_lines=True)


def render_reviewed_question(q, user, show_caption=True):
    """Read-only view of a question already answered earlier in this same
    session, reached via the Prev/Next history nav at the top of
    render_question(), or from the post-session "Review what you missed"
    expander. Shows exactly what was selected, whether it was correct, and
    the explanation -- but has no radio/submit control of its own, so the
    original answer can't be changed or resubmitted here. Bookmarking and
    community notes/flagging are both still offered, since spotting a
    question worth saving or flagging is exactly as likely while reviewing
    as it is right after first answering it.

    The selected answer is looked up from responses.json by session_id +
    question_id rather than trusted from any in-memory list, since
    session_results (the in-memory per-session tally) only tracks
    theme/correct/skipped -- not which letter was picked -- and wouldn't
    survive a resume anyway.

    show_caption defaults to True (the history-nav call site, where a
    single question is on screen and the "you already answered this" framing
    is the only context the learner has). The post-session review list
    passes False -- that caption would otherwise repeat once per question in
    a list that's already titled "Review what you missed", adding scroll
    with no new information."""
    inject_option_justify_css()
    session_id = st.session_state.get("session_id")
    match = None
    for r in load_responses():
        if r.get("session_id") == session_id and r.get("question_id") == q["question_id"]:
            match = r  # a later entry would win if one ever somehow existed
    if show_caption:
        st.caption("📖 Reviewing a question you've already answered in this session — it can't be resubmitted here.")
    render_justified(
        format_question_stem(q["question"]),
        container_key=f"reviewstem_{q['question_id']}",
    )

    selected_letter = match.get("selected") if match else None
    correct_letter = q.get("answer")
    for o in q.get("options", []):
        if not o.get("text"):
            continue
        label = f"{o['label']}) {o['text']}"
        if o["label"] == correct_letter:
            st.success(label)
        elif o["label"] == selected_letter:
            st.error(label)
        else:
            st.write(label)

    if selected_letter == OPT_SKIPPED:
        st.info("You skipped this question.")
    elif match:
        st.caption("Correct." if selected_letter == correct_letter else "Not quite.")

    with st.expander("Explanation"):
        explanation = q.get("explanation")
        if explanation:
            render_explanation_text(explanation)
        else:
            st.caption("No explanation available for this question yet.")
        correlation = q.get("syllabus_correlation")
        if correlation:
            st.caption(
                f"📖 Syllabus: {correlation['topic']} — Chapter {correlation['chapter_2025_act']} of the "
                f"Income-tax Act, 2025 (Chapter {correlation['chapter_1961_act']} of the 1961 Act)"
            )

    my_bookmarks = load_bookmarks().get(user, {})
    is_bookmarked = q["question_id"] in my_bookmarks
    bookmark_now = st.checkbox(
        "⭐ Bookmark this Qn",
        value=is_bookmarked,
        key=f"bookmark_review_{q['question_id']}",
    )
    if bookmark_now != is_bookmarked:
        toggle_bookmark(user, q["question_id"])
        st.rerun()

    render_community_notes(q["question_id"], user)


def render_question(practice_questions, user):
    qs = st.session_state.session_questions
    idx = st.session_state.current_idx

    if idx >= len(qs):
        clear_active_session(user)
        results = st.session_state.get("session_results", [])
        # Centered by request -- st.success() left-aligns by default and has
        # no built-in alignment option, so this wraps it in its own keyed
        # container and targets stAlert's inner <p> the same way this app
        # already does elsewhere (e.g. the revealed-answer a/b/c/d boxes).
        st.markdown(
            """
            <style>
            [class*="st-key-session_complete_box"] [data-testid="stAlert"] p {
                text-align: center !important;
            }
            </style>
            """,
            unsafe_allow_html=True,
        )
        with st.container(key="session_complete_box"):
            st.success(f"Session complete — {len(qs)} questions attempted.")
        if results:
            correct_count = sum(1 for r in results if r["correct"])
            pct = round(100 * correct_count / len(results))

            # Score card + motivational quote, tiered by performance --
            # replaces the plain st.metric() by request. Same three-color
            # density used everywhere else in the app (green/amber/red),
            # picked by score rather than by function this time. Quotes are
            # picked at random per tier so repeat sessions in the same
            # range don't always show the identical line; all written fresh
            # for this app rather than sourced from anywhere, since quoting
            # someone else's words isn't something to do lightly.
            if pct >= 80:
                tier = "high"
            elif pct >= 50:
                tier = "mid"
            else:
                tier = "low"
            tier_colors = {
                "high": {"bg": "#f0fdf4", "border": "#86efac", "text": "#15803d"},
                "mid":  {"bg": "#fffbeb", "border": "#fcd34d", "text": "#b45309"},
                "low":  {"bg": "#fef2f2", "border": "#fca5a5", "text": "#b91c1c"},
            }
            tier_quotes = {
                "high": [
                    "Excellent work — you're exam-ready on this material.",
                    "Outstanding accuracy. This is the consistency that gets you through Paper I.",
                    "Sharp session — keep this pace going.",
                ],
                "mid": [
                    "Solid effort. A little more revision on the misses and this becomes a strong score.",
                    "You're on the right track — review what tripped you up and go again.",
                    "Good progress. Consistency will close the gap from here.",
                ],
                "low": [
                    "Tough one — but every wrong answer here is one you won't get wrong in the real exam.",
                    "This is exactly what practice is for. Review the explanations and come back stronger.",
                    "Don't be discouraged. Spotting the gaps now is how you close them before it counts.",
                ],
            }
            c = tier_colors[tier]
            quote = random.choice(tier_quotes[tier])
            # White background now (was tinted by tier before) -- keeps the
            # colored border and text as the only tier signal, by request,
            # rather than a full colored fill.
            st.markdown(
                f"""
                <style>
                [class*="st-key-score_summary_box"] {{
                    background-color: white !important;
                    border-color: {c['border']} !important;
                }}
                </style>
                """,
                unsafe_allow_html=True,
            )
            with st.container(key="score_summary_box", border=True):
                st.markdown(
                    f"""
                    <div style="text-align:center; padding:0.5rem 0;">
                        <div style="font-size:0.9rem; color:{c['text']}; opacity:0.85;">Score this session</div>
                        <div style="font-size:2.75rem; font-weight:700; color:{c['text']}; line-height:1.2;">
                            {correct_count}/{len(results)}
                        </div>
                        <div style="font-size:0.95rem; font-style:italic; color:{c['text']}; margin-top:0.4rem;">
                            {quote}
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            skipped_count = sum(1 for r in results if r["skipped"])
            attempted_count = len(results) - skipped_count
            if skipped_count:
                attempted_correct = sum(1 for r in results if r["correct"] and not r["skipped"])
                st.caption(
                    f"{skipped_count} skipped (counted as 0, same as the real exam). "
                    f"Of the {attempted_count} you actually answered, {attempted_correct} were correct."
                )

            # Marks-based score with real negative marking, alongside the
            # plain accuracy count above -- accuracy % alone hides how
            # costly wrong guesses are under UPSC/HCS's actual marking
            # scheme, which is the whole point of practicing attempt
            # strategy (skip vs. guess) rather than just recall. Only
            # questions whose (exam, paper) has a MARKING_CONFIG entry are
            # included, so a session mixing in un-configured data (e.g. a
            # future paper added without updating MARKING_CONFIG) degrades
            # gracefully instead of crashing or silently mis-scoring it.
            marked_results = [r for r in results if marking_for(r["exam"], r["paper"])]
            if marked_results:
                total_marks = 0.0
                max_marks = 0.0
                for r in marked_results:
                    scheme = marking_for(r["exam"], r["paper"])
                    max_marks += scheme["marks_correct"]
                    if r["skipped"]:
                        continue
                    total_marks += scheme["marks_correct"] if r["correct"] else scheme["marks_wrong"]
                schemes_used = sorted({marking_for(r["exam"], r["paper"])["label"] for r in marked_results})
                st.caption(
                    f"Estimated marks (with negative marking): {total_marks:.2f} / {max_marks:.2f} "
                    f"— {', '.join(schemes_used)}"
                )
        if _github_config():
            # Session-complete re-renders on every interaction on this screen
            # (any button click reruns the whole script), and github_backup()
            # iterates every backup file -- so without this guard, sitting on
            # this screen re-triggers a full GitHub sync on every rerun, not
            # just once at session end. Gate it on session_id (fresh per
            # start_session() call) so a new session still backs up once.
            this_session = st.session_state.get("session_id")
            if st.session_state.get("_backed_up_session_id") != this_session:
                backed_up, backup_msg = github_backup()
                st.session_state._backed_up_session_id = this_session
                st.session_state._last_backup_result = (backed_up, backup_msg)
            backed_up, backup_msg = st.session_state._last_backup_result
            st.caption("✅ Backed up." if backed_up else f"⚠️ Backup failed: {backup_msg}")
        # Same light green tint as Submit/Next question/Smart Quiz elsewhere.
        # negative margin-top pulls this row closer to whatever's above it
        # (the score box, or the backup caption) -- by request, the default
        # Streamlit spacing between stacked elements here left a bigger gap
        # than intended above this button.
        inject_button_tint_css("new_session_box", "#f0fdf4", "#86efac", "#15803d", hover_bg="#dcfce7")
        st.markdown(
            """
            <style>
            [class*="st-key-new_session_row"] {
                margin-top: -1.5rem !important;
            }
            </style>
            """,
            unsafe_allow_html=True,
        )
        # Centered below the score box now (was left-aligned before) --
        # narrow side columns push the middle column, and the button inside
        # it, toward the center of the row.
        with st.container(key="new_session_row"):
            _, col_new_session, _ = st.columns([1, 2, 1])
            with col_new_session:
                with st.container(key="new_session_box"):
                    start_new = st.button("Start a new session", use_container_width=True)
        if start_new:
            params = st.session_state.get("last_session_params")
            next_session = build_session_from_filters(practice_questions, user, params) if params else []
            if next_session:
                # Same filters, freshly sampled -- straight into the next quiz,
                # skipping the setup screen. Due-dates/weakness just shifted
                # from the session that ended, so this re-samples rather than
                # reusing the old list.
                start_session(next_session, user)
            else:
                # No stored filters (e.g. this was a bookmarks-practice session)
                # or nothing matches them anymore -- fall back to the setup screen.
                st.session_state.session_questions = []
            st.rerun()

        # ---- Review what you missed ----
        # Sits below everything else on this screen (score card, skip
        # caption, backup status, Start-a-new-session button) and folded
        # into one collapsed-by-default expander rather than shown inline --
        # a full read-through of every wrong/skipped question would
        # otherwise push Start-a-new-session an unpredictable distance down
        # the page after a rough session, so this keeps that button exactly
        # as reachable as it's always been, with the review one tap away.
        #
        # Wrong-answer and skipped are the same check here, by request: a
        # skip's `selected` is the OPT_SKIPPED sentinel, which by
        # construction never equals a real answer letter, so
        # `selected != correct_answer` already covers both without a
        # separate branch -- render_reviewed_question() below already
        # renders the "You skipped this question" case correctly.
        #
        # Looked up from responses.json by session_id, same source of truth
        # render_reviewed_question() below already uses -- session_results
        # (the in-memory tally) only tracks theme/correct/skipped, not which
        # question. Iterating `qs` (this session's own question order)
        # rather than responses.json's append order keeps the list numbered
        # the same way the Prev/Next history nav does during the session
        # itself.
        session_id = st.session_state.get("session_id")
        missed_ids = {
            r["question_id"] for r in load_responses()
            if r.get("session_id") == session_id
            and r.get("selected") != r.get("correct_answer")
        }
        missed_qs = [qq for qq in qs if qq["question_id"] in missed_ids]
        if missed_qs:
            with st.expander(f"📖 Review what you missed ({len(missed_qs)})"):
                for rq in missed_qs:
                    with st.container(border=True):
                        st.markdown(f"**{rq['question_id']}**  ·  _{rq['theme']}_  ·  {rq['exam']} {rq['year']}")
                        render_reviewed_question(rq, user, show_caption=False)
        return

    # ---- Prev/Next history nav ----
    # Lets the learner look back at questions already answered earlier in
    # this same session (and post a flag via community notes) with no way
    # to touch what was already submitted. `nav_idx` is which question is
    # actually on screen right now; `idx` (current_idx) always stays the
    # pointer to the live/current question. nav_idx is explicitly reset
    # back to idx wherever idx itself advances -- start_session(), the
    # "Next question ->" handler, the forced-skip handler below, and
    # session resume -- so the check here only covers the (normally
    # unreachable) case where it was never set at all. Both buttons use
    # stable keys tied only to idx (not to nav_idx) and are toggled via
    # `disabled=` rather than being conditionally rendered, since a
    # Streamlit widget needs a stable key across reruns for `disabled=`
    # changes to actually take effect.
    if "nav_idx" not in st.session_state:
        st.session_state.nav_idx = idx
    nav_idx = st.session_state.nav_idx

    if idx > 0:
        col_prev, col_next = st.columns(2)
        with col_prev:
            if st.button("⬅ Prev", disabled=(nav_idx == 0), key=f"histprev_{idx}", use_container_width=True):
                st.session_state.nav_idx = nav_idx - 1
                st.rerun()
        with col_next:
            if st.button("Next ➡", disabled=(nav_idx >= idx), key=f"histnext_{idx}", use_container_width=True):
                st.session_state.nav_idx = nav_idx + 1
                st.rerun()

    if nav_idx < idx:
        rq = qs[nav_idx]
        st.markdown(f"**Question {nav_idx + 1} of {len(qs)}**  ·  _{rq['theme']}_  ·  {rq['exam']} {rq['year']}")
        render_reviewed_question(rq, user)
        return

    q = qs[idx]
    st.progress((idx + 1) / len(qs))
    st.markdown(f"**Question {idx + 1} of {len(qs)}**  ·  _{q['theme']}_  ·  {q['exam']} {q['year']}")
    render_justified(format_question_stem(q["question"]), container_key=f"stem_box_{q.get('question_id', 'current')}")

    # Justify the answer options, revealed-state feedback, and explanation text
    # the same way the question stem is justified above. This targets
    # Streamlit's own rendered containers via CSS rather than replacing the
    # native widgets, so selection behaviour, accessibility, and theme colours
    # (e.g. dark mode) are all unaffected -- only text alignment/size changes.
    #
    # Also equalizes font size between the question stem and the options.
    # Streamlit renders st.radio()'s option labels in a smaller default font
    # than a plain st.markdown() paragraph (its own built-in widget-label
    # styling, not anything this app sets) -- so without this, the options
    # look visibly smaller than the question above them even though nothing
    # here asked for that. 1rem matches Streamlit's normal body/markdown
    # text size, which the question stem already renders at by default;
    # !important is needed to outrank Streamlit's own more specific built-in
    # rule for radio-option text. Bundled into the same selector as the
    # justify rule below since it already covers all three text surfaces
    # (radio options, revealed-state success/error alerts, and plain
    # markdown) that need to match the question's size.
    #
    # Also fixes radio-circle alignment on wrapped, multi-line options.
    # Verified against the real rendered HTML (browser-inspected on the live
    # app): each option is a <label data-testid="stRadioOption">, and the
    # actual [circle, text] pair sits two auto-generated flex-wrapper <div>s
    # deep inside that label, not directly under it. Streamlit's wrapper class
    # names (st-emotion-cache-XXXXX) are version-specific hashes that can
    # change on any release, so this anchors on the one stable, documented
    # test-id instead: data-testid="stMarkdownContainer" (Streamlit's own tag
    # for text/markdown output). A depth-based selector (matching today's
    # exact structure) and a :has()-based selector (which keeps working even
    # if Streamlit adds/removes a wrapper level later) are both included for
    # redundancy.
    #
    # Deployment note: an earlier attempt at this exact fix looked broken in
    # testing, but the cause turned out to be Streamlit Cloud not picking up
    # the new commit (a manual "Reboot app" was needed) -- not the CSS itself.
    # If a future CSS change here ever looks like it's "not working," rule out
    # a stale deployment (reboot the app) before assuming the selector is wrong.
    #
    # margin-top on the circle (below) compensates for a mismatch between two
    # different "tops": the circle's own compact box has no leading above it,
    # while the option text's first line sits inside normal line-height, which
    # adds roughly half a line's worth of empty space ("half-leading") above
    # the glyphs themselves -- align-items: flex-start alone lines up the two
    # BOXES' top edges, not the circle against the actual visible text.
    # Tuning history (each step based on a screenshot of the live app, not
    # something derivable exactly from the CSS alone): 0.15rem -> circle
    # visibly high -> 0.25rem -> still visibly high, though the gap looked
    # smaller -> 0.35rem (current). If 0.35rem still isn't enough, keep
    # raising it in ~0.1rem steps rather than reverting to the smaller
    # ~0.05rem steps originally suggested -- the first two steps together
    # only closed part of the gap, so the right value is probably still a
    # bit further out than 0.35rem lands. If it ever overshoots (circle
    # sitting visibly LOWER than the text instead), split the difference
    # with the last value that was still too high.
    st.markdown(
        """
        <style>
        div[data-testid="stRadio"] p,
        div[data-testid="stAlert"] p,
        div[data-testid="stMarkdownContainer"] p {
            text-align: justify;
            font-size: 1rem !important;
        }
        label[data-testid="stRadioOption"] > div > div,
        label[data-testid="stRadioOption"] div:has(> div[data-testid="stMarkdownContainer"]) {
            display: flex !important;
            align-items: flex-start !important;
        }
        label[data-testid="stRadioOption"] > div > div > *:first-child,
        label[data-testid="stRadioOption"] div:has(> div[data-testid="stMarkdownContainer"]) > *:first-child {
            flex: 0 0 auto !important;
            margin-top: 0.35rem;
        }
        label[data-testid="stRadioOption"] div[data-testid="stMarkdownContainer"] {
            flex: 1 1 0% !important;
            min-width: 0;
        }
        /* Extra breathing room BETWEEN options in the pre-submit radio list --
           Streamlit's own default spacing there is noticeably tighter than
           the revealed (post-submit) state, which uses separate st.success/
           st.write elements per option and already has generous gaps between
           them for free. Scoped to stRadioOption specifically, so this only
           touches the pre-submit list and leaves the revealed state alone.
           :not(:last-child) skips the trailing gap before the Confidence
           slider below, so that spacing isn't doubled up unnecessarily. */
        label[data-testid="stRadioOption"]:not(:last-child) {
            margin-bottom: 0.85rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    revealed = st.session_state.get("revealed", False)

    if not revealed:
        option_labels = [f"{o['label']}) {o['text']}" for o in q["options"] if o["text"]]
        if not option_labels:
            st.error("This question has no valid options in the data — skipping it.")
            if st.button("Skip to next question", key=f"forceskip_{idx}"):
                st.session_state.current_idx += 1
                st.session_state.nav_idx = st.session_state.current_idx
                st.session_state.revealed = False
                st.session_state.selected_letter = None
                save_active_session(
                    user, qs, st.session_state.current_idx,
                    st.session_state.session_results, st.session_state.session_id,
                    revealed=False, selected_letter=None,
                )
                st.rerun()
            return

        radio_key = f"radio_{q['question_id']}_{idx}"
        choice = st.radio("Select your answer:", option_labels, key=radio_key, index=None)

        confidence = st.select_slider(
            "Confidence",
            options=[CONF_GUESSED, CONF_SOMEWHAT, CONF_CONFIDENT],
            value=CONF_SOMEWHAT,
            key=f"conf_{q['question_id']}_{idx}",
        )

        col_submit, col_skip, col_restart = st.columns(3)

        # Submit/Skip/Restart colored the same way as the top-nav boxes
        # (green="go"/commit, amber="pause without fully committing", red=
        # "discards your current progress") -- same scoped st-key container-
        # CSS pattern as those, and now the same LIGHT tint values too (not
        # the bold solid fill used at first) -- switched by request, since
        # three solid, fully-saturated buttons stacked right under the
        # question competed with it for attention rather than reading as
        # secondary controls the way the light top-nav boxes do. Uses
        # `[class*=...]` substring matching on a fixed prefix (submitbox/
        # skipbox/restartbox) rather than baking `idx` into the selector, so
        # one static block covers every question's container key (…_0, …_1,
        # …) without re-injecting CSS with a new idx-specific selector on
        # every question.
        inject_button_tint_css("submitbox", "#f0fdf4", "#86efac", "#15803d", hover_bg="#dcfce7")
        inject_button_tint_css("skipbox", "#fffbeb", "#fcd34d", "#b45309", hover_bg="#fef3c7")
        inject_button_tint_css("restartbox", "#fef2f2", "#fca5a5", "#b91c1c", hover_bg="#fee2e2")
        with col_submit:
            with st.container(key=f"submitbox_{idx}"):
                submit = st.button("Submit", key=f"submit_{idx}", use_container_width=True)
        with col_skip:
            with st.container(key=f"skipbox_{idx}"):
                skip = st.button("Skip", key=f"skip_{idx}", use_container_width=True)
        with col_restart:
            with st.container(key=f"restartbox_{idx}"):
                # Moved here by request -- this used to sit next to Next
                # question, only appearing after revealing an answer. It's
                # really a pre-answer decision ("I don't want this session
                # after all"), so it lives alongside Submit/Skip instead.
                # Label shortened by request too: it only ever restarts the
                # session with whatever filters were already chosen on the
                # setup screen -- it doesn't open a filter-editing UI itself,
                # so "Change filters /" was misleading.
                if st.button("⚙️ Restart session", key=f"changefilters_{idx}", use_container_width=True):
                    st.session_state.session_questions = []
                    st.rerun()

        if submit and choice is None:
            st.warning("Select an option before submitting, or use Skip.")
            return

        if submit or skip:
            selected_label = OPT_SKIPPED if skip else choice.split(")")[0]
            is_correct = selected_label == q.get("answer")

            save_response({
                "user": user,
                "session_id": st.session_state.session_id,
                "question_id": q["question_id"],
                "exam": q["exam"],
                "paper": q["paper"],
                "year": q["year"],
                "theme": q["theme"],
                "selected": selected_label,
                "confidence": None if skip else confidence,
                "correct_answer": q.get("answer"),
                "timestamp": datetime.now().isoformat(),
            })

            update_schedule_atomic(
                user, q["question_id"], is_correct, skip, datetime.now(),
                confidence=None if skip else confidence,
            )

            st.session_state.session_results.append({
                "theme": q["theme"], "correct": is_correct, "skipped": skip,
                "exam": q["exam"], "paper": q["paper"],
            })
            st.session_state.selected_letter = selected_label
            st.session_state.revealed = True
            save_active_session(
                user, qs, st.session_state.current_idx,
                st.session_state.session_results, st.session_state.session_id,
                revealed=True, selected_letter=selected_label,
            )
            st.rerun()
        return

    # ---- Revealed state: show correct/wrong instantly + explanation ----
    selected_letter = st.session_state.selected_letter
    correct_letter = q.get("answer")

    for o in q["options"]:
        if not o["text"]:
            continue
        label = f"{o['label']}) {o['text']}"
        if o["label"] == correct_letter:
            st.success(label)
        elif o["label"] == selected_letter:
            st.error(label)
        else:
            st.write(label)

    if selected_letter == OPT_SKIPPED:
        st.info("Skipped — counted as 0, same as the real exam.")
    elif selected_letter == correct_letter:
        st.caption("Correct.")
    else:
        st.caption("Not quite.")

    with st.expander("Explanation", expanded=True):
        explanation = q.get("explanation")
        if explanation:
            render_explanation_text(explanation)
        else:
            st.caption("No explanation available for this question yet.")
        correlation = q.get("syllabus_correlation")
        if correlation:
            st.caption(
                f"📖 Syllabus: {correlation['topic']} — Chapter {correlation['chapter_2025_act']} of the "
                f"Income-tax Act, 2025 (Chapter {correlation['chapter_1961_act']} of the 1961 Act)"
            )

    my_bookmarks = load_bookmarks().get(user, {})
    is_bookmarked = q["question_id"] in my_bookmarks

    # Narrow, equal-width columns (with a spacer soaking up the rest of the
    # row) rather than a 50/50 split -- previously Next question stretched
    # to fill half the row while the bare checkbox next to it stayed tiny by
    # comparison, even though they were nominally in equal columns. Boxing
    # the checkbox in a bordered st.container() and giving both boxes the
    # same narrow column width makes them read as two same-sized boxes
    # instead. Colors follow the same green="proceed/positive" and
    # amber="save/star" language used elsewhere (Submit and Smart Quiz are
    # also green; Bookmarks tab is also amber) -- same scoped st-key
    # container-CSS pattern as those, just applied to the container itself
    # for the checkbox box rather than to a nested button, since a checkbox
    # has no button element to target.
    inject_button_tint_css("nextbox", "#f0fdf4", "#86efac", "#15803d", hover_bg="#dcfce7")
    st.markdown(
        """
        <style>
        [class*="st-key-bookmarkbox"] {
            background-color: #fffbeb !important;
            border: 1px solid #fcd34d !important;
            border-radius: 0.5rem !important;
            padding: 0.5rem 1rem !important;
            display: flex !important;
            align-items: center !important;
        }
        [class*="st-key-bookmarkbox"] p {
            color: #b45309 !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    # Back to a plain 50/50 split (was narrowed with a trailing spacer
    # column before) -- that narrow width was what caused "Bookmark this Qn"
    # to wrap onto two lines, and clustered both boxes on the left side of
    # the row instead of opposite ends. Half-width was already proven to
    # comfortably fit "Next question ->" on one line, and "Bookmark this Qn"
    # is similar length, so this fixes the wrapping and the positioning at
    # the same time -- each box still matches the other's size (both are
    # exactly half the row), just half the row now instead of a quarter.
    col_next, col_bookmark = st.columns(2)
    with col_next:
        with st.container(key=f"nextbox_{idx}"):
            if st.button("Next question →", key=f"next_{idx}", use_container_width=True):
                st.session_state.current_idx += 1
                st.session_state.revealed = False
                st.session_state.selected_letter = None
                st.session_state.nav_idx = st.session_state.current_idx
                save_active_session(
                    user, qs, st.session_state.current_idx,
                    st.session_state.session_results, st.session_state.session_id,
                    revealed=False, selected_letter=None,
                )
                st.rerun()
    with col_bookmark:
        # Moved here from its own standalone row below, by request -- sits
        # next to Next question now. "Restart session" (previously in this
        # spot, under its old longer name "Change filters / Restart
        # session") moved the other way, down to the Submit/Skip row -- it's
        # really a pre-answer decision ("I don't want this session after
        # all"), not a post-reveal one, so it makes more sense paired with
        # Submit/Skip than with Next question.
        #
        # border=True dropped here -- Streamlit's own bordered-container
        # padding (a fixed ~1rem by default) was taller than a button's
        # natural padding, which is why this box was visibly taller than
        # "Next question" next to it. Building the border/padding entirely
        # from the CSS above instead (rather than overriding Streamlit's
        # own, whichever nested element it's actually applied to) sidesteps
        # that mismatch directly, sized to approximate a button's height.
        with st.container(key=f"bookmarkbox_{idx}"):
            bookmark_now = st.checkbox(
                "⭐ Bookmark this Qn",
                value=is_bookmarked,
                key=f"bookmark_{q['question_id']}_{idx}",
            )
    if bookmark_now != is_bookmarked:
        toggle_bookmark(user, q["question_id"])
        st.rerun()

    # Reuses the same plaintext formatter as Bookmarks/Question Bank
    # (format_question_plaintext) instead of the hand-built copy_text this
    # used to construct separately -- that hand-built version only ever
    # included the question stem and the answer, missing the options
    # entirely, which made a copied question useless for actually solving
    # or sharing without the source app open alongside it.
    with st.expander("Copy this question"):
        st.code(format_question_plaintext(q, include_answer=True), language=None, wrap_lines=True)

    render_community_notes(q["question_id"], user)


def render_community_notes(question_id, user):
    notes = load_notes().get(question_id, [])
    flags = [n for n in notes if n.get("is_flag")]
    if flags:
        st.warning(f"⚠️ This question has been flagged {len(flags)} time(s) for a possible error — see notes below.")

    with st.expander(f"Community notes ({len(notes)})", expanded=bool(flags)):
        for n in notes:
            icon = "🚩" if n.get("is_flag") else "💬"
            when = n["timestamp"][:10]
            st.markdown(f"{icon} **{n['user']}** · {when}")
            st.write(n["note"])
            st.divider()
        if not notes:
            st.caption("No notes yet — be the first to flag an issue or add a tip.")

        new_note = st.text_area("Add a note or challenge the answer key:", key=f"note_input_{question_id}")
        is_flag = st.checkbox(
            "This flags a possible error in the question or answer key",
            key=f"flag_{question_id}",
        )
        if st.button("Post note", key=f"post_note_{question_id}"):
            if new_note.strip():
                add_note(question_id, user, new_note.strip(), is_flag)
                st.rerun()
            else:
                st.warning("Write something before posting.")


# ---------- Report tab ----------

def render_report(questions, user, source_file_by_id):
    responses = [r for r in load_responses() if r.get("user") == user]
    if not responses:
        st.info("No practice sessions recorded yet — attempt some questions first.")
        return

    # Responses to questions currently excluded from practice (answer key
    # missing/disputed/conflicting) are dropped here, up front, rather than
    # counted and then caveated with an explanatory caption -- by request.
    # Filtered by response RECORD, not by distinct question id: a question
    # attempted more than once before being excluded has ALL of those
    # attempts dropped, not just one, so this stays exactly consistent with
    # Overall Score's denominator below rather than the two numbers
    # quietly describing different populations.
    #
    # Real tradeoff worth knowing: these all-time numbers can now DECREASE
    # over time (if a question you'd already answered gets excluded later,
    # its responses disappear from every count here), where before they
    # only ever went up. That's the direct consequence of removing the old
    # "N of these are excluded" caption that used to explain the gap.
    excluded_ids = {q["question_id"] for q in questions if not is_valid_for_practice(q)}
    responses = [r for r in responses if r.get("question_id") not in excluded_ids]
    if not responses:
        st.info("No practice sessions recorded yet for questions still eligible for practice.")
        return

    theme_attempts = defaultdict(int)
    for r in responses:
        theme_attempts[r["theme"]] += 1

    graded = [r for r in responses if r.get("correct_answer")]
    ungraded_count = len(responses) - len(graded)

    # All-time activity and Scored results merged into one card (were two
    # side-by-side plain st.metric() columns before) -- styled table format
    # by request, modeled on a reference mockup. See render_metric_card()
    # for why this needs custom HTML rather than a native Streamlit widget.
    if not graded:
        st.subheader("All-time activity")
        st.metric("Total Questions attempted", len(responses))
        st.info("No graded responses yet.")
    else:
        correct = sum(1 for r in graded if r["selected"] == r["correct_answer"])
        all_time_pct = 100 * correct / len(graded)

        # Rolling recent accuracy -- the lifetime Overall Score below moves
        # less and less per question the more history you build up, so it
        # can't show whether you're actually improving right now. This is
        # the same graded set, just the most recent slice of it (up to
        # 100), with a delta against the overall rate so the arrow/color
        # directly answers "am I trending up or down" rather than making
        # you compare two raw numbers yourself.
        graded_by_time = sorted(graded, key=lambda r: r.get("timestamp", ""))
        recent = graded_by_time[-100:]
        recent_correct = sum(1 for r in recent if r["selected"] == r["correct_answer"])
        recent_pct = 100 * recent_correct / len(recent)
        recent_delta = recent_pct - all_time_pct
        delta_arrow = "↑" if recent_delta >= 0 else "↓"
        delta_color = "#15803d" if recent_delta >= 0 else "#b91c1c"

        footer_lines = []
        if ungraded_count:
            footer_lines.append(
                f"{ungraded_count} attempted question(s) aren't included here — they were "
                "answered before a valid answer key existed for them, so there's nothing to "
                "grade against."
            )
        skipped_total = sum(1 for r in graded if r["selected"] == OPT_SKIPPED)
        if skipped_total:
            footer_lines.append(f"Includes {skipped_total} skipped questions, scored as 0 — same as the real exam.")

        render_metric_card(
            icon="📊", title="All-time Activity", subtitle="Your overall practice performance",
            header_bg="#eff6ff", accent="#93c5fd",
            rows=[
                {"icon": "📄", "label": "Total Questions attempted", "value": str(len(responses)), "detail": "—"},
                {"icon": "🏆", "label": "Overall Score", "value": f"{correct}/{len(graded)}",
                 "value_color": "#15803d", "detail": f"{all_time_pct:.1f}% correct"},
                {"icon": "📈", "label": f"Last {len(recent)} Qn Score", "value": f"{recent_correct}/{len(recent)}",
                 "value_color": "#6d28d9",
                 "detail": f'<span style="color:{delta_color};">{delta_arrow} {recent_delta:+.0f}% vs Overall Score</span>'},
            ],
            footer_lines=footer_lines,
        )

    # ---- Spaced repetition -- kept right below All-time activity and
    # above the backup button, by request: this is the most actionable
    # summary (what's due right now, how much of the bank you've covered),
    # so it leads the tab. Repeatedly missed / Last 5 quizzes / the other
    # breakdowns stay below the backup button -- see the comment down there.
    schedule = load_schedule()
    my_due = sum(
        1 for key, entry in schedule.items()
        if key.startswith(f"{user}::") and is_due(entry, datetime.now())
    )

    # Bank coverage is scoped to currently-practiceable questions FOR THIS
    # USER (same population the Smart Quiz setup screen draws from -- see
    # render_practice() and questions_flagged_by()), so a disputed/missing-
    # key question, or one this user has personally flagged and is waiting
    # on review, doesn't count against them as "not yet attempted."
    practiceable = [
        q for q in questions
        if is_valid_for_practice(q) and q["question_id"] not in questions_flagged_by(user)
    ]
    spaced_rows = [
        {"icon": "🗓️", "label": "Questions due for review right now", "value": str(my_due),
         "value_color": "#b91c1c", "detail": "Review these to improve retention."},
    ]
    if practiceable:
        # responses here has already had currently-excluded questions
        # filtered out (see the top of this function), but that has no
        # practical effect on this particular intersection -- an excluded
        # question can never appear in practiceable_ids anyway. Recomputed
        # locally rather than reusing a variable from up top, since this is
        # the only place in the function that needs it.
        attempted_ids = {r["question_id"] for r in responses}
        practiceable_ids = {q["question_id"] for q in practiceable}
        covered = len(attempted_ids & practiceable_ids)
        coverage_pct = 100 * covered / len(practiceable_ids)
        spaced_rows.append({
            "icon": "🗄️", "label": "Bank coverage", "value": f"{covered}/{len(practiceable_ids)}",
            "value_color": "#b45309",
            "detail": f"{coverage_pct:.0f}% of practiceable questions attempted at least once.",
        })

    render_metric_card(
        icon="🔄", title="Spaced Repetition", subtitle="Keep practicing to strengthen your weak areas",
        header_bg="#f0fdf4", accent="#86efac", rows=spaced_rows,
    )

    if _github_config():
        # Moved here (was at the very bottom of the tab) so it's reachable
        # right after the score summary without scrolling past everything
        # else -- by request. Light-green "go" tint, same scoped st-key
        # container-CSS pattern used for every other tinted button in the
        # app (Submit, Smart Quiz, Start Session, etc.).
        inject_button_tint_css("report_backup_box", "#f0fdf4", "#86efac", "#15803d", hover_bg="#dcfce7")
        with st.container(key="report_backup_box"):
            if st.button("Back up my progress now", use_container_width=True):
                ok, msg = github_backup()
                st.success("Backed up.") if ok else st.warning(f"Backup failed: {msg}")

    # ---- Repeatedly missed -- deliberately stays here, below the backup
    # button, rather than moving up with Spaced repetition above: this one
    # is grouped with the more analytical breakdowns further down (Last 5
    # quizzes, Current adaptive priority, Accuracy by theme, Questions
    # attempted by theme) instead of with the actionable Spaced repetition
    # summary. No divider before, between, or after any of those five --
    # they sit back-to-back with the same even Streamlit-default gap, by
    # request, rather than dividers breaking some of them apart from others.
    wrong_streaks = wrong_streak_counts(load_responses(), user)
    # "Twice or more" == streak >= 2.
    repeatedly_missed = sorted(
        ((qid, streak) for qid, streak in wrong_streaks.items() if streak >= 2),
        key=lambda pair: -pair[1],
    )
    if repeatedly_missed:
        questions_by_id_report = {q["question_id"]: q for q in questions}
        with st.expander(f"Repeatedly Missed Qns ({len(repeatedly_missed)})"):
            st.caption("Questions you're currently on a wrong-answer streak of 2 or more on.")
            # Every question meeting the threshold is listed -- no cap, by
            # request (the old version capped at 20 with a "...and N more"
            # caption, back when every entry was a permanently-visible
            # bullet; now that each one is collapsed by default until
            # clicked, a long list stays just as compact as a short one).
            #
            # Streamlit can't nest an expander inside another expander (this
            # whole section already sits inside one), so "click to read the
            # full question" is built as a toggle button + a session_state
            # flag instead -- same pattern already used for AI Explanation
            # Import's suggestion preview. key_prefix="repmiss" is safe to
            # share across every question in this loop despite render_full_
            # question()'s "unique per call site" rule (see its docstring):
            # each question_id can only appear once in wrong_streaks to
            # begin with, so there's no risk of two calls in this same loop
            # building the same widget key.
            for qid, streak in repeatedly_missed:
                qobj = questions_by_id_report.get(qid)
                if not qobj:
                    continue
                # Flattened to one line before truncating -- some question
                # types ("Match the following", multi-statement lists) have
                # real line breaks in their stem, and a literal newline
                # inside a button's label makes Streamlit render each side
                # of it as its own centered markdown paragraph instead of
                # one continuous left-aligned line, which looked broken/
                # misaligned specifically on those questions.
                stem = " ".join(qobj["question"].split())
                preview = (stem[:70] + "…") if len(stem) > 70 else stem
                open_key = f"repmiss_open_{qid}"
                is_open = st.session_state.get(open_key, False)
                arrow = "▾" if is_open else "▸"
                if st.button(
                    f"{arrow} {qid} — {preview} ({streak} wrong in a row)",
                    key=f"repmiss_btn_{qid}", use_container_width=True,
                ):
                    st.session_state[open_key] = not is_open
                    st.rerun()
                if is_open:
                    render_full_question(
                        qobj, key_prefix="repmiss",
                        include_answer_in_copy=True, copy_for_ai=True,
                    )

    # ---- Last 5 quizzes -- full wrong-answer review, not just a count.
    # Deliberately shows the actual questions (render_full_question, same
    # presentation Bookmarks and Reader mode use) rather than a bare list of
    # IDs, since the point is letting you re-study what you missed without
    # having to go find each question again yourself. copy_for_ai=True here
    # for the same reason Bookmarks uses it and Question Bank doesn't: a
    # question that was actually SERVED in a quiz already passed is_valid_
    # for_practice() (disputed/invalid questions never make it into a
    # session to begin with), so its answer is reliably confirmed -- no risk
    # of presenting an unconfirmed answer as settled fact to whatever AI
    # tool the copied text gets pasted into.
    #
    # One quiz shown at a time via a selectbox, rather than all 5 stacked
    # together -- by request. With every wrong question fully rendered
    # (stem, options, explanation), 5 sessions' worth stacked back-to-back
    # got long fast; picking one keeps the expander focused on a single
    # quiz's worth of review at a time.
    questions_by_id_last5 = {q["question_id"]: q for q in questions}
    last5 = last_n_sessions_wrong_questions(load_responses(), questions_by_id_last5, user, n=5)
    if last5:
        total_wrong = sum(len(qs) for _, qs in last5)
        is_admin_report = _is_admin(user)
        with st.expander(f"Last {len(last5)} quizzes — wrong answers ({total_wrong})"):
            def _quiz_label(i):
                session_label = last5[i][0][:16].replace("T", " ")
                wrong_count = len(last5[i][1])
                return f"{session_label} — {wrong_count} wrong"

            picked = st.selectbox(
                "Select a quiz", range(len(last5)), format_func=_quiz_label, key="last5_quiz_picker",
            )
            session_id, wrong_qs = last5[picked]
            if not wrong_qs:
                st.success("Nothing wrong in this quiz — perfect score.")
            for idx, q in enumerate(wrong_qs):
                # Wraps the qid label in its own keyed container so the
                # negative margin below has a container guaranteed to exist
                # and match -- the earlier version tried to target render_
                # full_question's internal stem container by reconstructing
                # its key from session_id, but session_id is a timestamp
                # containing colons and periods that Streamlit likely
                # strips/rewrites when building the real CSS class name, so
                # that selector probably never matched anything. `idx` here
                # is plain digits, safe either way, and only needs to be
                # unique among the questions actually on screen right now
                # (one quiz's wrong_qs at a time, per the selectbox above),
                # not globally.
                qid_key = f"last5_qidbox_{idx}"
                st.markdown(
                    f"""<style>[class*="st-key-{qid_key}"] p {{ margin-bottom: -1rem !important; }}</style>""",
                    unsafe_allow_html=True,
                )
                with st.container(key=qid_key):
                    st.markdown(f"**{q['question_id']}**")
                render_full_question(
                    q, key_prefix=f"last5_{session_id}",
                    include_answer_in_copy=True, copy_for_ai=True,
                )
                if is_admin_report:
                    render_explanation_editor(q, user, source_file_by_id)

    # ---- Current adaptive priority + Accuracy by theme -- paired right
    # below Spaced repetition, by request, and both now expanders (were
    # plain subheader+list sections before) rather than always-open
    # walls of text. They're natural neighbors: one is the algorithm's
    # current per-theme weakness signal, the other is your actual
    # historical per-theme accuracy -- worth reading side by side, and
    # collapsed by default so this section doesn't dominate the tab on a
    # bank with many themes. No divider before, between, or after this
    # pair -- all four expanders in this cluster (Repeatedly missed,
    # Current adaptive priority, Accuracy by theme, Questions attempted
    # by theme) now sit back-to-back with the same even Streamlit-default
    # gap, by request, rather than dividers breaking some of them apart
    # from others.
    weakness_scores = compute_theme_weakness(load_responses(), user)
    if weakness_scores:
        ranked = sorted(weakness_scores.items(), key=lambda x: -x[1])
        with st.expander("Current adaptive priority (weakest first)"):
            st.caption(
                "This is the theme-level signal 'Prioritize weak spots and due reviews' uses "
                "to weight which new (not-yet-attempted) questions come up first."
            )
            for theme, score in ranked:
                st.write(f"- {theme}: {score * 100:.0f}% weak")

    if graded:
        theme_stats = defaultdict(lambda: {"correct": 0, "total": 0})
        for r in graded:
            t = r["theme"]
            theme_stats[t]["total"] += 1
            if r["selected"] == r["correct_answer"]:
                theme_stats[t]["correct"] += 1

        ranked = sorted(theme_stats.items(), key=lambda x: x[1]["correct"] / x[1]["total"])
        with st.expander("Accuracy by theme (weakest first)"):
            for theme, stats in ranked:
                pct = 100 * stats["correct"] / stats["total"]
                st.write(f"**{theme}** — {stats['correct']}/{stats['total']} ({pct:.0f}%)")

    with st.expander("Questions attempted by theme"):
        for theme, count in sorted(theme_attempts.items(), key=lambda x: -x[1]):
            st.write(f"- {theme}: {count}")


# ---------- Bookmarks tab ----------

def format_question_plaintext(q, include_answer=False):
    """Plain, unformatted rendering of a question + its options -- meant for
    st.code()'s built-in copy-to-clipboard icon, so line breaks and option
    labels survive a paste into WhatsApp/notes/etc. Explanation is always
    left out (too long to be useful in a quick-copy context).

    include_answer defaults to False -- e.g. the Question Bank tab's copy
    text stays answer-free, since those questions are exactly the ones
    whose recorded answer isn't reliably confirmed yet (missing, disputed,
    or conflicting) and a bare "Answer: X" line there would misrepresent
    that. Only render_full_question's Bookmarks-tab caller currently opts
    in. Even with include_answer=True, the line is only added when the
    recorded answer actually names an option with real text -- same
    validity check as is_valid_for_practice() -- so an unresolved/invalid
    answer key still doesn't get a fabricated-looking "Answer:" line."""
    lines = [f"{q['question_id']} · {q['theme']} · {q['exam']} {q['year']}", "", q["question"]]
    opts = [o for o in q.get("options", []) if o.get("text")]
    if opts:
        lines.append("")
        lines.extend(f"{o['label']}) {o['text']}" for o in opts)
    if include_answer:
        correct_opt = next((o for o in opts if o["label"] == q.get("answer")), None)
        if correct_opt:
            lines.append("")
            lines.append(f"Answer: {correct_opt['label']}) {correct_opt['text']}")
    return "\n".join(lines)


def format_question_for_ai_copy(q):
    """Formats a question (stem, options, correct answer) plus a short
    instruction preamble aimed at an AI chat tool, for the "Copy question
    text for AI" button in render_full_question() (Bookmarks tab).

    The instructions deliberately steer AWAY from tables. An earlier
    version asked for "real Markdown table syntax" instead of CSV/Sheets
    output, which helps only up to a point: once an AI tool renders a
    table in its OWN chat UI, copying from that rendered view captures
    whatever the browser hands back, which isn't reliably the underlying
    markdown source -- that step happens outside the AI's output entirely,
    so no wording here can fully control it. Bullet points sidestep the
    problem instead of trying to word around it: there's no row/column
    structure to lose in the first place. The "Often confused with" bullet
    is a pedagogical add specifically for exam prep -- flagging related
    provisions students commonly mix up on a differently-phrased question.

    LaTeX math was also downgraded to plain-text math, for the identical
    reason as the table instruction above: an AI tool's rendered math
    notation may not survive a copy out of its chat UI intact either (this
    is plausibly what mangled a fraction into "31" in one incident, on top
    of the separate CSV-export issue). render_explanation_text() still
    renders $.../$$...$$ correctly if it ever shows up anyway (nothing
    about display support changed here) -- this only changes what's
    requested, since asking for it hasn't proven reliable enough in
    practice to be worth the risk.

    Deliberately does NOT include the question's current explanation --
    this should be a clean input every time: question, options, correct
    answer, then whatever AI tool it's pasted into writes a fresh
    explanation, rather than being anchored on (or asked to just tweak)
    whatever's already saved.

    Nothing here is sent anywhere automatically -- this only produces text
    for the person to paste into whatever AI tool they're already using,
    then bring the result back to render_explanation_editor() themselves
    if they want to save it. Any explanation from that still needs
    verifying against a primary source before saving, same as always."""
    lines = [
        "Please write a clear, accurate explanation for this UPSC/HCS General "
        "Studies exam practice question. Requirements:",
        "- Cite the specific fact, date, article/provision, or data point that "
        "supports the answer.",
        "- Structure it as short bullet points, not paragraphs or tables -- "
        "tables and other multi-column formatting often lose their structure "
        "when copied out of a chat interface; plain bullets survive far more "
        "reliably.",
        "- Add a short 'Often confused with' bullet list if relevant -- related "
        "or similarly-worded provisions a student might mix this up with on a "
        "differently-phrased exam question, and the one or two details that "
        "actually distinguish them.",
        "- Write any formula or fraction in plain, readable text (e.g., "
        "\"1/3rd of the pension amount, capped at ₹15,000\") rather than "
        "LaTeX or other special math notation -- same reliability reason as "
        "the bullet-point instruction above: rendered math notation can "
        "also fail to survive a copy out of a chat interface intact.",
        "- Keep it concise and exam-focused.",
        "",
        f"Question ({q.get('exam', '')} {q.get('paper', '')} {q.get('year', '')}"
        f"{', ' + q['theme'] if q.get('theme') else ''}):",
        q["question"],
        "",
    ]
    opts = [o for o in q.get("options", []) if o.get("text")]
    if opts:
        lines.extend(f"{o['label']}) {o['text']}" for o in opts)
        lines.append("")
    correct_opt = next((o for o in opts if o["label"] == q.get("answer")), None)
    if correct_opt:
        lines.append(f"Correct answer: {correct_opt['label']}) {correct_opt['text']}")
    return "\n".join(lines)


def render_bookmarks(questions, user, source_file_by_id):
    my_bookmarks = load_bookmarks().get(user, {})

    if not my_bookmarks:
        st.info("No bookmarks yet — star a question after answering it (in the Smart Quiz tab) to save it here.")
        return

    is_admin = _is_admin(user)
    questions_by_id = {q["question_id"]: q for q in questions}

    # Solid green fill for "go/start" actions, matching Smart Quiz/Submit/
    # Next question elsewhere in the app.
    inject_button_tint_css("start_bookmarks_practice_box", "#f0fdf4", "#86efac", "#15803d", hover_bg="#dcfce7")

    with st.container(key="start_bookmarks_practice_box"):
        start_bookmark_session = st.button(
            "🎯 Start practice with my bookmarks", use_container_width=True,
        )
    if start_bookmark_session:
        session_qs = [
            questions_by_id[qid] for qid in my_bookmarks
            if qid in questions_by_id and is_valid_for_practice(questions_by_id[qid])
        ]
        skipped = len(my_bookmarks) - len(session_qs)
        if not session_qs:
            st.warning(
                "None of your bookmarked questions currently have a scorable "
                "answer key, so there's nothing to start a session with yet."
            )
        else:
            random.shuffle(session_qs)
            start_session(session_qs, user)
            # Not a filter-built session -- clear any stored filters so
            # "Start a new session" at the end falls back to the setup
            # screen instead of silently resuming an older filtered session.
            st.session_state.last_session_params = None
            if skipped:
                st.session_state.pending_toast = (
                    f"Session ready — {skipped} bookmarked question(s) skipped "
                    "(no scorable answer key yet)."
                )
            else:
                st.session_state.pending_toast = "Session ready — here we go."
            st.session_state.active_tab = "practice"
            st.rerun()

    st.divider()

    # Theme filter -- st.selectbox is already "folded" (shows just the
    # current selection) until tapped, then opens into the full option
    # list, so no extra expander/toggle scaffolding is needed to get that
    # collapsed-by-default behavior. Full width rather than sharing a row:
    # theme names (e.g. "Definitions & Basis of Charge") run long enough
    # that a half-width box would clip them, especially on mobile, and
    # this is the only control governing the list below it, so giving it
    # the full row also makes that scope visually clear.
    theme_counts = defaultdict(int)
    for qid in my_bookmarks:
        q = questions_by_id.get(qid)
        if q:
            theme_counts[q["theme"]] += 1

    theme_options = ["All themes"] + sorted(theme_counts)
    theme_labels = {"All themes": f"All themes ({len(my_bookmarks)})"}
    theme_labels.update({t: f"{t} ({theme_counts[t]})" for t in theme_options[1:]})

    selected_theme = st.selectbox(
        "Filter by theme",
        theme_options,
        format_func=lambda t: theme_labels[t],
        key="bookmarks_theme_filter",
    )

    if selected_theme == "All themes":
        filtered_ids = list(my_bookmarks.keys())
    else:
        filtered_ids = [
            qid for qid in my_bookmarks
            if questions_by_id.get(qid, {}).get("theme") == selected_theme
        ]

    sorted_ids = sorted(filtered_ids, key=lambda qid: my_bookmarks[qid], reverse=True)

    # Paginated rendering -- each bookmarked question below renders a
    # "Copy for AI" button via render_copy_button(), which is a real
    # sandboxed <iframe> (st.components.v1.html()), not a lightweight
    # widget; admin accounts also get a full render_explanation_editor()
    # per question on top of that. 200+ of those stacked on one page is
    # what actually causes the lag, not the bookmark data itself -- so
    # revealing only a handful at a time keeps every rerun's render cost
    # small no matter how many bookmarks exist in total.
    #
    # reveal_count lives in session_state so "Next 5" ACCUMULATES (never
    # loses a question you already had open) and survives ordinary
    # reruns -- but resets back to 5 whenever the theme filter changes, so
    # switching to a narrower theme opens the same "last 5 first" way the
    # tab itself does, rather than carrying over however many were
    # revealed under the previous filter.
    if st.session_state.get("bookmarks_reveal_theme") != selected_theme:
        st.session_state.bookmarks_reveal_count = 5
        st.session_state.bookmarks_reveal_theme = selected_theme
    reveal_count = st.session_state.get("bookmarks_reveal_count", 5)

    visible_ids = sorted_ids[:reveal_count]
    theme_note = f" in {selected_theme}" if selected_theme != "All themes" else ""
    st.caption(f"Showing {len(visible_ids)} of {len(sorted_ids)} bookmarks{theme_note}.")

    for qid in visible_ids:
        q = questions_by_id.get(qid)
        with st.container(border=True):
            if q:
                st.markdown(f"**{qid}**  ·  _{q['theme']}_  ·  {q['exam']} {q['year']}")
                render_full_question(q, key_prefix="bookmark", include_answer_in_copy=True, copy_for_ai=True)
                if is_admin:
                    render_explanation_editor(q, user, source_file_by_id)
            else:
                # No quiz question exists for this id anymore to star/unstar
                # from -- unlike a normal bookmark (removed via the star
                # checkbox during a quiz, not from this list), a stale
                # entry like this has no other way to ever clear itself, so
                # it keeps its own explicit removal button.
                st.markdown(f"**{qid}**")
                st.caption("This question is no longer in the current data file.")
                if st.button("Remove bookmark", key=f"unbookmark_{qid}"):
                    toggle_bookmark(user, qid)
                    st.rerun()

    remaining = len(sorted_ids) - len(visible_ids)
    if remaining > 0:
        col_next, col_all = st.columns(2)
        with col_next:
            if st.button(f"Next five ({remaining} more)", key="bookmarks_next_5", use_container_width=True):
                st.session_state.bookmarks_reveal_count = reveal_count + 5
                st.rerun()
        with col_all:
            if st.button(f"Complete list ({len(sorted_ids)})", key="bookmarks_show_all", use_container_width=True):
                st.session_state.bookmarks_reveal_count = len(sorted_ids)
                st.rerun()


# ---------- Question Bank browser (replaces the old Needs Review dashboard) ----------
# Formerly three separate always-visible scrolling lists (Community-flagged /
# Missing keys / Disputed). Replaced by a single Load-a-paper -> Filter ->
# one-question-at-a-time browser, modeled on the standalone Question Bank
# Editor app's own Filters + Prev/picker/Next UI -- but reading from the
# already-loaded, always-current in-memory bank (load_questions()) instead
# of a separately-fetched GitHub file, and using this app's own richer
# is_valid_for_practice() definition of "scorable" rather than a plain
# answer-is-not-None check. Community-flagged / missing-key / disputed are
# now just three of the filter toggles below (plus Year/Theme/Question type/
# Subtopic/Difficulty/Question type/free-text search), not separate sections
# -- so sweeping the whole bank for issues means checking each toggle in turn
# per loaded paper, rather than scrolling three long lists.

def _missing_key_note(q):
    """Returns why a question can't be scored (is_valid_for_practice()==False),
    or None if it's fine -- same wording the old Missing-keys section used,
    just extracted so the browser can show it for any question, regardless of
    which filter got you there."""
    if q.get("_conflict"):
        return (
            "⚠️ Conflicting answer across source files — see Developer Mode "
            "for the details. Any highlight below reflects only one file's "
            "version, not a confirmed answer."
        )
    if not q.get("answer"):
        return "🔑 No recorded answer yet — nothing is highlighted below."
    matching_option = next((o for o in q.get("options", []) if o.get("label") == q["answer"]), None)
    if matching_option is None:
        return "🔑 The recorded answer doesn't match any option below — nothing is highlighted."
    if not matching_option.get("text"):
        return (
            "🔑 The recorded answer matches an option below, but that option has no "
            "text — nothing is highlighted, and the correct choice wouldn't be "
            "selectable in a live quiz either."
        )
    return None


def render_question_bank(questions, user, source_file_by_id):
    is_admin = _is_admin(user)
    open_flags = get_open_flags(load_notes(), load_resolved())

    exam_paper_options = sorted({(q["exam"], q["paper"]) for q in questions})

    def _paper_label(exam, paper):
        # "1" -> "I" via the same Roman-numeral table used elsewhere in this
        # file, so the label reads "ITI Paper I" the way the rest of the app
        # already talks about papers -- falls back to the raw value for any
        # non-numeric paper id rather than crashing on int().
        roman = _ROMAN_ORDER_UPPER[int(paper) - 1] if str(paper).isdigit() and 0 < int(paper) <= len(_ROMAN_ORDER_UPPER) else paper
        return f"{exam} Paper {roman} (all years)"

    with st.expander("📂 Load a paper", expanded=not st.session_state.get("qb_loaded", False)):
        choice = st.selectbox(
            "File", exam_paper_options, format_func=lambda ep: _paper_label(*ep), key="qb_file_choice",
        )
        if st.button("🔄 Load / Reload", type="primary", use_container_width=True, key="qb_load_btn"):
            # Clears the cached load_questions() result so this always reflects
            # the latest local /data files, not a copy cached earlier in the
            # session -- the closest equivalent here to the standalone editor's
            # "Load / Reload from GitHub", since this app already keeps the
            # whole merged bank in memory rather than fetching one file at a time.
            load_questions.clear()
            exam, paper = choice
            pool = [q for q in questions if q["exam"] == exam and q["paper"] == paper]
            st.session_state.qb_pool = sorted(pool, key=lambda q: q["question_id"])
            st.session_state.qb_nav_idx = 0
            st.session_state.qb_loaded = True
            # A fresh file needs fresh filters -- otherwise a Theme/Year pick
            # left over from the previous paper could silently filter the
            # newly-loaded one down to zero with no obvious reason why.
            for k in ("qb_f_years", "qb_f_themes", "qb_f_text",
                      "qb_f_has_explanation", "qb_f_has_answer", "qb_f_has_flag"):
                st.session_state.pop(k, None)
            st.rerun()

    if not st.session_state.get("qb_loaded"):
        st.info("Pick a file above, then tap Load / Reload to start browsing.")
        return

    pool = st.session_state.qb_pool

    with st.expander("🔍 Filters", expanded=False):
        years = sorted({q.get("year") for q in pool if q.get("year") is not None})
        themes = sorted({q.get("theme") for q in pool if q.get("theme")})

        # Centers every filter's own label ("Year", "Theme", "Has
        # explanation", etc.) above its box, and the selected/placeholder
        # text ("Choose options", "Any") inside it -- by request. Scoped to
        # this one keyed container so it only touches the Question Bank
        # tab's own filter row, not every selectbox/multiselect elsewhere in
        # the app (the profile picker, Smart Quiz setup screen, etc.).
        #
        # First attempt at the label half only set text-align on the inner
        # <p>, which had no visible effect -- DevTools inspection (thanks,
        # Sunny) showed why: data-testid="stWidgetLabel" is itself a flex
        # container, and its child (the span wrapping the actual text) is a
        # single flex item that hugs its own content width by default --
        # exactly the same "no leftover space for text-align to redistribute
        # within" situation already solved for the expander headers in
        # main() (see that comment for the fuller explanation). The fix is
        # the same one used there: justify-content on the FLEX CONTAINER
        # itself, not text-align on the text node buried inside it.
        #
        # The selected-value half (inside each box) gets the other half of
        # that same main()-expander-header technique too, rather than the
        # narrower 2-level guess from the first attempt: a blanket
        # justify-content on data-baseweb="select" AND every descendant of
        # it, so whichever level actually turns out to be the one with real
        # slack gets centered regardless of exactly how many BaseWeb wrapper
        # divs sit in between -- a no-op on any level that isn't a flex
        # container or has no slack, same reasoning as the expander fix.
        st.markdown(
            """
            <style>
            [class*="st-key-qb_filters_box"] [data-testid="stWidgetLabel"] {
                justify-content: center !important;
            }
            [class*="st-key-qb_filters_box"] [data-testid="stWidgetLabel"] p {
                text-align: center !important;
                width: 100%;
            }
            [class*="st-key-qb_filters_box"] div[data-baseweb="select"],
            [class*="st-key-qb_filters_box"] div[data-baseweb="select"] * {
                justify-content: center !important;
            }
            /* Three rounds of DevTools screenshots each caught a different
               level of BaseWeb's internal markup (a <p>, then a real
               <input placeholder="...">, then an unrelated overlay portal
               div) without ever quite landing on the actual text-bearing
               element for every filter's box -- selectbox and multiselect
               don't necessarily share identical internal structure even
               though they look alike. Rather than keep guessing tag by
               tag, this forces text-align onto literally EVERY descendant
               of the select control, whatever tag it turns out to be --
               !important here also overrides any inline style="text-align:
               left" BaseWeb may set directly on a given element, which a
               plain (non-!important) inherited value from an ancestor
               wouldn't have beaten.  */
            [class*="st-key-qb_filters_box"] div[data-baseweb="select"] * {
                text-align: center !important;
            }
            </style>
            """,
            unsafe_allow_html=True,
        )
        with st.container(key="qb_filters_box"):
            # Question type dropped -- every question in this bank is MCQ, so it
            # was never a real distinction to filter on. Theme and Search text
            # absorb the freed-up width instead of leaving a gap where it sat.
            fc1, fc2, fc3 = st.columns([1, 1.5, 2])
            f_years = fc1.multiselect("Year", years, key="qb_f_years")
            f_themes = fc2.multiselect("Theme", themes, key="qb_f_themes")
            f_text = fc3.text_input("Search text (ID or question)", key="qb_f_text")

            # Disputed dropped -- Has community flag already surfaces the same
            # "needs a second look" questions (a dispute is raised via the same
            # flagging path), so this was a redundant second way to find the
            # same set. The remaining three now split the row into equal thirds
            # instead of quarters, picking up the freed width proportionately.
            fc6, fc7, fc8 = st.columns(3)
            f_has_explanation = fc6.selectbox("Has explanation", ["Any", "Yes", "No"], key="qb_f_has_explanation")
            # Uses is_valid_for_practice() rather than a plain "answer is not
            # None" check -- this app's own definition of "confirmed/scorable"
            # also requires the answer to match a real, non-blank option (see
            # that function's docstring), which is stricter than the generic
            # editor's schema-agnostic version of this same filter.
            f_has_answer = fc7.selectbox("Has confirmed answer", ["Any", "Yes", "No"], key="qb_f_has_answer")
            f_has_flag = fc8.selectbox("Has community flag", ["Any", "Yes", "No"], key="qb_f_has_flag")

    filtered = pool
    if f_years:
        filtered = [q for q in filtered if q.get("year") in f_years]
    if f_themes:
        filtered = [q for q in filtered if q.get("theme") in f_themes]
    if f_has_explanation != "Any":
        want = f_has_explanation == "Yes"
        filtered = [q for q in filtered if bool((q.get("explanation") or "").strip()) == want]
    if f_has_answer != "Any":
        want = f_has_answer == "Yes"
        filtered = [q for q in filtered if is_valid_for_practice(q) == want]
    if f_has_flag != "Any":
        want = f_has_flag == "Yes"
        filtered = [q for q in filtered if (q["question_id"] in open_flags) == want]
    if f_text:
        ft = f_text.lower()
        filtered = [q for q in filtered if ft in q["question_id"].lower() or ft in q["question"].lower()]

    st.markdown(
        """<style>[class*="st-key-qb_match_count"] p { text-align: center !important; }</style>""",
        unsafe_allow_html=True,
    )
    with st.container(key="qb_match_count"):
        st.caption(f"{len(filtered)} of {len(pool)} questions match current filters.")

    if not filtered:
        st.warning("No questions match these filters.")
        return

    # Resets navigation to the first match whenever the actual filtered set
    # of ids changes (a filter was touched) -- comparing the full id tuple
    # rather than just the count, since two different filter combinations
    # can coincidentally match the same NUMBER of questions.
    filtered_ids = tuple(q["question_id"] for q in filtered)
    if st.session_state.get("qb_filtered_signature") != filtered_ids:
        st.session_state.qb_nav_idx = 0
        st.session_state.qb_filtered_signature = filtered_ids

    nav_idx = min(st.session_state.qb_nav_idx, len(filtered) - 1)
    st.markdown(
        f"<p style='text-align:center;'><b>Question {nav_idx + 1} of {len(filtered)}</b></p>",
        unsafe_allow_html=True,
    )

    col_prev, col_pick, col_next = st.columns([1, 5, 1])
    with col_prev:
        if st.button("◀ Prev", disabled=(nav_idx == 0), use_container_width=True, key="qb_prev"):
            st.session_state.qb_nav_idx = nav_idx - 1
            st.rerun()
    with col_pick:
        def _pick_label(i):
            # Flattened before truncating -- same reasoning as the
            # Repeatedly Missed Qns fix: some question types have real
            # line breaks in their stem, and a raw newline mid-label looks
            # broken in a compact single-line dropdown entry.
            full_stem = " ".join(filtered[i]["question"].split())
            stem = full_stem[:55]
            return f"{filtered[i]['question_id']} — {stem}{'…' if len(full_stem) > 55 else ''}"
        # Writes directly into the picker's own session_state key before it's
        # instantiated this run, rather than relying on the selectbox's
        # `index=` argument -- Streamlit only honors `index=` the FIRST time
        # a given key is created, so after a Prev/Next click this is what
        # actually keeps the dropdown showing the right question instead of
        # freezing on whatever was selected when the widget was first built.
        if st.session_state.get("qb_picker_synced") != nav_idx:
            st.session_state["qb_picker"] = nav_idx
            st.session_state["qb_picker_synced"] = nav_idx
        picked = st.selectbox(
            "Jump to", range(len(filtered)), format_func=_pick_label,
            key="qb_picker", label_visibility="collapsed",
        )
        if picked != nav_idx:
            st.session_state.qb_nav_idx = picked
            st.session_state.qb_picker_synced = picked
            st.rerun()
    with col_next:
        if st.button("Next ▶", disabled=(nav_idx == len(filtered) - 1), use_container_width=True, key="qb_next"):
            st.session_state.qb_nav_idx = nav_idx + 1
            st.rerun()

    q = filtered[nav_idx]
    qid = q["question_id"]
    meta_bits = [
        f"{q['exam']} Paper {q['paper']}", q.get("theme"), q.get("subtopic"),
        str(q["year"]) if q.get("year") else None,
    ]
    # Scoped negative margin (same st-key-container trick used for the
    # Repeatedly Missed Qns question-id label above) -- st.caption followed
    # by render_full_question's own keyed stem container left a noticeably
    # bigger gap here than the ~1rem default between ordinary stacked
    # elements, since the caption's own bottom margin and the stem
    # container's top margin were both adding up on top of Streamlit's
    # normal inter-block spacing. Pulling the caption's own box up instead
    # of trying to reach into render_full_question's internals keeps this
    # fix local to the Question Bank tab, without touching the shared
    # render_full_question()/render_justified() spacing used everywhere else.
    st.markdown(
        """<style>
        [class*="st-key-qb_meta_caption"] { margin-bottom: -1rem !important; }
        [class*="st-key-qb_meta_caption"] p { text-align: center !important; }
        </style>""",
        unsafe_allow_html=True,
    )
    with st.container(key="qb_meta_caption"):
        st.caption(" · ".join(b for b in meta_bits if b))

    key_note = _missing_key_note(q)
    if qid in open_flags:
        note = (
            "🚩 Community-flagged as a possible answer-key error — the "
            "highlighted option below is only the currently recorded "
            "answer, not a confirmed-correct one."
        )
    elif key_note:
        note = key_note
    elif q.get("disputed"):
        note = (
            "⚠️ This answer is disputed — a later review found reason to doubt "
            "it. Treat the highlighted option as currently recorded, not confirmed."
        )
    else:
        note = None

    render_full_question(q, key_prefix="qbank", note=note, include_answer_in_copy=True, copy_for_ai=True)
    if qid in open_flags:
        for f in open_flags[qid]:
            st.caption(f"🚩 **{f['user']}** · {f['timestamp'][:10]}")
            st.write(f["note"])
    if q.get("disputed") and q.get("dispute_note"):
        st.caption(q["dispute_note"])
    if key_note and q.get("verification_note"):
        st.caption(q["verification_note"])

    if is_admin:
        if qid in open_flags and st.button("Mark resolved", key=f"qb_resolve_{qid}"):
            mark_resolved(qid)
            st.rerun()
        render_typo_editor(q, user, context="qbank", source_file_by_id=source_file_by_id)
        render_answer_key_editor(q, user, context="qbank", source_file_by_id=source_file_by_id)
        render_explanation_editor(q, user, source_file_by_id)


if __name__ == "__main__":
    main()
