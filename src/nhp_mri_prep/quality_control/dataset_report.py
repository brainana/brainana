"""
All-subjects QC report (``all_subjects_report.html``).

One page next to the per-subject ``sub-XXX.html`` reports with two views:

* **Subject** -- exactly what a subject's own report shows, with a switcher
  (dropdown, prev/next, arrow keys) instead of one file per subject.
* **By step** -- one QC figure (e.g. skullstripping) for every subject stacked,
  which is how an outlier stands out.

Each subject's body and each step's figure list sit in a ``<template>``: only the
one on screen is in the DOM, so only its PNGs load. Figure paths are the same
relative ``sub-XXX/figures/...`` paths the subject reports use, which holds because
this page is written to the same directory.
"""

import html
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .data_findings import has_findings
from .reports import (
    ANAT_BLOCK_EYEBROWS,
    ANAT_BLOCK_ORDER,
    MODALITY_SECTIONS,
    SNAPSHOT_ORDER_INDEX,
    HtmlGenerator,
    _REPORT_CSS,
    _REPORT_JS,
    _render_sections,
    build_report_data,
)

DATASET_REPORT_NAME = "all_subjects_report.html"

# The subject body in the order the per-subject template lays it out.
_SUBJECT_BODY_SLOTS = (
    "SUMMARY_SECTION",
    "STATUS_SECTION",
    "DATA_FINDINGS_SECTION",
    "ANATOMICAL_SECTION",
    "FUNCTIONAL_SECTION",
    "FIELD_MAPPING_SECTION",
    "ABOUT_SECTION",
    "METHODS_SECTION",
)

_MODALITY_ORDER = {key: i for i, key in enumerate(MODALITY_SECTIONS)}


def _iter_subject_figures(organized: Dict[str, Any]):
    """Yield ``(modality, block, group_key, snapshot)`` in the subject report's order.

    Goes through the same grouping the subject page renders from, so the chips on a
    by-step figure match the headings on that subject's page.
    """
    for modality, (section_id, _title) in MODALITY_SECTIONS.items():
        data = organized.get(modality)
        if not data:
            continue
        groups = HtmlGenerator._group_snapshots_by_entities(data, section_id.lower())
        for group_key, members in groups.items():
            if isinstance(members, dict):  # anatomical: group -> T1w/T2w block -> list
                for block in ANAT_BLOCK_ORDER:
                    for snap in members.get(block, []):
                        yield modality, block, group_key, snap
            else:
                for snap in members:
                    yield modality, "", group_key, snap


def _step_key(modality: str, block: str, snap: Dict[str, Any]) -> Tuple:
    step_type = snap.get("snapshot_type") or snap["entities"].get("desc", "")
    return (modality, block, step_type)


def _step_sort_key(key: Tuple) -> Tuple:
    modality, block, step_type = key
    block_order = ANAT_BLOCK_ORDER.index(block) if block in ANAT_BLOCK_ORDER else 0
    return (
        _MODALITY_ORDER.get(modality, 99),
        # T1w steps, then T2w steps, as the subject page reads -- except the
        # longitudinal base blocks, which follow their cross-sectional twin.
        block_order,
        SNAPSHOT_ORDER_INDEX.get(step_type, 999),
        step_type,
    )


def _step_title(modality: str, block: str, snap: Dict[str, Any]) -> str:
    title = snap.get("description") or snap["filename"]
    if block:
        return f"{title} · {ANAT_BLOCK_EYEBROWS.get(block, block)}"
    if modality != "anatomical":
        return f"{title} · {MODALITY_SECTIONS[modality][1]}"
    return title


def _collect_steps(
    subjects: List[Tuple[str, Dict[str, Any]]],
) -> List[Dict[str, Any]]:
    """Regroup every subject's figures by processing step."""
    steps: Dict[Tuple, Dict[str, Any]] = {}
    # (modality, block) -> subjects that have any figure there; a subject without
    # T2w is not "missing" the T2w steps.
    present: Dict[Tuple[str, str], set] = {}
    for sub, data in subjects:
        for modality, block, group_key, snap in _iter_subject_figures(
            data["organized_snapshots"]
        ):
            present.setdefault((modality, block), set()).add(sub)
            key = _step_key(modality, block, snap)
            step = steps.setdefault(
                key,
                {
                    "key": key,
                    "title": _step_title(modality, block, snap),
                    "caption": snap.get("figure_description", ""),
                    "entries": [],
                },
            )
            step["entries"].append(
                {
                    "sub": sub,
                    "chips": HtmlGenerator._group_chips(group_key),
                    "path": snap["path"],
                }
            )

    ordered = [steps[k] for k in sorted(steps, key=_step_sort_key)]
    all_subs = [sub for sub, _ in subjects]
    for i, step in enumerate(ordered):
        step["id"] = f"step-{i + 1}"
        modality, block, _ = step["key"]
        have = {e["sub"] for e in step["entries"]}
        step["missing"] = [
            s for s in all_subs if s in present[(modality, block)] and s not in have
        ]
    return ordered


def _render_step(step: Dict[str, Any], n_subjects: int) -> str:
    n_have = len({e["sub"] for e in step["entries"]})
    caption = step["caption"]
    if caption:
        caption = caption[0].upper() + caption[1:]
    parts = [
        f'<div id="Step"><h1 class="section">{html.escape(step["title"])}</h1>',
        f'<p class="note step-meta">{n_have} of {n_subjects} subjects'
        + (
            f" · {len(step['entries'])} figures"
            if len(step["entries"]) != n_have
            else ""
        )
        + "</p>",
    ]
    if caption:
        parts.append(f'<div class="cap step-cap">{caption}</div>')
    if step["missing"]:
        links = " ".join(
            f'<a class="gchip sub-link" href="#{html.escape(s)}">{html.escape(s)}</a>'
            for s in step["missing"]
        )
        # Neutral, not a warning: some figures are optional (the full-FOV conform
        # view only exists when the FOV was cropped), so absence alone is no failure.
        parts.append(f'<div class="note step-missing">No figure for {links}</div>')
    for e in step["entries"]:
        path = html.escape(e["path"])
        sub = html.escape(e["sub"])
        parts.append(
            f"""<div class="fig step-fig">
<h2 class="group-head"><a class="gchip sub-link" href="#{sub}" title="Open this subject">{sub}</a>{e["chips"]}</h2>
<a href="{path}" target="_blank"><img class="svg-reportlet" loading="lazy" src="{path}" alt="{sub}" /></a>
</div>"""
        )
    parts.append("</div>")
    return "\n".join(parts)


def _count_figures(data: Dict[str, Any]) -> int:
    return sum(1 for _ in _iter_subject_figures(data["organized_snapshots"]))


def _menu_link(href: str, label: str, count: str, flag: str = "") -> str:
    return (
        f'<a href="#{html.escape(href)}" data-route="{html.escape(href)}">'
        f'<span class="lbl">{html.escape(label)}</span>{flag}'
        f'<span class="cnt">{html.escape(count)}</span></a>'
    )


def render_dataset_report(
    subjects: List[Tuple[str, Dict[str, Any]]], version: str
) -> str:
    """Render the all-subjects page from ``(sub-XXX, report_data)`` pairs."""
    steps = _collect_steps(subjects)
    n = len(subjects)

    templates, sub_links = [], []
    for sub, data in subjects:
        sections = _render_sections(data)
        body = "\n".join(sections[slot] for slot in _SUBJECT_BODY_SLOTS)
        templates.append(f'<template id="t-{sub}">\n{body}\n</template>')
        templates.append(
            f'<template id="n-{sub}">\n{sections["NAVIGATION_MENU"]}\n</template>'
        )
        n_figs = _count_figures(data)
        flag = (
            '<span class="flag" title="Data findings">●</span>'
            if has_findings(data.get("data_findings"))
            else ""
        )
        sub_links.append(
            _menu_link(
                sub, sub, f"{n_figs} fig" if n_figs == 1 else f"{n_figs} figs", flag
            )
        )

    step_links = []
    for step in steps:
        templates.append(
            f'<template id="t-{step["id"]}">\n{_render_step(step, n)}\n</template>'
        )
        n_have = len({e["sub"] for e in step["entries"]})
        step_links.append(_menu_link(step["id"], step["title"], f"{n_have}/{n}"))

    routes = {
        "subjects": [sub for sub, _ in subjects],
        "steps": [s["id"] for s in steps],
        "stepTitles": {s["id"]: s["title"] for s in steps},
    }
    first_sub = subjects[0][0]

    return _DATASET_TEMPLATE.format(
        VERSION=html.escape(version),
        STYLE=_REPORT_CSS + _DATASET_CSS,
        FIRST_SUB=html.escape(first_sub),
        SUB_MENU="\n".join(sub_links),
        STEP_MENU="\n".join(step_links),
        N_SUBJECTS=n,
        TEMPLATES="\n".join(templates),
        ROUTES=json.dumps(routes).replace("</", "<\\/"),
        SCRIPT=_REPORT_JS + _DATASET_JS,
    )


def generate_dataset_report(
    output_dir: Path,
    subject_ids: List[str],
    config: Dict[str, Any],
    run_status: Optional[Dict[str, Any]] = None,
    report_path: Optional[Path] = None,
    logger: Optional[logging.Logger] = None,
) -> Path:
    """Write ``all_subjects_report.html`` for the given subjects of a pipeline output directory."""
    logger = logger or logging.getLogger(__name__)
    output_dir = Path(output_dir)
    report_path = Path(report_path or output_dir / DATASET_REPORT_NAME)

    subjects = []
    for sid in subject_ids:
        sub = f"sub-{sid}"
        try:
            data = build_report_data(
                output_dir / sub / "figures",
                output_dir / f"{sub}.html",
                config,
                logger,
                run_status=run_status,
            )
        except Exception as e:  # one broken subject must not cost the others
            logger.warning("Report: skipping %s in %s - %s", sub, report_path.name, e)
            continue
        subjects.append((sub, data))

    return write_dataset_report(subjects, report_path, logger)


def write_dataset_report(
    subjects: List[Tuple[str, Dict[str, Any]]],
    report_path: Path,
    logger: Optional[logging.Logger] = None,
) -> Path:
    """Write the all-subjects page from already-built ``(sub-XXX, report_data)`` pairs."""
    from nhp_mri_prep.version import get_version

    logger = logger or logging.getLogger(__name__)
    if not subjects:
        raise RuntimeError("no subject could be read")

    report_path = Path(report_path)
    report_path.write_text(
        render_dataset_report(subjects, get_version()), encoding="utf-8"
    )
    logger.info("Output: all-subjects report written - %s", report_path)
    return report_path


_DATASET_CSS = """
.switch{display:flex;align-items:center;gap:4px;min-width:0}
.switch .arrow{border:1px solid var(--bn-border-mid);background:var(--bn-inset);color:var(--bn-ink);border-radius:6px;
  width:28px;height:28px;padding:0;cursor:pointer;font:inherit;font-size:17px;line-height:1}
.switch .arrow:hover:not(:disabled){background:var(--bn-surface)}
.switch .arrow:disabled{opacity:.35;cursor:default}
.switch .nav-dd>summary{font-weight:700;font-size:16px;color:var(--bn-ink);max-width:420px;overflow:hidden;text-overflow:ellipsis}
.switch .menu{min-width:300px}
.switch .menu input{width:100%;font:inherit;font-size:14px;padding:6px 10px;margin:0 0 6px;border:1px solid var(--bn-border-mid);
  border-radius:6px;background:var(--bn-inset);color:var(--bn-text)}
.switch .menu a{display:flex;align-items:baseline;gap:8px}
.switch .menu a .lbl{flex:1}
.switch .menu a .cnt{color:var(--bn-muted);font-family:var(--bn-mono);font-size:12px}
.switch .menu a .flag{color:var(--bn-warn);font-size:10px}
.switch .menu a.cur{background:var(--bn-surface);font-weight:600}
.mode{display:inline-flex;flex:none;border:1px solid var(--bn-border-mid);border-radius:var(--bn-r-pill);padding:2px;margin-left:10px}
.mode a{font-size:13px;padding:3px 11px;border-radius:var(--bn-r-pill);color:var(--bn-text);white-space:nowrap}
.mode a:hover{text-decoration:none;background:var(--bn-surface)}
.mode a.on{background:var(--bn-accent);color:var(--bn-ink);font-weight:600}
body[data-mode="step"] .topbar nav{display:none}
a.gchip.sub-link{color:var(--bn-ink)}
a.gchip.sub-link:hover{text-decoration:none;background:var(--bn-accent)}
.fig.step-fig .group-head{margin:0;padding-top:0;border-top:none}
.fig.step-fig img{margin-top:10px}
.step-missing{display:flex;flex-wrap:wrap;align-items:center;gap:6px;margin:8px 0 0}
.step-cap{margin:-8px 0 8px}
"""

# Hash routes: #sub-XXX (subject view) and #step-N (by-step view). Any other
# in-page anchor (the section nav, group headings) scrolls without touching the
# hash, so the route -- and reload / a shared link -- keeps pointing at the view.
_DATASET_JS = """
<script>
(function () {
  var R = JSON.parse(document.getElementById('routes').textContent);
  var main = document.querySelector('main'), nav = document.getElementById('nav');
  var label = document.getElementById('cur-label'), dd = document.getElementById('switch-dd');
  var prev = document.getElementById('prev'), next = document.getElementById('next');
  var last = {sub: R.subjects[0], step: R.steps[0]}, mode = 'sub', cur = null;

  function isRoute(id) { return R.subjects.indexOf(id) >= 0 || R.steps.indexOf(id) >= 0; }
  function list() { return mode === 'sub' ? R.subjects : R.steps; }

  function render(id) {
    mode = R.steps.indexOf(id) >= 0 ? 'step' : 'sub';
    cur = id; last[mode] = id;
    document.body.dataset.mode = mode;
    main.replaceChildren(document.getElementById('t-' + id).content.cloneNode(true));
    var n = document.getElementById('n-' + id);
    nav.replaceChildren(n ? n.content.cloneNode(true) : '');
    label.textContent = mode === 'sub' ? id : R.stepTitles[id];
    document.title = (mode === 'sub' ? id : R.stepTitles[id]) + ' \\u00b7 all subjects';
    document.querySelectorAll('.switch .menu').forEach(function (m) {
      m.hidden = m.dataset.for !== mode;
    });
    document.querySelectorAll('.switch .menu a').forEach(function (a) {
      a.classList.toggle('cur', a.dataset.route === id);
    });
    var l = list(), i = l.indexOf(id);
    prev.disabled = i <= 0; next.disabled = i >= l.length - 1;
    document.getElementById('mode-sub').href = '#' + last.sub;
    document.getElementById('mode-step').href = '#' + last.step;
    document.getElementById('mode-sub').classList.toggle('on', mode === 'sub');
    document.getElementById('mode-step').classList.toggle('on', mode === 'step');
    window.scrollTo({top: 0, behavior: 'instant'});
  }

  function route() {
    var h = decodeURIComponent(location.hash.slice(1));
    render(isRoute(h) ? h : R.subjects[0]);
  }
  function go(id) { if (id) { if (location.hash === '#' + id) route(); else location.hash = id; } }
  function step(d) { var l = list(); go(l[l.indexOf(cur) + d]); }

  prev.addEventListener('click', function () { step(-1); });
  next.addEventListener('click', function () { step(1); });

  document.addEventListener('click', function (e) {
    var a = e.target.closest && e.target.closest('a[href^="#"]');
    if (!a) return;
    var id = decodeURIComponent(a.getAttribute('href').slice(1));
    document.querySelectorAll('details.nav-dd[open]').forEach(function (d) { d.open = false; });
    if (isRoute(id)) return;  // let the hash change drive the view
    e.preventDefault();
    var el = document.getElementById(id);
    if (el) el.scrollIntoView();
  });

  document.addEventListener('keydown', function (e) {
    if (e.altKey || e.ctrlKey || e.metaKey || e.shiftKey) return;
    var t = e.target.tagName;
    if (t === 'INPUT' || t === 'SELECT' || t === 'TEXTAREA') return;
    if (e.key === 'ArrowLeft') { step(-1); e.preventDefault(); }
    else if (e.key === 'ArrowRight') { step(1); e.preventDefault(); }
  });

  // Type-to-filter in the switcher; Enter opens the first match.
  document.querySelectorAll('.switch .menu input').forEach(function (inp) {
    inp.addEventListener('input', function () {
      var q = inp.value.toLowerCase();
      inp.parentNode.querySelectorAll('a').forEach(function (a) {
        a.hidden = q && a.textContent.toLowerCase().indexOf(q) < 0;
      });
    });
    inp.addEventListener('keydown', function (e) {
      if (e.key !== 'Enter') return;
      var a = inp.parentNode.querySelector('a:not([hidden])');
      if (a) { dd.open = false; go(a.dataset.route); }
    });
  });
  dd.addEventListener('toggle', function () {
    if (!dd.open) return;
    var m = dd.querySelector('.menu:not([hidden])'), inp = m.querySelector('input');
    if (inp) { inp.value = ''; inp.dispatchEvent(new Event('input')); inp.focus(); }
    var c = m.querySelector('a.cur');
    if (c) c.scrollIntoView({block: 'nearest'});
  });

  if (!R.steps.length) document.getElementById('mode-step').hidden = true;
  window.addEventListener('hashchange', route);
  route();
})();
</script>
"""

_DATASET_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<meta name="generator" content="brainana {VERSION}" />
<title>All subjects</title>
<style type="text/css">{STYLE}</style>
</head>
<body data-mode="sub">
<header class="topbar"><span class="brand">brainana</span><span class="brand-sep">|</span>
<div class="switch">
<button class="arrow" id="prev" type="button" title="Previous (←)">‹</button>
<details class="nav-dd" id="switch-dd"><summary><span id="cur-label">{FIRST_SUB}</span> ▾</summary>
<div class="menu" data-for="sub"><input type="search" placeholder="Filter {N_SUBJECTS} subjects…" />
{SUB_MENU}
</div>
<div class="menu" data-for="step" hidden><input type="search" placeholder="Filter steps…" />
{STEP_MENU}
</div></details>
<button class="arrow" id="next" type="button" title="Next (→)">›</button>
</div>
<span class="mode"><a id="mode-sub" class="on" href="#{FIRST_SUB}">Subject</a><a id="mode-step" href="#">By step</a></span>
<nav id="nav"></nav>
</header>
<main></main>
{TEMPLATES}
<script type="application/json" id="routes">{ROUTES}</script>
{SCRIPT}
</body>
</html>"""
