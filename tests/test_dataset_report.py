"""Tests for the all-subjects QC page (``all_subjects_report.html``).

The page is a browser over the per-subject reports: each subject's body in a
``<template>``, plus every figure regrouped by processing step. These pin that the
regrouping keeps the subject pages' chips and relative paths, that a subject which
lacks a figure is named, and that the per-subject reports link back.
"""

import json
import logging
import re

from nhp_mri_prep.quality_control.dataset_report import (
    generate_dataset_report,
    render_dataset_report,
)
from nhp_mri_prep.quality_control.reports import build_report_data, generate_qc_report

LOG = logging.getLogger(__name__)


def _figures(out, sub, names):
    fig_dir = out / f"sub-{sub}" / "figures"
    fig_dir.mkdir(parents=True)
    for name in names:
        (fig_dir / name).write_bytes(b"\x89PNG")


def _dataset(tmp_path):
    # sub-a: one session, with the optional full-FOV view.
    _figures(
        tmp_path,
        "a",
        [
            "sub-a_ses-01_desc-conformFullFOV_T1w.png",
            "sub-a_ses-01_desc-conform_T1w.png",
            "sub-a_ses-01_desc-skullstrip_T1w.png",
        ],
    )
    # sub-b: subject-level, no full-FOV view.
    _figures(
        tmp_path,
        "b",
        ["sub-b_desc-conform_T1w.png", "sub-b_desc-skullstrip_T1w.png"],
    )
    return tmp_path


def _routes(page):
    return json.loads(re.search(r'id="routes">(.*?)</script>', page, re.S).group(1))


def _template(page, tid):
    return re.search(rf'<template id="{tid}">(.*?)</template>', page, re.S).group(1)


def test_page_has_a_body_and_nav_per_subject(tmp_path):
    out = _dataset(tmp_path)
    path = generate_dataset_report(out, ["a", "b"], {}, logger=LOG)
    page = path.read_text(encoding="utf-8")

    assert path.name == "all_subjects_report.html"
    routes = _routes(page)
    assert routes["subjects"] == ["sub-a", "sub-b"]
    for sub in ("sub-a", "sub-b"):
        body = _template(page, f"t-{sub}")
        assert 'id="Summary"' in body and 'id="Methods"' in body
        assert f'<template id="n-{sub}">' in page
    # Paths stay relative to the output dir, where both kinds of report live.
    assert 'src="sub-a/figures/sub-a_ses-01_desc-skullstrip_T1w.png"' in page
    assert str(tmp_path) not in page


def test_steps_regroup_every_subject_and_name_the_absent(tmp_path):
    out = _dataset(tmp_path)
    page = generate_dataset_report(out, ["a", "b"], {}, logger=LOG).read_text(
        encoding="utf-8"
    )
    routes = _routes(page)
    titles = list(routes["stepTitles"].values())

    # Subject page order: full-FOV, conform, skullstrip.
    assert [t.split(" · ")[0] for t in titles] == [
        "Conform to template space — full field of view",
        "Conform to template space",
        "Skullstripping",
    ]
    full_fov, _, skull = routes["steps"]

    skull_body = _template(page, f"t-{skull}")
    assert "2 of 2 subjects" in skull_body
    assert 'href="#sub-a"' in skull_body and 'href="#sub-b"' in skull_body
    # Same chips as the subject pages' group headings.
    assert '<span class="gchip">ses-01</span>' in skull_body
    assert '<span class="gchip">subject-level</span>' in skull_body
    assert 'loading="lazy"' in skull_body

    fov_body = _template(page, f"t-{full_fov}")
    assert "1 of 2 subjects" in fov_body
    assert re.search(r'No figure for <a [^>]*href="#sub-b"', fov_body)


def test_t2w_steps_do_not_list_subjects_without_t2w(tmp_path):
    out = _dataset(tmp_path)
    (out / "sub-a" / "figures" / "sub-a_ses-01_desc-biascorrect_T2w.png").write_bytes(
        b"\x89PNG"
    )
    page = generate_dataset_report(out, ["a", "b"], {}, logger=LOG).read_text(
        encoding="utf-8"
    )
    routes = _routes(page)
    t2w = [s for s, t in routes["stepTitles"].items() if t.endswith("· T2w")]
    assert len(t2w) == 1
    assert "No figure for" not in _template(page, f"t-{t2w[0]}")


def test_render_takes_prebuilt_report_data(tmp_path):
    out = _dataset(tmp_path)
    data = [
        (
            f"sub-{s}",
            build_report_data(out / f"sub-{s}" / "figures", out / f"sub-{s}.html", {}),
        )
        for s in ("a", "b")
    ]
    page = render_dataset_report(data, "0.0.0")
    assert page.count('<template id="t-sub-') == 2


def test_subject_report_brand_links_to_the_dataset_page(tmp_path):
    out = _dataset(tmp_path)
    fig_dir = out / "sub-a" / "figures"

    generate_qc_report(fig_dir, out / "sub-a.html", {}, LOG)
    assert '<span class="brand">brainana</span>' in (out / "sub-a.html").read_text()

    generate_qc_report(
        fig_dir,
        out / "sub-a.html",
        {},
        LOG,
        dataset_report_href="all_subjects_report.html",
    )
    assert (
        '<a class="brand" href="all_subjects_report.html"'
        in (out / "sub-a.html").read_text()
    )
