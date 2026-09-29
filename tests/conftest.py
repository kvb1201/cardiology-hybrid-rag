"""
conftest.py
-----------
Shared pytest fixtures for the cardiology-hybrid-rag test suite.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import pytest

# Ensure the project root is on sys.path so imports work when pytest is
# invoked from any working directory.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


# ---------------------------------------------------------------------------
# Text fixtures
# ---------------------------------------------------------------------------

CLEAN_CARDIOLOGY_TEXT = """\
# Atrial Fibrillation: Diagnosis and Management

Atrial fibrillation (AF) is the most common sustained cardiac arrhythmia,
affecting approximately 33 million individuals worldwide.

## Pathophysiology

The hallmark of AF is chaotic, rapid atrial electrical activity resulting in
irregular ventricular response. Heart rate typically ranges from 100-160 bpm
during untreated AF.

## Diagnosis

Diagnosis is confirmed by 12-lead ECG showing irregularly irregular rhythm
with absence of distinct P waves. The QRS complex is usually narrow (<120 ms)
unless aberrant conduction exists.

Key diagnostic criteria:
- Irregular RR intervals with no distinct P waves
- Atrial rate 350-600 impulses per minute
- Ventricular rate typically 100-160 bpm (untreated)

## Treatment

Rate control is achieved with beta-blockers (metoprolol 25-200 mg/day) or
non-dihydropyridine calcium channel blockers (diltiazem, verapamil).
Anticoagulation with warfarin (target INR 2.0-3.0) or direct oral
anticoagulants (DOACs: apixaban 5 mg BID, rivaroxaban 20 mg OD) reduces the
risk of stroke by 60-70%.

The CHA2DS2-VASc score guides anticoagulation decisions:
- Score 0 (male) or 1 (female): anticoagulation not recommended
- Score >= 2: anticoagulation recommended
"""

RAW_PDF_ARTEFACT_TEXT = """\
Circulation. 2023;147:1234\u20131256
DOI: 10.1161/CIRCULATIONAHA.123.000001
\xa9 2023 American Heart Association, Inc.
Downloaded from http://ahajournals.org by guest on 2023-09-01
Received October 1, 2022; accepted January 15, 2023

\x0c\x0c

        1

Hypertension and Cardiovascular Risk

Hypertension a\ufb00ects more than 1.28 billion adults world\ufb01wide. It is the
leading modi\ufb01able risk factor for cardiovascular disease (CVD). Blood
pressure targets vary by guideline: ACC/AHA recommend <130/80 mmHg for most
patients, while ESC guidelines suggest <140/90 mmHg as an initial target.

        2

First-line antihypertensive agents include thiazide diuretics, ACE
inhibitors, ARBs, and calcium channel blockers. Combination therapy is
o\ufb00en required to achieve blood pressure goals.

"""

EMPTY_TEXT = "   \n  \t  \n   "
WHITESPACE_ONLY_TEXT = "\n\n\n"


# ---------------------------------------------------------------------------
# File fixtures (real files on disk, cleaned up after each test)
# ---------------------------------------------------------------------------

@pytest.fixture()
def tmp_dir(tmp_path: Path) -> Path:
    """Return a temporary directory (pytest built-in tmp_path)."""
    return tmp_path


@pytest.fixture()
def sample_txt_file(tmp_path: Path) -> Path:
    """Write a plain-text cardiology document to a temp file."""
    p = tmp_path / "sample_af.txt"
    p.write_text(CLEAN_CARDIOLOGY_TEXT, encoding="utf-8")
    return p


@pytest.fixture()
def sample_md_file(tmp_path: Path) -> Path:
    """Write a Markdown cardiology document to a temp file."""
    p = tmp_path / "sample_hypertension.md"
    p.write_text(
        "# Hypertension Management\n\n"
        "Blood pressure (BP) targets: <130/80 mmHg (ACC/AHA) or "
        "<140/90 mmHg (ESC).\n\n"
        "First-line agents: thiazide diuretics, ACE inhibitors (e.g. "
        "enalapril 5-40 mg/day), ARBs (losartan 25-100 mg/day), and "
        "calcium channel blockers (amlodipine 2.5-10 mg/day).\n\n"
        "## Secondary Hypertension\n\n"
        "Renovascular hypertension, primary aldosteronism (Conn syndrome), "
        "phaeochromocytoma, and Cushing syndrome are important secondary "
        "causes to exclude.\n",
        encoding="utf-8",
    )
    return p


@pytest.fixture()
def sample_html_file(tmp_path: Path) -> Path:
    """Write a minimal HTML cardiology document to a temp file."""
    p = tmp_path / "sample_hf.html"
    p.write_text(
        """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>Heart Failure Overview</title>
</head>
<body>
  <nav>Site navigation (should be stripped)</nav>
  <h1>Heart Failure Overview</h1>
  <p>Heart failure (HF) affects approximately 64 million people globally.
  It is characterised by the heart's inability to pump sufficient blood
  to meet the body's metabolic demands.</p>
  <h2>Classification</h2>
  <p>HF is classified by left ventricular ejection fraction (LVEF):
  HFrEF (LVEF &lt;40%), HFmrEF (LVEF 40-49%), and HFpEF (LVEF &ge;50%).</p>
  <h2>Treatment</h2>
  <p>Guideline-directed medical therapy (GDMT) for HFrEF includes:
  ACE inhibitors or ARBs, beta-blockers (carvedilol, metoprolol succinate),
  mineralocorticoid receptor antagonists (spironolactone 25-50 mg/day),
  and SGLT2 inhibitors (dapagliflozin 10 mg/day).</p>
  <footer>Copyright 2023</footer>
  <script>alert('ignored')</script>
</body>
</html>""",
        encoding="utf-8",
    )
    return p


@pytest.fixture()
def sample_raw_dir(tmp_path: Path, sample_txt_file: Path,
                   sample_md_file: Path, sample_html_file: Path) -> Path:
    """Return a directory containing all three sample files."""
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    for src in [sample_txt_file, sample_md_file, sample_html_file]:
        dest = raw_dir / src.name
        dest.write_bytes(src.read_bytes())
    return raw_dir
