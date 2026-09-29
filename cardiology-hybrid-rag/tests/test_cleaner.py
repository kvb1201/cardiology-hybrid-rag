"""
test_cleaner.py
---------------
Unit tests for src.preprocessing.cleaner.

Coverage
--------
* Whitespace normalisation (tabs, multiple spaces, trailing spaces)
* Blank-line collapsing
* PDF artefact removal (form feeds, DOI lines, copyright, boilerplate)
* Ligature and Unicode normalisation
* Soft-hyphen line-break repair
* Medical punctuation preservation
* Markdown heading preservation
* Empty / whitespace-only input edge cases
* clean_corpus() batch function
"""

from __future__ import annotations

import pytest

from src.preprocessing.cleaner import clean_document, clean_corpus
from tests.conftest import RAW_PDF_ARTEFACT_TEXT, EMPTY_TEXT, WHITESPACE_ONLY_TEXT


# ---------------------------------------------------------------------------
# Whitespace normalisation
# ---------------------------------------------------------------------------

class TestWhitespaceNormalisation:
    def test_tabs_replaced_by_single_space(self):
        text = "atrial\tfibrillation\tmanagement"
        result = clean_document(text)
        assert "\t" not in result
        assert "atrial fibrillation management" in result

    def test_multiple_spaces_collapsed(self):
        text = "heart   failure    treatment   guidelines"
        result = clean_document(text)
        assert "  " not in result

    def test_trailing_spaces_stripped(self):
        text = "Hypertension management   \nBlood pressure targets   "
        result = clean_document(text)
        for line in result.splitlines():
            assert line == line.rstrip(), f"Trailing space in: {repr(line)}"

    def test_leading_spaces_stripped_per_line(self):
        text = "   Atrial fibrillation\n   Heart failure"
        result = clean_document(text)
        for line in result.splitlines():
            if line:
                assert not line.startswith(" "), f"Leading space in: {repr(line)}"

    def test_excessive_blank_lines_collapsed(self):
        text = "First paragraph.\n\n\n\n\nSecond paragraph."
        result = clean_document(text)
        assert "\n\n\n" not in result

    def test_single_blank_line_preserved(self):
        text = "First paragraph.\n\nSecond paragraph."
        result = clean_document(text)
        assert "\n\n" in result


# ---------------------------------------------------------------------------
# PDF artefact removal
# ---------------------------------------------------------------------------

class TestPDFArtefactRemoval:
    def test_form_feed_removed(self):
        text = "Page one content.\x0cPage two content."
        result = clean_document(text)
        assert "\x0c" not in result
        assert "Page one content" in result
        assert "Page two content" in result

    def test_doi_line_removed(self):
        text = "Introduction\nDOI: 10.1161/CIRCULATIONAHA.123.000001\nMethods"
        result = clean_document(text)
        assert "10.1161" not in result
        assert "Introduction" in result
        assert "Methods" in result

    def test_doi_https_line_removed(self):
        text = "Results\nhttps://doi.org/10.1056/NEJMoa2304879\nConclusion"
        result = clean_document(text)
        assert "doi.org" not in result

    def test_copyright_line_removed(self):
        text = "Abstract\n© 2023 American Heart Association, Inc.\nBackground"
        result = clean_document(text)
        assert "American Heart Association" not in result
        assert "Abstract" in result

    def test_downloaded_from_line_removed(self):
        text = "Methods\nDownloaded from http://ahajournals.org by guest\nResults"
        result = clean_document(text)
        assert "Downloaded from" not in result
        assert "Methods" in result

    def test_received_accepted_lines_removed(self):
        text = "Study Design\nReceived: October 1, 2022\nAccepted: January 15, 2023\nPatients"
        result = clean_document(text)
        assert "Received:" not in result
        assert "Accepted:" not in result
        assert "Study Design" in result

    def test_standalone_page_number_removed(self):
        text = "Conclusion paragraph.\n\n42\n\nNext section."
        result = clean_document(text)
        lines = [l for l in result.splitlines() if l.strip()]
        assert "42" not in lines, "Standalone page number should be removed"

    def test_full_pdf_artefact_text(self):
        """End-to-end: the RAW_PDF_ARTEFACT_TEXT fixture should yield clean output."""
        result = clean_document(RAW_PDF_ARTEFACT_TEXT)
        assert "DOI:" not in result
        assert "Downloaded from" not in result
        assert "Received" not in result
        assert "\x0c" not in result
        # Medical content should survive
        assert "Hypertension" in result
        assert "cardiovascular" in result


# ---------------------------------------------------------------------------
# Ligature and Unicode normalisation
# ---------------------------------------------------------------------------

class TestLigatureNormalisation:
    def test_ff_ligature_decoded(self):
        result = clean_document("heart failure a\ufb00ects millions")
        assert "\ufb00" not in result
        assert "affects" in result

    def test_fi_ligature_decoded(self):
        result = clean_document("modi\ufb01able risk factors")
        assert "\ufb01" not in result
        assert "modifiable" in result

    def test_fl_ligature_decoded(self):
        result = clean_document("in\ufb02ammation markers")
        assert "\ufb02" not in result
        assert "inflammation" in result

    def test_en_dash_becomes_hyphen(self):
        result = clean_document("pages 1234\u20131256")
        assert "\u2013" not in result
        assert "-" in result

    def test_non_breaking_space_normalised(self):
        result = clean_document("10\u00a0mg/kg")
        assert "\u00a0" not in result
        assert "10 mg/kg" in result

    def test_smart_quotes_normalised(self):
        result = clean_document("\u201cAtrial fibrillation\u201d")
        assert "\u201c" not in result
        assert "\u201d" not in result
        assert '"Atrial fibrillation"' in result

    def test_plus_minus_preserved(self):
        result = clean_document("age 72 ± 5 years")
        assert "±" in result

    def test_greater_equal_converted(self):
        result = clean_document("CHA\u2082DS\u2082-VASc score \u22652")
        assert ">=" in result


# ---------------------------------------------------------------------------
# Soft-hyphen repair
# ---------------------------------------------------------------------------

class TestSoftHyphenRepair:
    def test_hyphenated_word_rejoined(self):
        text = "cardio-\nvascular disease"
        result = clean_document(text)
        assert "cardiovascular" in result

    def test_hyphenated_word_rejoined_with_spaces(self):
        text = "anti-  \nhypertensive therapy"
        result = clean_document(text)
        assert "antihypertensive" in result

    def test_intentional_hyphen_at_end_of_sentence_preserved(self):
        # A hyphen before an uppercase letter (new sentence) should NOT be joined
        text = "dose-response curve. High-dose therapy was used."
        result = clean_document(text)
        # The compound hyphen in "dose-response" must survive cleaning
        assert "dose-response" in result


# ---------------------------------------------------------------------------
# Medical punctuation preservation
# ---------------------------------------------------------------------------

class TestMedicalPunctuationPreservation:
    def test_hyphen_in_compound_terms(self):
        terms = [
            "ST-elevation", "dose-response", "beta-blocker",
            "non-dihydropyridine", "rate-limiting",
        ]
        for term in terms:
            result = clean_document(f"The {term} was observed.")
            assert term in result, f"Compound term lost: {term}"

    def test_slash_in_dose_notation(self):
        result = clean_document("Dose: 10 mg/kg/day administered IV.")
        assert "mg/kg/day" in result

    def test_colon_in_lab_values(self):
        result = clean_document("Electrolytes: Na: 138 mEq/L, K: 4.2 mEq/L")
        assert "Na: 138" in result
        assert "K: 4.2" in result

    def test_parentheses_for_abbreviations(self):
        result = clean_document("Atrial fibrillation (AF) is the most common arrhythmia.")
        assert "(AF)" in result

    def test_square_brackets_for_citations(self):
        result = clean_document("As previously reported [1,2,3], the incidence was high.")
        assert "[1,2,3]" in result

    def test_percentage_sign_preserved(self):
        result = clean_document("LVEF was 35% (reduced).")
        assert "35%" in result

    def test_blood_pressure_notation(self):
        result = clean_document("Target BP: <130/80 mmHg per ACC/AHA guidelines.")
        assert "130/80" in result
        assert "ACC/AHA" in result

    def test_inr_range_preserved(self):
        result = clean_document("Warfarin target INR: 2.0-3.0 for AF.")
        assert "2.0-3.0" in result


# ---------------------------------------------------------------------------
# Markdown heading preservation
# ---------------------------------------------------------------------------

class TestHeadingPreservation:
    def test_h1_heading_preserved(self):
        result = clean_document("# Atrial Fibrillation\n\nContent here.")
        assert "# Atrial Fibrillation" in result

    def test_h2_heading_preserved(self):
        result = clean_document("## Pathophysiology\n\nContent here.")
        assert "## Pathophysiology" in result

    def test_h3_heading_preserved(self):
        result = clean_document("### Diagnosis Criteria\n\nContent here.")
        assert "### Diagnosis Criteria" in result

    def test_multiple_headings_preserved(self):
        text = "# Title\n\n## Section 1\n\nContent.\n\n## Section 2\n\nMore."
        result = clean_document(text)
        assert "# Title" in result
        assert "## Section 1" in result
        assert "## Section 2" in result


# ---------------------------------------------------------------------------
# Edge cases: empty / invalid input
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_empty_string_returns_empty(self):
        assert clean_document("") == ""

    def test_whitespace_only_returns_empty(self):
        assert clean_document(EMPTY_TEXT) == ""
        assert clean_document(WHITESPACE_ONLY_TEXT) == ""

    def test_newlines_only_returns_empty(self):
        assert clean_document("\n\n\n\n") == ""

    def test_single_word_preserved(self):
        result = clean_document("Cardiology")
        assert result == "Cardiology"

    def test_unicode_content_preserved(self):
        # Greek letters in drug names / physiology
        result = clean_document("Alpha-1 adrenoceptor blockade reduces BP.")
        assert "Alpha-1" in result

    def test_numeric_lab_values_preserved(self):
        result = clean_document("Troponin I: 0.45 ng/mL (ref: <0.04 ng/mL)")
        assert "0.45" in result
        assert "0.04" in result


# ---------------------------------------------------------------------------
# clean_corpus()
# ---------------------------------------------------------------------------

class TestCleanCorpus:
    def test_returns_same_length(self):
        docs = ["Doc one.", "Doc two.", "Doc three."]
        result = clean_corpus(docs)
        assert len(result) == len(docs)

    def test_empty_doc_in_corpus_preserved_as_empty(self):
        docs = ["Real content here.", "", "More content."]
        result = clean_corpus(docs)
        assert result[1] == ""

    def test_all_docs_cleaned(self):
        docs = [
            "heart   failure",
            "atrial\tfibrillation",
            "DOI: 10.1000/xyz\nHypertension",
        ]
        result = clean_corpus(docs)
        assert "  " not in result[0]
        assert "\t" not in result[1]
        assert "DOI:" not in result[2]
        assert "Hypertension" in result[2]

    def test_empty_list_returns_empty_list(self):
        assert clean_corpus([]) == []
