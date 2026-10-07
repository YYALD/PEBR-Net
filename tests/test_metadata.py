"""Release metadata: the version and the citation records agree."""
import json
import re
from pathlib import Path

import pebrnet as pn

ROOT = Path(__file__).resolve().parents[1]
ZENODO_KEYS = {"title", "upload_type", "description", "creators", "access_right", "license", "keywords",
               "related_identifiers", "communities", "grants", "notes", "version", "publication_date", "contributors",
               "references"}


def test_versions_agree():
    v = pn.__version__
    assert re.fullmatch(r"\d+\.\d+\.\d+", v)
    assert re.search(r'^version = "%s"$' % re.escape(v), (ROOT / "pyproject.toml").read_text(encoding="utf-8"), re.M)
    assert re.search(r"^version: %s$" % re.escape(v), (ROOT / "CITATION.cff").read_text(encoding="utf-8"), re.M)
    assert re.search(r"^## (\S+)", (ROOT / "CHANGELOG.md").read_text(encoding="utf-8"), re.M).group(1) == v
    assert json.loads((ROOT / "BUILD_INFO.json").read_text(encoding="utf-8"))["version"] == v
    assert "(Version %s)" % v in (ROOT / "README.md").read_text(encoding="utf-8")


def test_zenodo_metadata():
    z = json.loads((ROOT / ".zenodo.json").read_text(encoding="utf-8"))
    assert set(z) <= ZENODO_KEYS
    assert z["upload_type"] == "software" and z["access_right"] == "open" and z["license"] == "mit"
    assert [c["name"] for c in z["creators"]] == ["Ye, Yi", "Zhang, Chao", "Yu, Nian"]
    assert 'title: "%s"' % z["title"] in (ROOT / "CITATION.cff").read_text(encoding="utf-8")


def test_software_doi_is_the_same_everywhere():
    cff = (ROOT / "CITATION.cff").read_text(encoding="utf-8")
    doi = re.search(r"^doi: (10\.5281/zenodo\.\d+)$", cff, re.M).group(1)
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert "https://zenodo.org/badge/DOI/%s.svg" % doi in readme and "https://doi.org/%s" % doi in readme

