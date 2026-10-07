"""Release metadata: the version, the title, the authors and the DOIs agree across CITATION.cff, .zenodo.json,
pyproject.toml, BUILD_INFO.json, CHANGELOG.md and README.md."""
import json
import re
from pathlib import Path

import pebrnet as pn

ROOT = Path(__file__).resolve().parents[1]
ZENODO_KEYS = {"title", "upload_type", "description", "creators", "access_right", "license", "keywords",
               "related_identifiers", "communities", "grants", "notes", "version", "publication_date", "contributors",
               "references"}
CFF = (ROOT / "CITATION.cff").read_text(encoding="utf-8")
ZEN = json.loads((ROOT / ".zenodo.json").read_text(encoding="utf-8"))
README = " ".join((ROOT / "README.md").read_text(encoding="utf-8").split())
PYPROJECT = (ROOT / "pyproject.toml").read_text(encoding="utf-8")


def _cff(key: str) -> str:
    return re.search(r"^%s: (.+)$" % re.escape(key), CFF, re.M).group(1).strip().strip('"')


def test_versions_agree():
    v = pn.__version__
    assert re.fullmatch(r"\d+\.\d+\.\d+", v)
    assert re.search(r'^version = "%s"$' % re.escape(v), PYPROJECT, re.M)
    assert json.loads((ROOT / "BUILD_INFO.json").read_text(encoding="utf-8"))["version"] == v
    assert _cff("version") == "V" + v                     # the release tag and the Zenodo version label
    assert re.search(r"^## V(\S+)", (ROOT / "CHANGELOG.md").read_text(encoding="utf-8"), re.M).group(1) == v
    assert "(Version V%s)" % v in README


def test_title_is_pebr_net_and_the_article_title():
    title = _cff("title")
    article = re.search(r'^  - type: article\n    title: "(.+)"$', CFF, re.M).group(1)
    assert title == "PEBR-Net: " + article
    assert ZEN["title"] == title
    assert re.search(r'^description = "%s"$' % re.escape(title), PYPROJECT, re.M)
    assert "**%s**" % title in README and "*%s*" % article in README


def test_authors_and_affiliations_agree():
    cff = re.findall(r'^  - family-names: (\S+)\n    given-names: (\S+)\n    affiliation: "(.+)"$', CFF, re.M)
    zen = [(c["name"], c["affiliation"]) for c in ZEN["creators"]]
    assert [("%s, %s" % (f, g), a) for f, g, a in cff] == zen
    assert [n for n, _ in zen] == ["Ye, Yi", "Zhang, Chao", "Yu, Nian"]


def test_zenodo_metadata():
    assert set(ZEN) <= ZENODO_KEYS
    assert ZEN["upload_type"] == "software" and ZEN["access_right"] == "open" and ZEN["license"] == "mit"


def test_dois_agree():
    version_doi = _cff("doi")
    ids = re.findall(r'^  - type: doi\n    value: (\S+)\n    description: "(.+)"$', CFF, re.M)
    concept = ids[1][0]
    assert ids[0][0] == version_doi
    assert all(re.fullmatch(r"10\.5281/zenodo\.\d+", d) for d in (version_doi, concept))
    assert "https://zenodo.org/badge/DOI/%s.svg" % concept in README
    assert "https://doi.org/%s" % version_doi in README and "https://doi.org/%s" % concept in README
    dataset = re.search(r"^    doi: (\S+)$", CFF, re.M).group(1)
    assert [r["identifier"] for r in ZEN["related_identifiers"] if r["resource_type"] == "dataset"] == [dataset]
    doi_url = json.loads((ROOT / "data" / "sources.json").read_text(encoding="utf-8"))["dataset"]["doi"]
    assert doi_url == "https://doi.org/" + dataset and doi_url in README
