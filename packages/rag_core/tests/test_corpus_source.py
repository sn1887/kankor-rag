from __future__ import annotations

import json

from rag_core.impl.corpus_pages import CanonicalPageCorpusSource, StructureIndexBuilder
from rag_core.impl.corpus_hf_dataset import HFDatasetCorpusSource


def test_local_jsonl_corpus_source_streams_documents(tmp_path) -> None:
    path = tmp_path / "corpus.jsonl"
    path.write_text(
        "\n".join(
            [
                "",
                json.dumps({"id": "doc-1", "text": "hello", "metadata": {"language": "en"}}),
                json.dumps({"id": "doc-2", "text": "world", "metadata": {"language": "fa"}}),
                "",
            ]
        ),
        encoding="utf-8",
    )
    source = HFDatasetCorpusSource(local_path=str(path))
    documents = list(source.load_documents())
    assert [item.id for item in documents] == ["doc-1", "doc-2"]


def test_canonical_page_corpus_source_builds_page_documents(tmp_path) -> None:
    books_dir = tmp_path / "books" / "G10-Dr-Physics"
    books_dir.mkdir(parents=True)
    page_row = {
        "page_number": 7,
        "pdf_page_number": 9,
        "chapter_title": "حرکت",
        "topic_title": "سرعت",
        "headers": ["درس سرعت"],
        "raw_text": "سرعت مقدار مسافت طی شده در واحد زمان است.",
        "equations": [{"raw": "v=d/t", "description": "فرمول سرعت"}],
        "tables": [],
        "figures": [],
        "quality_flags": ["none"],
        "extraction_confidence": "high",
        "source_id": "G10-Dr-Physics",
        "title": "G10-Dr-Physics",
        "grade_band": "10",
        "language": "fa",
        "subject_category": "natural_science",
        "subject": "physics",
        "source_type": "explanation",
        "retrieval_weight": 1.0,
    }
    (books_dir / "pages.jsonl").write_text(json.dumps(page_row, ensure_ascii=False) + "\n", encoding="utf-8")

    source = CanonicalPageCorpusSource(corpus_root=tmp_path, corpus_version="test@1")
    documents = list(source.load_documents())

    assert len(documents) == 1
    assert documents[0].id == "G10-Dr-Physics:page:7"
    assert documents[0].metadata["page"] == 7
    assert "حرکت" in documents[0].text
    assert "فرمول سرعت" in documents[0].text


def test_canonical_page_corpus_source_projects_frontmatter_fields_without_overwriting_raw_metadata(tmp_path) -> None:
    books_dir = tmp_path / "books" / "G10-Dr-Physics"
    books_dir.mkdir(parents=True)
    rows = [
        {
            "page_number": 7,
            "pdf_page_number": 9,
            "chapter_title": None,
            "topic_title": None,
            "headers": ["جمهوری اسلامی افغانستان", "درس سرعت"],
            "raw_text": "سرعت مقدار مسافت طی شده در واحد زمان است.",
            "equations": [],
            "tables": [],
            "figures": [],
            "source_id": "G10-Dr-Physics",
            "title": "G10-Dr-Physics",
            "grade_band": "10",
            "language": "fa",
            "subject_category": "natural_science",
            "subject": "physics",
            "source_type": "explanation",
        }
    ]
    (books_dir / "pages.jsonl").write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
        encoding="utf-8",
    )
    toc_path = tmp_path / "frontmatter.jsonl"
    toc_path.write_text(
        json.dumps(
            {
                "normalized_source_id": "G10-Dr-Physics",
                "normalized_chapters": [
                    {
                        "chapter_index": 2,
                        "chapter_number": "2",
                        "chapter_title": "حرکت",
                        "start_page": 7,
                        "end_page": 15,
                        "topics": [
                            {
                                "topic_index": 1,
                                "title": "سرعت",
                                "start_page": 7,
                                "end_page": 8,
                            }
                        ],
                    }
                ],
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    source = CanonicalPageCorpusSource(
        corpus_root=tmp_path,
        corpus_version="test@1",
        frontmatter_toc_path=toc_path,
    )
    documents = list(source.load_documents())

    assert len(documents) == 1
    assert documents[0].metadata["chapter_title"] == ""
    assert documents[0].metadata["topic_title"] == ""
    assert documents[0].metadata["resolved_chapter_title"] == "حرکت"
    assert documents[0].metadata["resolved_topic_title"] == "سرعت"
    assert documents[0].metadata["resolved_chapter_number"] == "2"
    assert documents[0].metadata["structure_source"] == "frontmatter"
    assert "جمهوری اسلامی افغانستان" not in documents[0].text
    assert "درس سرعت" in documents[0].text


def test_canonical_page_corpus_source_falls_back_to_folder_identity(tmp_path) -> None:
    books_dir = tmp_path / "books" / "G12-Dr-Geography"
    books_dir.mkdir(parents=True)
    page_row = {
        "page_number": 1,
        "pdf_page_number": 1,
        "chapter_title": "جغرافیه",
        "topic_title": "صنف ۱۲",
        "raw_text": "افغانستان در قلب آسیا موقعیت دارد.",
        "headers": [],
        "equations": [],
        "tables": [],
        "figures": [],
    }
    (books_dir / "pages.jsonl").write_text(json.dumps(page_row, ensure_ascii=False) + "\n", encoding="utf-8")

    source = CanonicalPageCorpusSource(corpus_root=tmp_path, corpus_version="test@1")
    documents = list(source.load_documents())

    assert len(documents) == 1
    assert documents[0].id == "G12-Dr-Geography:page:1"
    assert documents[0].metadata["source_id"] == "G12-Dr-Geography"
    assert documents[0].metadata["title"] == "G12-Dr-Geography"


def test_canonical_page_corpus_source_adds_trusted_glossary_descriptor(tmp_path) -> None:
    books_dir = tmp_path / "books" / "G11-Dr-Dari"
    books_dir.mkdir(parents=True)
    page_row = {
        "page_number": 162,
        "pdf_page_number": 162,
        "headers": ["أ", "الف"],
        "raw_text": "واژه نامه (لغت نامه) آفرینش به وجود آوردن، انشا، ابداع، آفریدن",
        "source_id": "G11-Dr-Dari",
        "title": "G11-Dr-Dari",
        "grade_band": "11",
        "language": "fa",
        "subject_category": "language",
        "subject": "dari",
        "source_type": "explanation",
    }
    (books_dir / "pages.jsonl").write_text(json.dumps(page_row, ensure_ascii=False) + "\n", encoding="utf-8")
    toc_path = tmp_path / "frontmatter.jsonl"
    toc_path.write_text(
        json.dumps(
            {
                "normalized_source_id": "G11-Dr-Dari",
                "normalized_structure_mode": "topics_only",
                "normalized_topics": [
                    {
                        "topic_index": 29,
                        "title": "واژه نامه (لغت نامه)",
                        "start_page": 155,
                        "end_page": 170,
                    }
                ],
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    source = CanonicalPageCorpusSource(
        corpus_root=tmp_path,
        corpus_version="test@1",
        frontmatter_toc_path=toc_path,
    )
    document = next(iter(source.load_documents()))

    assert document.metadata["is_glossary"] is True
    assert document.metadata["glossary_kind"] == "word_meanings"
    assert document.metadata["glossary_source"] == "frontmatter"
    assert "glossary section" in document.text


def test_canonical_page_corpus_source_keeps_manual_glossary_metadata_without_prefix_when_page_has_no_glossary_evidence(tmp_path) -> None:
    books_dir = tmp_path / "books" / "G12-Dr-Geography"
    books_dir.mkdir(parents=True)
    page_row = {
        "page_number": 207,
        "pdf_page_number": 207,
        "headers": [],
        "raw_text": "تولید گاز C.F.C از دستگاههای سرد کننده در فضا نفوذ مینماید.",
        "source_id": "G12-Dr-Geography",
        "title": "G12-Dr-Geography",
        "grade_band": "12",
        "language": "fa",
        "subject_category": "social_science",
        "subject": "geography",
        "source_type": "explanation",
    }
    (books_dir / "pages.jsonl").write_text(json.dumps(page_row, ensure_ascii=False) + "\n", encoding="utf-8")
    toc_path = tmp_path / "frontmatter.jsonl"
    toc_path.write_text(
        json.dumps(
            {
                "normalized_source_id": "G12-Dr-Geography",
                "normalized_chapters": [
                    {
                        "chapter_title": "واژه نامه",
                        "start_page": 207,
                        "end_page": 209,
                        "issues": ["manually_added_glossary_chapter"],
                        "topics": [
                            {
                                "title": "اصطلاحات",
                                "start_page": 207,
                                "end_page": 209,
                                "issues": ["manually_added_in_notebook"],
                            }
                        ],
                    }
                ],
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    source = CanonicalPageCorpusSource(
        corpus_root=tmp_path,
        corpus_version="test@1",
        frontmatter_toc_path=toc_path,
    )
    document = next(iter(source.load_documents()))

    assert document.metadata["is_glossary"] is True
    assert document.metadata["glossary_kind"] == "terminology"
    assert document.metadata["glossary_source"] == "frontmatter_manual"
    assert "glossary section" not in document.text


def test_structure_index_builder_creates_chapter_and_topic_documents(tmp_path) -> None:
    books_dir = tmp_path / "books" / "G10-Dr-Physics"
    books_dir.mkdir(parents=True)
    rows = [
        {
            "page_number": 7,
            "chapter_title": "حرکت",
            "topic_title": "سرعت",
            "raw_text": "سرعت",
            "headers": [],
            "source_id": "G10-Dr-Physics",
            "title": "G10-Dr-Physics",
            "grade_band": "10",
            "language": "fa",
            "subject_category": "natural_science",
            "subject": "physics",
            "source_type": "explanation",
        },
        {
            "page_number": 8,
            "chapter_title": "حرکت",
            "topic_title": "سرعت",
            "raw_text": "سرعت متوسط",
            "headers": [],
            "source_id": "G10-Dr-Physics",
            "title": "G10-Dr-Physics",
            "grade_band": "10",
            "language": "fa",
            "subject_category": "natural_science",
            "subject": "physics",
            "source_type": "explanation",
        },
    ]
    (books_dir / "pages.jsonl").write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
        encoding="utf-8",
    )
    toc_path = tmp_path / "frontmatter.jsonl"
    toc_path.write_text(
        json.dumps(
            {
                "normalized_source_id": "G10-Dr-Physics",
                "normalized_chapters": [
                    {
                        "chapter_index": 2,
                        "chapter_number": "2",
                        "chapter_title": "حرکت",
                        "start_page": 7,
                        "end_page": 15,
                    }
                ],
                "normalized_topics": [
                    {
                        "topic_index": 1,
                        "title": "سرعت",
                        "start_page": 7,
                        "end_page": 8,
                        "source_chapter_title": "حرکت",
                        "source_chapter_number": "2",
                    }
                ],
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    builder = StructureIndexBuilder(
        corpus_root=tmp_path,
        corpus_version="test@1",
        frontmatter_toc_path=toc_path,
    )
    chapter_docs = builder.build_chapter_documents()
    topic_docs = builder.build_topic_documents()

    assert chapter_docs and chapter_docs[0].metadata["chapter_title"] == "حرکت"
    assert topic_docs and topic_docs[0].metadata["topic_title"] == "سرعت"


def test_structure_index_builder_builds_nested_topic_documents_with_glossary_metadata(tmp_path) -> None:
    books_dir = tmp_path / "books" / "G12-Dr-Geography"
    books_dir.mkdir(parents=True)
    rows = [
        {
            "page_number": 207,
            "raw_text": "اصطلاحات",
            "headers": [],
            "source_id": "G12-Dr-Geography",
            "title": "G12-Dr-Geography",
            "grade_band": "12",
            "language": "fa",
            "subject_category": "social_science",
            "subject": "geography",
            "source_type": "explanation",
        }
    ]
    (books_dir / "pages.jsonl").write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
        encoding="utf-8",
    )
    toc_path = tmp_path / "frontmatter.jsonl"
    toc_path.write_text(
        json.dumps(
            {
                "normalized_source_id": "G12-Dr-Geography",
                "normalized_chapters": [
                    {
                        "chapter_title": "واژه نامه",
                        "start_page": 207,
                        "end_page": 209,
                        "issues": ["manually_added_glossary_chapter"],
                        "topics": [
                            {
                                "title": "اصطلاحات",
                                "start_page": 207,
                                "end_page": 209,
                                "issues": ["manually_added_in_notebook"],
                            }
                        ],
                    }
                ],
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    builder = StructureIndexBuilder(
        corpus_root=tmp_path,
        corpus_version="test@1",
        frontmatter_toc_path=toc_path,
    )
    chapter_docs = builder.build_chapter_documents()
    topic_docs = builder.build_topic_documents()

    assert chapter_docs[0].metadata["is_glossary"] is True
    assert chapter_docs[0].metadata["glossary_kind"] == "word_meanings"
    assert topic_docs[0].metadata["is_glossary"] is True
    assert topic_docs[0].metadata["glossary_kind"] == "terminology"
    assert topic_docs[0].metadata["structure_source"] == "frontmatter_manual"
    assert "glossary" in topic_docs[0].metadata["normalized_lookup_text"]
