from memlab.evaluate import alternate


def test_alternate_base_first_dedup():
    assert alternate(["a", "b", "c", "d"], ["x", "b", "y"]) == ["a", "x", "b", "c", "y", "d"]


def test_prefixes_by_model_and_channel():
    from memlab.__main__ import prefixes
    assert prefixes("intfloat/multilingual-e5-small", False) == ("query: ", "passage: ", None)
    q, d, n = prefixes("Qwen/Qwen3-Embedding-0.6B", True)
    assert q.startswith("Instruct: ") and q.endswith("\nQuery: ") and (d, n) == ("", 512)
    g = "google/embeddinggemma-2"
    assert prefixes(g, False) == ("task: code retrieval | query: ", "title: none | text: ", 512)
    assert prefixes(g, True) == ("task: search result | query: ", "title: none | text: ", 512)
    assert prefixes("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2", False) == ("", "", None)
