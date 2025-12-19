from dk_politics_topics.analysis.controversial import DEFAULT_SEEDS, match_topics_to_areas


def test_match_topics_to_areas_keywords():
    topic_terms = {
        0: [{"term": "skat", "weight": 0.5}],
        1: [{"term": "klima", "weight": 0.5}],
        2: [{"term": "integration", "weight": 0.5}],
    }
    matches = match_topics_to_areas(topic_terms, DEFAULT_SEEDS)
    assert 0 in matches["taxation"]
    assert 1 in matches["climate"]
    assert any(t in matches["immigration"] for t in [2])
