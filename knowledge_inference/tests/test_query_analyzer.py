from knowledge_inference.query_analyzer import analyze_query


def test_query_analyzer_flags():
    q = "Compare across videos: when does Pyke appear and what is shown in the frame?"
    intent = analyze_query(q)
    assert intent.is_cross_video is True
    assert intent.is_temporal is True
    assert intent.is_visual_detail is True
    assert "pyke" in intent.entity_focus_terms
