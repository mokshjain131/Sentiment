from src.labeling.lexicon import score_lexicon, label_from_score


def test_lexicon_positive():
    s = score_lexicon("record profit growth and upgrade")
    assert s > 0
    assert label_from_score(s) == "positive"


def test_lexicon_negative():
    s = score_lexicon("loss decline lawsuit and probe")
    assert s < 0
    assert label_from_score(s) == "negative"
