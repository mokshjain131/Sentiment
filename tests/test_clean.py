from src.processing.clean import clean_text

def test_basic_clean():
    dirty = "<p>(Reuters) - Apple’s   shares </p>"
    out = clean_text(dirty)
    assert "Reuters" not in out
    assert "Apple" in out
    assert "  " not in out
