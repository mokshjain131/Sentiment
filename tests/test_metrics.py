from src.models.metrics import confusion_matrix, precision_recall_f1, accuracy

def test_metrics_basic():
    y_true = [0,0,1,1,2,2]
    y_pred = [0,1,1,1,2,0]
    cm = confusion_matrix(y_true, y_pred, 3)
    assert sum(sum(r) for r in cm) == len(y_true)
    prf = precision_recall_f1(cm)
    acc = accuracy(y_true, y_pred)
    assert 0 <= prf['macro_f1'] <= 1
    assert 0 <= acc <= 1
