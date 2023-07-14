from sklearn.cluster import DBSCAN, OPTICS
from clustering.relabel import relabel


def dbscan(position_matrix):
    print("DBSCAN clustering", flush=True)
    db = DBSCAN(eps=0.1)
    labels = relabel(db.fit_predict(position_matrix))
    print(f"found {max(labels)+1} clusters", flush=True)
    print(labels, flush=True)
    return labels


def optics(position_matrix):
    print("OPTICS clustering", flush=True)
    op = OPTICS()
    labels = relabel(op.fit_predict(position_matrix))
    print(f"found {max(labels)+1} clusters", flush=True)
    print(labels, flush=True)
    return labels
