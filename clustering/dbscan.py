from sklearn.cluster import DBSCAN, OPTICS


def dbscan(position_matrix):
    print("DBSCAN clustering", flush=True)
    db = DBSCAN(eps=0.1)
    labels = db.fit_predict(position_matrix)
    print(f"found {max(labels)+1} clusters", flush=True)
    print(labels, flush=True)
    return labels


def optics(position_matrix):
    print("OPTICS clustering", flush=True)
    op = OPTICS()
    labels = op.fit_predict(position_matrix)
    print(f"found {max(labels)+1} clusters", flush=True)
    print(labels, flush=True)
    return labels
