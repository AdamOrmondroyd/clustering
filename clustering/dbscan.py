from sklearn.cluster import DBSCAN, OPTICS


def dbscan(position_matrix):
    print("DBSCAN clustering", flush=True)
    db = DBSCAN(eps=0.1).fit(position_matrix)
    print(f"found {max(db.labels_)+1} clusters", flush=True)
    print(db.labels_, flush=True)
    return db.labels_


def optics(position_matrix):
    print("OPTICS clustering", flush=True)
    op = OPTICS().fit(position_matrix)
    print(f"found {max(op.labels_)+1} clusters", flush=True)
    print(op.labels_, flush=True)
    return op.labels_
