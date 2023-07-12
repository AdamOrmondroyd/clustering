from sklearn.cluster import DBSCAN, OPTICS


def dbscan(position_matrix):
    print("a")
    db = DBSCAN(eps=0.5).fit(position_matrix)
    print(f"found {max(db.labels_)+1} clusters")
    print("b")
    print(db.labels_)
    return db.labels_


def optics(position_matrix):
    print("a")
    op = OPTICS().fit(position_matrix)
    print(f"found {max(op.labels_)+1} clusters")
    print("b")
    print(op.labels_)
    return op.labels_
