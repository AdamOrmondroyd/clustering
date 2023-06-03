from sklearn.cluster import DBSCAN

def dbscan(position_matrix):
    db = DBSCAN(eps=0.1).fit(position_matrix)
    print(f"found {max(db.labels_)+1} clusters")
    return db.labels_
