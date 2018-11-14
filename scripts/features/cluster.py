from sklearn.cluster import KMeans


def elbow(train, obj_id):
    data = train.query("object_id == @obj_id").mjd.values.reshape([-1, 1])
    kms = [KMeans(n_clusters=i, n_jobs=-1).fit(data) for i in range(2, 6)]
    inertias = [km.inertia_ for km in kms]
    diff1 = inertias[0] - inertias[1]
    diff2 = inertias[1] - inertias[2]
    diff3 = inertias[2] - inertias[3]
    if diff1 / diff2 > diff2 / diff3:
        return kms[1].predict(data)
    else:
        return kms[2].predict(data)


def add_cluster(df):
    dfs = []
    for objid in df.object_id.unique():
        dfs += elbow(df, objid).tolist()
    df["cluster"] = dfs
    return df
