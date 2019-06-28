import numpy as np
from experiments.dbscan import MyDBSCAN
from experiments.cluster_sdrs import get_clusters


def create_vec(vec_size,parts, num_parts=2):

    part_size = int(vec_size/parts)
    which_bit = np.random.randint(0,parts, num_parts)
    res = np.empty([0],dtype=np.int32)

    for f in range(parts):
        if f in which_bit:
            res = np.hstack([res, np.random.binomial(1, 0.7, part_size).astype(np.int32)])
        else:
            res = np.hstack([res,np.zeros(part_size)])
    return res


def create_vec_stack(vec_size,parts, num):

    result = [create_vec(vec_size, parts) for i in range(num)]

    return result


def similarity(x, y):
    return np.dot(x,y)


def check_test_data():
    np.set_printoptions(linewidth=200)

    # effectively creates 2 x 4 bit clusters within 80 bit vector i.e. sparsity = 0.1 and 2000 samples
    a = create_vec_stack(80,20,100)

    db = MyDBSCAN(a,3,2)

    #db = DBSCAN(eps=3, min_samples=5, metric=similarity).fit(a)
    print(db)

    clusters, idx = get_clusters(a,3,True,8)
    print(idx)
    print(f"Clusters: {len(list(set(idx)))}")
    for idx, cluster in enumerate(clusters):
        if cluster['id'] != -1:
            print(f"Cluster: {cluster['id']}")
            for h in cluster['history']:
                print(h['data'].astype(int))
        else:
            print(f"Cluster: {idx} merged with {cluster['merged_with']}")


def check_mnist():
    lines = None

    input_sdrs = []
    with open("out/output_mnist_sdrs.txt", "r") as f:
        lines = f.readlines()
        f.close()

    for line in lines:
        input_sdrs.append([float(v) for v in line.split(',')])

    clusters, idx = get_clusters(input_sdrs, 6, False,60)
    print(idx)
    print(f"Clusters: {len(list(set(idx)))}")
    for idx, cluster in enumerate(clusters):
        if cluster['id'] != -1:
            print(f"Cluster: {cluster['id']}")
            # for h in cluster['history']:
            #     print(h['data'].astype(int))
        else:
            print(f"Cluster: {idx} merged with {cluster['merged_with']}")


#check_test_data()
check_mnist()