import sys
import numpy as np
from tqdm import tqdm


def get_clusters(sdr_inputs, overlap_dist = 20, store_history = True, max_cluster_bits = 40, max_clusters = 10):
    clusters = []
    idx = [None] * len(sdr_inputs)

    for f in tqdm(range(len(sdr_inputs))):

        current_sdr = sdr_inputs[f]

        # find which cluster is closest
        # this uses the union'd sdr for each cluster
        # effectively making a very quick search across all data
        matched_cluster, overlap = find_closest_sdr(clusters, current_sdr)

        # Overlap is a similarity measure
        # OVERLAP_DIST - is used to control clustering
        if overlap > overlap_dist:
            matched_cluster['count'] += 1
            idx[f] = matched_cluster['id']

            if store_history:
                found = sdr_in(matched_cluster['history'], current_sdr)
                if not found:
                    matched_cluster['history'].append({'data':current_sdr})
                    matched_cluster['merged'] += 1

            # at the cluster level we store the union'd sdr
            # TODO: should this be the mean?

            # find the bits that are different
            diffs = np.logical_xor(matched_cluster['data'], current_sdr).astype(int)
            # get a list of idxs with diffs
            diff_idxs = np.where(diffs == 1)[0]
            # select one of the bits to randomly modify
            if (len(diff_idxs) > 0):
                # we could alter mor than 1 bit at a time here
                choice = np.random.choice(diff_idxs, 1)
                matched_cluster['data'][choice[0]] = current_sdr[choice[0]]

        else:
            # different - create new entry
            print(f"Adding cluster number: {len(clusters)}")
            idx[f] = len(clusters)
            clusters.append({'id':len(clusters),'data': current_sdr, 'merged': 0, 'count': 1, 'history': [{"data":current_sdr}]})

        clusters = merge_clusters(clusters, overlap_dist, max_cluster_bits)

    # todo: this only works for one level deep
    return clusters, [idx[i] if clusters[idx[i]]['id'] != -1 else clusters[idx[i]]['merged_with'] for i in range(len(idx))]


# To merge replce the target cluster 'id' with -1 and copy history to the source
# also update the source matching cluster union with new history values
def merge_sdrs(sdrs, source_id, target_id):
    print(f"Merging: {target_id} into {source_id}")
    sdrs[source_id]['history'] = sdrs[source_id]['history'] + sdrs[target_id]['history']
    sdrs[target_id]['id'] = -1
    sdrs[source_id]['count'] += sdrs[target_id]['count']
    # create the newly merged union
    sdrs[source_id]['data'] = np.logical_or(sdrs[source_id]['data'], sdrs[target_id]['data']).astype(int)
    sdrs[target_id]['merged_with'] = source_id
    return sdrs


def merge_clusters(sdrs, overlap_dist, max_overlap_dist):

    random_order = np.random.permutation(len(sdrs))

    for r in random_order:
        sdr = sdrs[r]
        min_overlap = sys.maxsize

        for r2 in random_order:
            i = sdrs[r2]
            if i['id'] != sdr['id'] and i['id'] != -1 and sdr['id'] != -1:  # -1 means it was merged
                overlap = np.dot(sdr['data'],i['data'])

                if overlap < min_overlap and overlap != 0:
                    min_overlap = overlap
                    sdr['min_overlap'] = min_overlap

                    # these clusters overlap - so we should merge them?
                    if min_overlap > overlap_dist:
                        unioned_total = np.sum(np.logical_or(sdr['data'],i['data']).astype(int))

                        if unioned_total < max_overlap_dist:
                            sdrs = merge_sdrs(sdrs, sdr['id'],i['id'])
                            # recurse back - as other changes may now be needed
                            sdrs = merge_clusters(sdrs, overlap_dist,max_overlap_dist)
                            break
                        else:
                            print("cannot grow any more")

    return sdrs


def sdr_in(sdrs, target):
    found = False

    for sdr in sdrs:
        found = np.equal(sdr['data'], target).all()
        if found:
            break

    return found


def find_closest_sdr(sdrs, target):
    max_overlap = 0
    found = None

    random_order = np.random.permutation(len(sdrs))

    for i in random_order:
        sdr = sdrs[i]
        if sdr['id'] != -1:
            overlap  = np.dot(sdr['data'], target)
            if overlap > max_overlap:
                max_overlap = overlap
                found = sdr

    return found, max_overlap


def find_closest_sdrs(sdrs, target, overlap_dist):
    result_list = []
    max_overlap = 0
    avg_dist = 0

    random_order = np.random.permutation(len(sdrs))

    for i in random_order:
        sdr = sdrs[i]
        overlap = np.dot(sdr['data'], target)
        if overlap > overlap_dist:
            result_list.append(sdr)
            avg_dist += overlap
        if overlap > max_overlap:
            max_overlap = overlap

    return result_list, max_overlap, avg_dist/max(len(result_list),1)
