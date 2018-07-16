import numpy as np
import traceback

def find_prob_matrix(l):
    try:
        uniq = np.unique(l)
        res = np.zeros([uniq.shape[0], uniq.shape[0]])
        l1 = [l[i+1] if i!=len(l)-1 else l[0] for i, w in enumerate(l)]
        cnts = {}
        tots = {}
        for q, w in zip(l, l1):
            cnts[(q, w)] = cnts.get((q, w), 0) + 1
            tots[q] = tots.get(q, 0) + 1
        ind = {v:i for i, v in enumerate(list(uniq))}
        for q in list(uniq):
            row = ind[q]
            for w in list(uniq):
                col = ind[w]
                res[row][col] = cnts.get((q, w), 0)/float(tots[q])
        return res, ind
    except Exception as e:
        print(traceback.format_exc())
        raise e

if __name__ == '__main__':
    l = ['c', 's', 's', 'r', 'c', 'r', 's']
    res, ind = find_prob_matrix(l)
    print(res, ind)
