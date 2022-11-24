import numpy as np
from .dictionary import pair_dict


def seq2matrix(seq):
    leng = len(seq)
    M = np.zeros((leng, leng), dtype=np.int32)
    for n in range(leng):
        for p in range(n-1):
            fx = seq[n]
            fy = seq[p]

            fx1 = ''
            fy1 = ''
          
            if n<leng-1:
                fx1 = seq[n+1]
            if p<leng-1:
                fy1 = seq[p+1]
                
            M[n][p] = pair_dict[fx+fx1+'/'+fy1+fy]
            M[p][n] = pair_dict[fy+fy1+'/'+fx1+fx]
    return M


def pad_bounds(seq):
    left = (256 - len(seq))// 2
    right = 256 - len(seq) - left
    if right==0: right = -1
    return left, right
