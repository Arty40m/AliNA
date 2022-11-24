import numpy as np


def quantize_matrix(M, sensitivity = 0.5):
    seq_length = M.shape[-1]
    diag = (np.diag(np.ones((seq_length)))==0)
    fm = np.zeros((seq_length, seq_length))
    
    thmask = M>sensitivity
    s = M*diag*thmask

    while np.sum(s)>0:
        m = np.argmax(s)
        r = m//seq_length
        c = m%seq_length

        fm[r,c] = 1
        s[r] = 0
        s[c] = 0
        s[:, r] = 0
        s[:, c] = 0

    x = (fm + fm.T)
    return x


def get_struct_layers(vec):
    layers = []
    op = []
    end = []
    
    for i in range(vec.shape[0]):
        if vec[i]==0:
            continue
        if len(op)==0:
            op.append(i)
            end.append(vec[i])
            continue

        if i-op[-1]==1 and end[-1]-vec[i]==1:
            op.append(i)
            end.append(vec[i])
        else:
            layers.append((tuple(op), tuple(end)))
            op = [i]
            end = [vec[i]]

    if len(op)!=0:
        layers.append((tuple(op), tuple(end)))

    return layers
    
    
def assemble_from_layers(layers, seq_length):
    types = {0:'()', 1:'[]', 2:'{}', 3:'<>', 4:'Aa', 5:'Bb', 6:'Cc', 7:'Dd', 8:'Ee'}
    intersection = [0 for _ in range(len(layers))]
    dots = ['.' for _ in range(seq_length)]
    
    def intersect(l, prel):
        if l[0][-1]<prel[1][0] and l[1][0]>prel[1][-1]:# .(((..[[..)))..]].
            return True
    
    for l in range(len(layers)):
        layer = layers[l]
        
        int_types = []
        for prel in range(l):
            if intersect(layer, layers[prel]):
                int_types.append(intersection[prel])
        
        types_set = set(int_types)
        if len(types_set)>=9:
            break
        typ = min(set([i for i in range(9)]) - types_set)# lowest type of brackets
        intersection[l] = typ
                
        brackets = types[typ]
        for c in range(len(layer[0])):
            dots[layer[0][c]] = brackets[0]# open bracket
            dots[layer[1][c]] = brackets[1]# close bracket
    
    return ''.join(dots)

    
def matrix2struct(M):
    M = np.triu(M, k=1)
    vec = np.argmax(M, axis=-1)
    layers = get_struct_layers(vec)
    struct = assemble_from_layers(layers, vec.shape[0])

    return struct


class NaStack():
    inv_dict = {')':'(', ']':'[', '}':'{', '>':'<', 
                         'a':'A', 'b':'B', 'c':'C', 'd':'D', 'e':'E'}
    
    def __init__(self,):
        self.st = {'(':[], '[':[], '{':[], '<':[], 
                   'A':[], 'B':[], 'C':[], 'D':[], 'E':[]}

    def __setitem__(self, k, v):
        self.st[k].append(v)
    
    def __getitem__(self, k):
        return self.st[self.inv_dict[k]].pop()
        
    def isempty(self):
        for k in self.st:
            if len(self.st[k])!=0:
                return False
        return True


def get_na_pairs(struct):
    pairs = []
    stack = NaStack()

    for i in range(len(struct)):
        if struct[i] == '.':
            continue

        if struct[i] in '([{<ABCDE':
            stack[struct[i]] = i

        elif struct[i] in ')]}>abcde':
            op_ind = stack[struct[i]]
            pairs.append((op_ind, i))
            
    assert stack.isempty()
        
    return pairs