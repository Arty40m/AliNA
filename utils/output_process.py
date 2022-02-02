import numpy as np



def probable_triang_matrix(M, sensitivity = 0.5):
    mol_size = M.shape[-1]
    diag = (np.diag(np.ones((mol_size)))==0)
    fm = np.zeros((mol_size, mol_size))
    
    thmask = s>sensitivity
    s = M*diag*thmask

    while np.sum(s)>0:
      m = np.argmax(s)
      r = m//mol_size
      c = m%mol_size

      fm[r,c] = 1
      s[r] = 0
      s[c] = 0
      s[:, r] = 0
      s[:, c] = 0

    x = np.triu((fm + fm.T), k=1)
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
    
    
def assemble_from_layers(layers, mol_size):
	types = {0:'()', 1:'[]', 2:'{}', 3:'<>', 4:'Aa', 5:'Bb', 6:'Cc', 7:'Dd', 8:'Ee'}
	intersection = [0 for _ in range(len(layers))]
	dots = ['.' for _ in range(mol_size)]
    
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
			return None
		typ = min(set([i for i in range(9)]) - types_set)# lowest type of brackets
		intersection[l] = typ
		        
		brackets = types[typ]
		for c in range(len(layer[0])):
			dots[layer[0][c]] = brackets[0]# open bracket
			dots[layer[1][c]] = brackets[1]# close bracket
	
	return ''.join(dots)


def matrix2struct(M, sensitivity = 0.5):
	M = probable_triang_matrix(M, sensitivity = sensitivity)
	vec = np.argmax(M, axis=-1)
	layers = get_struct_layers(vec)
	struct = assemble_from_layers(layers, vec.shape[0])
	
	return struct
	
	
def unpad(struct, seq):
	left = (256 - len(seq))// 2
	right = 256 - len(seq) - left
	return struct[left:(left+len(seq))]



