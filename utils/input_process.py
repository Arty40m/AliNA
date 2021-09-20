import numpy as np
import warnings
from utils.dictionary import pair_dict



class FastaRead():
	def __init__(self, file):
		self.file_name = file

	def __enter__(self):
		self.file = open(self.file_name, 'r')
		self.gen = self.iter_func()
		return self

	def __exit__(self, _type, value, traceback):
		self.file.close()

	def __iter__(self):
		return self.gen

	def __next__(self):
		return next(self.gen)

	def __len__(self):
		self.reopen()
		self.gen = self.iter_func()
		
		n=0
		for _ in self.gen:
			n+=1
			
		self.reopen()
		self.gen = self.iter_func()
		return n
		
	def reopen(self):
		self.file.close()
		self.file = open(self.file_name, 'r')

	def iter_func(self):
		name=None
		
		for l in self.file:
            if l.isspace():
                continue
			if l[0] == '>':
				if name is not None:
					raise ValueError(f'Name "{name}" without sequence')
				name = l[1:-1]
			
			else:
				if name is None:
					raise ValueError(f'Sequence without name "{l[:-1]}"')
				
				yield name, l[:-1]
				name=None


def validate_seq(seq):
	if len(seq)>256:
		raise ValueError(f'Max sequence length is 256, got {len(seq)}')
	s = []
	for c in seq:
		uc = c.upper()
		if uc in 'AUGCIN':
			s.append(uc)
		elif uc == 'T':
			s.append('U')
		else:
			warnings.warn(f'unknown symbol {c}, changed to N')
			s.append('N')
 			
	return ''.join(s)


def pad(seq):
	left = (256 - len(seq))// 2
	right = 256 - len(seq) - left
	return 'N'*left + seq + 'N'*right


def make_mtx(seq):
	leng = len(seq)
	M = np.zeros((leng, leng))
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
    
    
def seq2matrix(seq):
    seq = validate_seq(seq)
    seq = pad(seq)
    matrix = make_mtx(seq)
    
    return matrix
    
    
