from .validation import FastaError

class NARead():
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

    
    def __len__(self):
        self.file.seek(0)
        for n, _ in enumerate(self.gen, start=1):
            pass
        self.file.seek(0)
        self.gen = self.iter_func()
        return n
        

    def iter_func(self):
        name=None
        seq=None
        
        for n, l in enumerate(self.file):
            if len(l.strip())==0: continue
                
            if l[0] == '>':
                if seq is None:
                    if n!=0:
                        raise FastaError(f'Name with no sequence at line {n}')
                else:
                    yield name, seq
                    
                name = l.rstrip('\n')[1:]
                seq = None
                
            elif seq is None and name is not None:
                seq = l.rstrip('\n')
                
            else:
                raise FastaError(f'Name must start with ">" symbol at line {n+1}')
        
        if seq is None:
            raise FastaError(f'Name with no sequence at line {n+1}')
            
        yield name, seq
        
        
class NAWrite():
    def __init__(self, file):
        self.file_name = file


    def __enter__(self):
        self.file = open(self.file_name, 'w')
        return self


    def __exit__(self, _type, value, traceback):
        self.file.close()


    def write(self, d):
        name, seq, struct = d
        
        self.file.write(f'>{name}\n')
        self.file.write(f'{seq}\n')
        self.file.write(f'{struct}\n')
        