Data description
===

Initial MXFold2 data can be found at - https://doi.org/10.5281/zenodo.4430150
For pseudoknots data PseudoBase database was parsed - https://www.ekevanbatenburg.nl/PKBASE (https://doi.org/10.1093/nar/28.1.201)
For augmentation RNAcentral data was used - https://rnacentral.org/downloads

# Files
- AllData750_family_names.fasta - filtered MXFold2 data with family names assigned by Rfam
- AllData750_OneInFamily.newick - phylogenetic tree file (family presented by its longest sequence) 
- Balanced750_family_names.fasta - file "AllData750_family_names.fasta" with 80% sequenced randomly discarded from 5S_rRNA and tRNA families 
- PseudoBase.fasta - sequences obtained by PseudoBase parsing
- RNAcentral256.fasta - filtered RNAcentral sequences with length in range (0, 256]
- family_names_RNAcentral256.fasta - RNAcentral sequences with family names assigned by Rfam
-
- UnseenTest750_245fams.fasta - 245 most dissimilar to train families on tree
- Valid750_200fams.fasta - 200 dissimilar families after UnseenTest on tree
- CloseTest750_200fams.fasta - 200 randomly sampled families
- WithinTest750_1500structs.fasta - 1500 randomly sampled sequences from all remaining families
- Train750.fasta - remaining data used for training model
- PseudoKnots750_889structs.fasta - pseudoknot structures collected from all sets except "Train750.fasta"
- RNAcentral256_Tree750FamFiltered_04-01_ss.fasta - RNAcentral sequences filtered from UnseenTest and validation by families and by homology with similarity threshold 0.6, then by homology with train set with similarity threshold 0.9

# Similarity metric
```
from Levenshtein import distance

def is_similar(s1, s2, th):
    m = max(len(s1), len(s2))
    if (1 - abs(len(s1)-len(s2))/m)<th:
        return False
    
    d = 1 - distance(s1, s2)/m
    if d>th:
        return True
    return False
    
s1 = 'ABABABABCC'
s2 = 'ABABABABAB'

is_similar(s1, s2, th=0.5)
>>> True
is_similar(s1, s2, th=0.9)
>>> False
```