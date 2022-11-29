import os
import pytest

import alina
from alina.utils import SequenceError



invalid_seqs = ['', 'A'*257, 'Ab', 'A.']
single_seq_invalid_types = [tuple(), 0, 1., dict()]
mult_seq_invalid_types = [
        ['AUAU', list()], 
        ['AUAU', tuple()], 
        ['AUAU', 0], 
        ['AUAU', 1.], 
        ['AUAU', dict()]
]
    

class TestErrors:
    
    @pytest.fixture(scope="session")
    def model(self):
        Alina = alina.AliNA()
        return Alina
    
    def test_single_seq(self, model):
        for t in single_seq_invalid_types:
            with pytest.raises(TypeError):
                s = model.fold(t)
                
    def test_mult_seq(self, model):
        for t in mult_seq_invalid_types:
            with pytest.raises(TypeError):
                s = model.fold(t)
    
    @pytest.mark.parametrize(
        "seq",
        invalid_seqs
    )
    def test_invalid_seq(self, seq, model):
        with pytest.raises(SequenceError):
            s = model.fold(seq)
        

class TestSkips:
    
    @pytest.fixture(scope="session")
    def model(self):
        Alina = alina.AliNA(skip_error_data=True, warn=False)
        return Alina
    
    @pytest.mark.parametrize(
        "seq",
        invalid_seqs
    )
    def test_invalid_seq(self, seq, model):
        s = model.fold(seq)
        assert s is None
            
    @pytest.mark.parametrize(
        "seq",
        [['auaugc', s] for s in invalid_seqs]
    )
    def test_invalid_seq(self, seq, model):
        ss = model.fold(seq)
        assert ss[1] is None
        
        
class TestThreshold:
    
    @pytest.mark.parametrize(
        "struct, th",
        [
            ('([[[[..[[[[.[[[...{{)]]].]]]]..]]]].......}}', 0.2), 
            ('([[[[..[[[[.[.[.....)].].]]]]..]]]].........', 0.4),
            ('.((((..((((..............))))..)))).........', 0.6),
            ('............................................', 0.8),
        ]
    )
    def test_threshold(self, struct, th):
        seq = 'AGCGAGGAGUUAUUUGAUGCUAGGAGGCUGAUCGCGGCCGAGGC'
        Alina = alina.AliNA()
        ss = Alina.fold(seq, threshold=th)
        assert ss == struct
        
        
        
        
        
        
        
        
        