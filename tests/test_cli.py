import os
import subprocess
import pytest


def test_run_cli():
    result = subprocess.run(['alina', '-h'])
    assert result.returncode==0


@pytest.mark.parametrize(
    "seq, struct, th",
    [
        (
            'GCAGUACUGGUGUAAUGGUUAGCGCCUUAGAUUUCCAAUCUAAAGGUAGGGGUUCGAUUCCCCUGUACUGCUCCA', 
            '(((((((..((((........)))).(((((.......)))))....(((((.......))))))))))))....',
            '0.5'
        ),
        (
            'GCCAACCCGGUCAGGUCCGGAAGGAAGCAGCCG', 
            '.((...((((......))))..)).........',
            '0.5'
        ),
        (
            'CGCUCAACUCAUGGAGCGCAAGACGAAUAGCUACAUAUUCGACAUGAG', 
            '.((((..[[[[[.))))......((((((......))))))..]]]]]',
            '0.5'
        ),
        (
            'UGUGUUUCACUCCUUUUACAUGUAUUGUAAACCUUCACACUAAUGUGAAGG',
            '.(((...)))....((((((.....))))))((((((((....))))))))',
            '0.2'
        ),
        (
            'UGUGUUUCACUCCUUUUACAUGUAUUGUAAACCUUCACACUAAUGUGAAGG',
            '................((.........))...(((((((....))))))).',
            '0.90'
        ),
    ]
)
def test_single_sequence(seq, struct, th):
    tmp_path = os.path.join('tests', 'tmp.txt')
    result = subprocess.run(['alina', '-i', seq, '-th', th], capture_output=True)
    out = result.stdout.decode().strip()
    
    assert out==struct
    
    
def test_fasta():
    expected = (
        '((((((....)))))).........................((((((((...))))))))......',
        '......(..[(((.((((((........))))))))))]...................',
        '((((((((.....(.........[[.....)..]]))))))))......(((((....)))))............'
    )
    
    fasta_path = os.path.join('tests', 'fasta.fasta')
    tmp_path = os.path.join('tests', 'tmp.txt')
    
    subprocess.run(['alina', '-m', 'file', '-i', fasta_path, '-o', tmp_path])
    
    with open(tmp_path) as f:
        lines = [l.strip() for l in f]
    
    preds = lines[2::3]
    for p, t in zip(preds, expected):
        assert p==t