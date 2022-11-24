from typing import Union

class SequenceError(ValueError):
    pass


class FastaError(ValueError):
    pass


def validate_sequence(seq : str):
    if not isinstance(seq, str):
        raise SequenceError(f'Sequence must be a string, got {type(seq)}')

    seq = seq.upper()
    if len(seq)>256 or len(seq)==0:
        raise SequenceError(f'Sequence length must be in range (0, 256], got {len(seq)}')

    rn = set(seq) - {'A', 'U', 'G', 'C'}
    if len(rn)!=0:
        raise SequenceError(f'Sequence contains unknown symbols: {tuple(rn)}')

    return seq


def validate_data(seq : Union[str, list]):
    if not isinstance(seq, (str, list)):
        raise TypeError(f'Input data must be string or list of strings, got {type(seq)}')
        
    if isinstance(seq, list):
        for d in seq:
            if not isinstance(d, str):
                raise TypeError(f'Input list must contain only strings, got {type(d)}')