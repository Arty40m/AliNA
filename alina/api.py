import warnings
import sys
warnings.simplefilter('always', UserWarning)
from typing import Union

if sys.version_info>=(3, 9):
    import importlib.resources as pkg_resources
else:
    import importlib_resources as pkg_resources

import torch

from .utils import seq2matrix, pad_bounds, quantize_matrix, matrix2struct, validate_sequence, validate_data
from .model import Alina, model_parameters



class AliNA:
    
    MODEL_WEIGHTS = 'Tuned_AliNA.pth'
    
    def __init__(self,
                skip_error_data : bool = False,
                warn : bool = True,
                gpu : bool = False
                ):
        
        self.skip_error_data = skip_error_data
        self.warn = warn
        
        self.device = 'cpu'
        if gpu:
            if torch.cuda.is_available():
                self.device = 'cuda'
            else:
                warnings.warn('Cuda is not available, AliNA will run on cpu!')
        else:
            self.device = 'cpu'
        
        self.model = self.load_model()
        
        
    def fold(self, 
             seq : Union[str, list],
             threshold : int = 0.5,
             with_probs : bool = False
            ):
        
        if threshold>1 or threshold<0:
            raise ValueError(f'Threshold value must be in the range [0, 1], got {threshold}')
        
        validate_data(seq)
        
        single_sequence = False
        if isinstance(seq, str):
            single_sequence = True
            seq = [seq]
            
        results = []
        for n, s in enumerate(seq):
            try:
                val_s = validate_sequence(s)
            except Exception as e: 
                if self.skip_error_data:
                    if self.warn: warnings.warn(str(e))
                    results.append(None)
                    continue
                else:
                    raise e
                    
            struct, probs = self._predict(val_s, threshold)
            if with_probs:
                results.append((struct, probs))
            else:
                results.append(struct)
            
        if single_sequence:
            return results[0]
        return results
        
        
    def _predict(self, seq: str, threshold: float):
        l, r = pad_bounds(seq)
        padded_seq = ''.join(['N'*l, seq, 'N'*r])
        
        inp = seq2matrix(padded_seq).reshape((1, 256, 256))
        inp = torch.from_numpy(inp)
        if self.device=='cuda':
            inp = inp.to(self.device)
        
        with torch.no_grad():
            pred = self.model(inp)
        
        if self.device=='cuda':
            pred = pred.to('cpu')
        pred = pred.numpy()
        pred = pred[l:-r, l:-r]
        
        M = quantize_matrix(pred, threshold = threshold)
        struct = matrix2struct(M)
        
        return struct, pred
        
        
    def load_model(self):
        model_pkg = pkg_resources.files("alina.model")
        weights_path = model_pkg.joinpath(self.MODEL_WEIGHTS)
        
        model = Alina(**model_parameters)
        if self.device=='cuda':
            model = model.to(self.device)
        model.load_state_dict(torch.load(weights_path))
        model.eval()

        return model
        
        
        
        
        