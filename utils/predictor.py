import numpy as np
import sys
import os
import warnings
from model.model import AliNA
from utils.input_process import FastaRead, seq2matrix
from utils.output_process import matrix2struct, unpad




def load_model():
    print('Creating model...')
    model = AliNA()
    name =  'Att256Emb16_7-4_18k_l3_val_acc.ckpt'
    weights = os.sep.join([sys.path[0], 'model', name])
    model.compile()
    print('Loading weights...')
    model.load_weights(weights)
    return model


def progress_bar(i, N):
    n = round(30*((i+1)/N))
    return '|{}{}| {}/{}'.format('='*n, '.'*(30-n), i+1, N)


def process_data(args):
    if args.mode=='seq':
        inp = seq2matrix(args.input)
        if inp is None:
            return
			
        model = load_model()
        pred = model.predict(np.expand_dims(inp, axis=0))
        struct = matrix2struct(np.squeeze(pred), args.threshold)
        
        if struct is None:
            warnings.warn("Couldn't process prediction, the structure contains exceedingly high-order knots. See manual for more information")
            return
			
        print(f'Prediction:\n{unpad(struct, args.input)}')
        return
	
    # file
    with FastaRead(args.input) as f, open(args.out, 'w') as w:
        print('Checking data...')
        N = len(f)
        model = load_model()
        p=0
        for i, d in enumerate(f):
            name, rowseq = d
            inp = seq2matrix(rowseq)
            if inp is None:
                continue
			
            pred = model.predict(np.expand_dims(inp, axis=0))
            struct = matrix2struct(np.squeeze(pred), args.threshold)
        
            if struct is None:
                warnings.warn(f"Couldn't process prediction for sequence {name}, too hight order knots in the structure")
                continue
			
            struct = unpad(struct, rowseq)
            w.write('>'+name+'\n')
            w.write(rowseq+'\n')
            w.write(struct+'\n')
            p+=1
			
            bar = progress_bar(i, N)
            print(f'\r{bar}', end='')

        print(f'\nDone\n{p} predictions were written into "{args.out}"')
	
        
