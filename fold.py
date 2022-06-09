import sys
import os
import argparse


	
if __name__=='__main__':
    parser = argparse.ArgumentParser(description='AliNA args')
    
    parser.add_argument('-m', '--mode', type=str, 
                        choices=['seq', 'file'], default='seq', 
                        help='Prediction mode: "seq" - for single sequence typed in arguments. "file" - for multiple predictions from .fasta file.')
                        
    parser.add_argument('-i', '--input', type=str, 
                        required=True, metavar='Input', 
                        help='Input sequence or path to fasta file')
                        
    parser.add_argument('-o', '--out', type=str, 
                        help='Path to the output file for "file" mode. Default - Prediction_<input file name>')
    
    parser.add_argument('-th', '--threshold', type=float, default=0.5, 
                        help="Threshold value in range [0, 1] for model's output processing. The bigger value gives less complementary bonds. Default - 0.5")
    
    args = parser.parse_args()
    assert 0.<args.threshold<1.
    
    if args.out is None and args.mode=='file':
        path_comp = args.input.split(os.sep)
        path_comp[-1] = f'Prediction_{path_comp[-1]}'
        args.out = os.sep.join(path_comp)
    
    from utils.predictor import process_data
    process_data(args)
    
	
	
    
    
    
   
