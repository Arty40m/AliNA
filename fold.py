import sys
import os
from utils.predictor import process_data




def parse_args(arg_list):
    args = {
            'mode':None,
            'data':None,
            'out':None,
            'th':0.5
            }

    if len(arg_list)==1:
        raise ValueError('No arguments')
    if len(arg_list)%2==0:
        raise ValueError(f'Argument without value {arg_list[-1]}')

    for i in range(1, len(arg_list), 2):
        a = arg_list[i]
        if a not in ('-s', '-f', '-o', '-th'):
            raise ValueError(f'Invalid argument {a}')
        if a == '-s':
            args['mode'] = 'seq'
            args['data'] = arg_list[i+1]
        elif a == '-f':
            args['mode'] = 'file'
            args['data'] = arg_list[i+1]
        elif a == '-o':
            args['out'] = arg_list[i+1]
        elif a == '-th':
            args['th'] = float(arg_list[i+1]) 
            
    if args['mode'] is None:
        raise ValueError('Prediction mode is not defined, must be -s or -f')
    if args['mode'] == 'file':
        path = args['data']
        if not os.path.isfile(path):
            raise ValueError(f'Invalid data path {path}')
            
        if args['out'] is None:
            path_comp = args['data'].split(os.sep)
            out_name = 'Prediction_'+path_comp[-1]
            path_comp[-1] = out_name
            args['out'] = os.sep.join(path_comp)

    return args


if __name__=='__main__':
   args = parse_args(sys.argv)
   process_data(args)
   
   
   
   
