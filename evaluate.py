import argparse
from unsupervised.evaluators.utils import get_args_parser
from unsupervised.evaluators.evaluator import BaseEvaluator

if __name__ == '__main__':
    ##### Defining arguments #####
    parser = argparse.ArgumentParser(
        "Evaluating UCPD results on multi-modal data", parents=[get_args_parser()])
    args = parser.parse_args()
    
    evaluator = BaseEvaluator(args)
    evaluator.evaluate()
    

        
   
    
    
    
