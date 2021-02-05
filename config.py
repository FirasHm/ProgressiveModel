'''
HYPERPARAMETER 입력받는거 정의하는 부분
'''
import argparse

def get_params():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--embedding_size", type=int, default=512)
    parser.add_argument("--hidden_size", type=int, default=512)
    parser.add_argument("--tgt_dm", type=str, )
    parser.add_argument("--ide_type", type=str, default="t")

    params = parser.parse_args()
    return params