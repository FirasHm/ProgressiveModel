'''
데이터 전처리
SNIPS 데이터셋 사용
'''
import numpy as np
import torch
import json

def snips_preprocess(root='./data'):
    intents = ['AddToPlaylist', 'BookRestaurant', 'GetWeather', 'PlayMusic', 'RateBook', 'SearchCreativeWork', 'SearchScreeningEvent']
    for intent in intents:
        slot_class = []
        