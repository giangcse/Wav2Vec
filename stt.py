# -*- coding: utf-8 -*-
import json
import os, zipfile
import difflib
import re
import string, random
import multiprocessing

import warnings 
warnings.filterwarnings("ignore")

from vietnamesemodel import BaseVietnamese_Model
from vlsp2020 import LargeVLSP_Model

from normalize_text.infer import infer
from vfastpunct import VFastPunct
         
class STT:
    def __init__(self):
        self.VLSP = LargeVLSP_Model()
        self.VSM = BaseVietnamese_Model()  
        self.punc = VFastPunct(model_name='mBertPunctCap', no_cuda=False)        

    def convert(self, file_path='', enable_lm=1, key_frame=0, model=''):
        self.data = {'audio': file_path, 'keyframe': key_frame, 'LM': enable_lm, 'model': model}
        VLSP_string = ''
        VSM_string = ''
        return_string = ''  
        if(os.path.exists(file_path)):
            try:
                if(self.data['model'].lower()=='vlsp'):
                    for i in self.VLSP.speech_to_text(self.data):
                        return_string += str(i) + ' '
                elif(self.data['model'].lower()=='250h'):
                    for i in self.VSM.speech_to_text(self.data):
                        return_string += str(i) + ' '
                else:
                    for i in self.VLSP.speech_to_text(self.data):
                        VLSP_string += str(i) + ' '
                    for i in self.VSM.speech_to_text(self.data):
                        VSM_string += str(i) + ' '
                    return_string = self.punc(self.show_comparison(VLSP_string, VSM_string, sidebyside=False))
                return return_string
            except Exception:
                return "Error when reading file"
        else:
            return "File not found"
        

    def tokenize(self, s):
        return re.split('\s+', s)
    
    def untokenize(self, ts):
        return ' '.join(ts)
            
    def equalize(self, s1, s2):
        l1 = self.tokenize(s1)
        l2 = self.tokenize(s2)
        res1 = []
        res2 = []
        prev = difflib.Match(0,0,0)
        for match in difflib.SequenceMatcher(a=l1, b=l2).get_matching_blocks():
            if (prev.a + prev.size != match.a):
                for i in range(prev.a + prev.size, match.a):
                    res2 += ['_']
                res1 += l1[prev.a + prev.size:match.a]
            if (prev.b + prev.size != match.b):
                for i in range(prev.b + prev.size, match.b):
                    res1 += ['_']
                res2 += l2[prev.b + prev.size:match.b]
            res1 += l1[match.a:match.a+match.size]
            res2 += l2[match.b:match.b+match.size]
            prev = match
        return self.untokenize(res1), self.untokenize(res2)

    def insert_newlines(self, string, every=64, window=10):
        result = []
        from_string = string
        while len(from_string) > 0:
            cut_off = every
            if len(from_string) > every:
                while (from_string[cut_off-1] != ' ') and (cut_off > (every-window)):
                    cut_off -= 1
            else:
                cut_off = len(from_string)
            part = from_string[:cut_off]
            result += [part]
            from_string = from_string[cut_off:]
        return result

    def format_text(self,text_input, list_bias_input):
        bias_list = list_bias_input.strip().split('\n')
        norm_result = infer([text_input], bias_list)
        return norm_result[0]

    def show_comparison(self, s1, s2, width=40, margin=10, sidebyside=True, compact=False):
        s1, s2 = self.equalize(s1,s2)

        if sidebyside:
            s1 = self.insert_newlines(s1, width, margin)
            s2 = self.insert_newlines(s2, width, margin)
            if compact:
                for i in range(0, len(s1)):
                    lft = re.sub(' +', ' ', s1[i].replace('_', '')).ljust(width)
                    rgt = re.sub(' +', ' ', s2[i].replace('_', '')).ljust(width) 
                    print(lft + ' | ' + rgt + ' | ')        
            else:
                for i in range(0, len(s1)):
                    lft = s1[i].ljust(width)
                    rgt = s2[i].ljust(width)
                    print(lft + ' | ' + rgt + ' | ')
        else:
            sentence_1=str(s1).split(' ')
            sentence_2=str(s2).split(' ')
            return_data = ''
            for i in range(len(sentence_1)):
                if str(sentence_1[i]).lower() == 'tỷ':
                    sentence_1[i]='tỉ'
                if str(sentence_2[i]).lower() == 'tỷ':
                    sentence_2[i]='tỉ'

                if(sentence_1[i] == sentence_2[i]):
                    return_data += "{} ".format(sentence_1[i])
                else:
                    if('_' in sentence_1[i]):
                        return_data += "{}".format(sentence_2[i])
                    elif('_' in sentence_2[i]):
                        return_data += "{}".format(sentence_1[i])
                    return_data += ' '
            return self.format_text(return_data, '')

# if __name__=='__main__':
#     s2t = STT()
#     while True:
#         audio = input("Nhap duong dan: ")
#         if(audio=='/f'):
#             break
#         result = s2t.convert(file_path=audio)
#         print('\n\nResult: {}\n----------------------------------------------'.format(result))