# -*- coding: utf-8 -*-
import json
import os, zipfile
import uvicorn
import shutil
import soundfile as sf
import torch
import kenlm
import librosa

from fastapi import FastAPI, Form, File, UploadFile
from fastapi.requests import Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import Union

from datasets import load_dataset
from pyctcdecode import Alphabet, BeamSearchDecoderCTC, LanguageModel
from transformers.file_utils import cached_path, hf_bucket_url
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC

class API:
    def __init__(self) -> None:
        # API Initial
        print('*********** API Initial ***********')
        self.app = FastAPI()
        self.app.mount("/static", StaticFiles(directory="static"), name="static")
        self.templates = Jinja2Templates(directory="templates/")
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=['*'],
            allow_credentials=True,
            allow_methods=['*'],
            allow_headers=['*']
        )
        # Model Initial
        print('*********** Model Initial ***********')
        self.cache_dir = './cache/'
        self.processor = Wav2Vec2Processor.from_pretrained("nguyenvulebinh/wav2vec2-base-vietnamese-250h", cache_dir=self.cache_dir)
        self.model = Wav2Vec2ForCTC.from_pretrained("nguyenvulebinh/wav2vec2-base-vietnamese-250h", cache_dir=self.cache_dir)
        self.lm_file = hf_bucket_url("nguyenvulebinh/wav2vec2-base-vietnamese-250h", filename='vi_lm_4grams.bin.zip')
        self.lm_file = cached_path(self.lm_file,cache_dir=self.cache_dir)
        print('*********** Model OK ***********')
        print('*********** get_decoder_ngram_model ***********')
        with zipfile.ZipFile(self.lm_file, 'r') as zip_ref:
            zip_ref.extractall(self.cache_dir)
        self.lm_file = self.cache_dir + 'vi_lm_4grams.bin'

        self.ngram_lm_model = self.get_decoder_ngram_model(self.processor.tokenizer, self.lm_file)
        print('*********** get_decoder_ngram_model OK ***********')

        @self.app.get("/")
        async def root(request: Request):
            return self.templates.TemplateResponse('index.html', context={'request': request})
        
        @self.app.post("/")
        async def upload(request: Request, file: UploadFile = File (...)):
            with open(os.path.join('static', file.filename), 'wb') as pdf:
                shutil.copyfileobj(file.file, pdf)
            return self.templates.TemplateResponse('index.html', context={'request': request, 'result': self.wav2vec(os.path.join('static', file.filename))})

    def get_decoder_ngram_model(self, tokenizer, ngram_lm_path):
        vocab_dict = tokenizer.get_vocab()
        sort_vocab = sorted((value, key) for (key, value) in vocab_dict.items())
        vocab = [x[1] for x in sort_vocab][:-2]
        vocab_list = vocab
        # convert ctc blank character representation
        vocab_list[tokenizer.pad_token_id] = ""
        # replace special characters
        vocab_list[tokenizer.unk_token_id] = ""
        # vocab_list[tokenizer.bos_token_id] = ""
        # vocab_list[tokenizer.eos_token_id] = ""
        # convert space character representation
        vocab_list[tokenizer.word_delimiter_token_id] = " "
        # specify ctc blank char index, since conventially it is the last entry of the logit matrix
        alphabet = Alphabet.build_alphabet(vocab_list, ctc_token_idx=tokenizer.pad_token_id)
        lm_model = kenlm.Model(ngram_lm_path)
        decoder = BeamSearchDecoderCTC(alphabet,
                                    language_model=LanguageModel(lm_model))
        return decoder

    def map_to_array(self, batch):
        speech, sampling_rate = sf.read(batch["file"])
        batch["speech"] = speech
        batch["sampling_rate"] = sampling_rate
        return batch

    def wav2vec(self, audio_path):
        y, sr = librosa.load(audio_path, mono=False, sr=16000)
        y_mono = librosa.to_mono(y)
        chunk_duration = 5 # sec
        padding_duration = 1 # sec
        sample_rate = 16000

        chunk_len = chunk_duration*sample_rate
        input_padding_len = int(padding_duration*sample_rate)
        output_padding_len = self.model._get_feat_extract_output_lengths(input_padding_len)
        # if os.path.isfile('./cache/mono.wav'):
        #     os.remove('./cache/mono.wav')
        # sf.write('./cache/mono.wav', y_mono, 16000)


        # ds = self.map_to_array({"file": './cache/mono.wav'})
        # infer model
        all_preds = []
        for start in range(input_padding_len, len(y_mono)-input_padding_len, chunk_len):
            chunk = y_mono[start-input_padding_len:start+chunk_len+input_padding_len]
            input_values = self.processor(
                chunk, 
                sampling_rate=sample_rate, 
                return_tensors="pt"
            ).input_values
            # ).input_values.to("cuda")
            # model.to("cuda")
            logits = self.model(input_values).logits[0]
            # decode ctc output
            pred_ids = torch.argmax(logits, dim=-1)
            greedy_search_output = self.processor.decode(pred_ids)
            beam_search_output = self.ngram_lm_model.decode(logits.cpu().detach().numpy(), beam_width=500)
            print({'greedy': str(greedy_search_output), 'beam': str(beam_search_output)})
        return('ok')

api = API()

if __name__=='__main__':
    uvicorn.run('api:api.app', host='0.0.0.0', port=9090)
    # api.wav2vec(r'static/test.mp3')