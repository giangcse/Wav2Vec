# -*- coding: utf-8 -*-
#####################################################################################
#   File API gồm các class chứa các Model Speech to text và model khử noise audio   #
#   Cùng với class API chứa các API endpoint để tương tác với hệ thống              #
#                                                                                   #
#####################################################################################
import json
import os, zipfile
import uvicorn
import shutil
import soundfile as sf
import torch
import kenlm
import librosa
import numpy as np
import sqlite3
import datetime

from fastapi import FastAPI, Form, File, UploadFile, WebSocket
from fastapi.requests import Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from starlette.responses import FileResponse
from pydantic import BaseModel
from typing import Union

from datasets import load_dataset
from pyctcdecode import Alphabet, BeamSearchDecoderCTC, LanguageModel
from transformers.file_utils import cached_path, hf_bucket_url
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC

from pydub import AudioSegment
from pydub.silence import split_on_silence

from speechbrain.pretrained import SepformerSeparation as separator
import torchaudio


class BaseVietnamese_Model:
    def __init__(self) -> None:
        # Khoi tao model
        self.cache_dir = './cache/'
        self.processor = Wav2Vec2Processor.from_pretrained("models/wav2vec2-base-vietnamese-250h", local_files_only=True)
        self.model = Wav2Vec2ForCTC.from_pretrained("models/wav2vec2-base-vietnamese-250h", local_files_only=True)
        self.lm_file = "models/wav2vec2-base-vietnamese-250h/vi_lm_4grams.bin.zip"
        print('[INFO]\t{}'.format('Model Wav2Vec2 Base Vietnamese 250h'))
        with zipfile.ZipFile(self.lm_file, 'r') as zip_ref:
            zip_ref.extractall(self.cache_dir)
        self.lm_file = self.cache_dir + 'vi_lm_4grams.bin'

        self.ngram_lm_model = self.get_decoder_ngram_model(self.processor.tokenizer, self.lm_file)
        print('[INFO]\t{}'.format('N-Grams models load completed'))

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

    def wav2vec(self, y_mono, start, input_padding_len, chunk_len):
        chunk = y_mono[start-input_padding_len:start+chunk_len+input_padding_len]
        input_values = self.processor(
            chunk, 
            sampling_rate=16000, 
            return_tensors="pt"
        ).input_values
        # ).input_values.to("cuda")
        # model.to("cuda")
        logits = self.model(input_values).logits[0]
        # decode ctc output
        pred_ids = torch.argmax(logits, dim=-1)
        beam_search_output = self.ngram_lm_model.decode(logits.cpu().detach().numpy(), beam_width=500)
        greedy_search_output = self.processor.decode(pred_ids)
        return {"LM": beam_search_output, "notLM": greedy_search_output}

    def split_audio_on_silence(self, audio_path):
        #reading from audio mp3 file
        sound = AudioSegment.from_mp3(audio_path)

        # spliting audio files
        audio_chunks = split_on_silence(sound, min_silence_len=500, silence_thresh=-40 )

        # Create new folder with name of audio
        head, tail = os.path.split(audio_path)
        if not os.path.exists(os.path.join('audio', tail.split('.')[0])):
            os.mkdir(os.path.join('audio', tail.split('.')[0]))
        
        list_of_dir = []
        #loop is used to iterate over the output list
        for i, chunk in enumerate(audio_chunks):
            output_file = "chunk{0}.mp3".format(i)
            chunk.export(os.path.join('audio', tail.split('.')[0], output_file), format="mp3")
            list_of_dir.append(os.path.join('audio', tail.split('.')[0], output_file))
        return list_of_dir

class DenoiseAudio:
    def __init__(self) -> None:
        self.model = separator.from_hparams(source="speechbrain/sepformer-wham16k-enhancement", savedir='models/sepformer-wham16k-enhancement')
        print('[INFO]\t{}'.format('Denoise model has been loaded'))

    def denoise(self, audio_path):
        if(os.path.exists(audio_path)):
            est_sources = self.model.separate_file(path=audio_path)
            try:
                # torchaudio.save("fail_e.wav", est_sources[:, :, 0].detach().cpu(), 16000)
                mono_audio = (est_sources[:, :, 0].detach().cpu())[0].numpy()
                return mono_audio
            except Exception:
                return Exception


class API:
    def __init__(self) -> None:
        # API Initial
        self.app = FastAPI()
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=['*'],
            allow_credentials=True,
            allow_methods=['*'],
            allow_headers=['*']
        )
        print('[INFO]\t{}'.format('API initial'))

        # Model speech to text
        self.BVM = BaseVietnamese_Model()
        # Model denoise
        self.DA = DenoiseAudio()

        # endpoint websocket
        @self.app.websocket("/ws")
        async def websocket_endpoint(websocket: WebSocket):
            await websocket.accept()
            while True:
                with open('config.json', encoding='utf8') as f:
                    config = json.loads(f.read())
                data = await websocket.receive_text()
                data = json.loads(data)
                if (int(data['denoise'])==0):
                    y, sr = librosa.load(data['audio'], mono=False, sr=16000)
                    y_mono = librosa.to_mono(y)
                else:
                    y_mono = self.DA.denoise(data['audio'])
                chunk_duration = config['chunk_duration'] # sec
                padding_duration = config['padding_duration'] # sec
                sample_rate = config['sample_rate']

                chunk_len = chunk_duration*sample_rate
                input_padding_len = int(padding_duration*sample_rate)
                output_padding_len = self.BVM.model._get_feat_extract_output_lengths(input_padding_len)

                # Ghi file log
                log_file =  open((data['audio'])[:-4] + '_250H.txt', 'a', encoding='utf8')
                sec = 0
                for start in range(input_padding_len, len(y_mono)-input_padding_len, chunk_len):
                    result = self.BVM.wav2vec(y_mono, start, input_padding_len, chunk_len)
                    if data['LM']==1:
                        text = result['LM']
                    else:
                        text = result['notLM']

                    if data['keyframe']==1:
                        return_data = {"time": str(datetime.timedelta(seconds=sec)), "text": text}
                        await websocket.send_text(f"{return_data}")
                        log_file.write("{:>12} {}\n".format(str(datetime.timedelta(seconds=sec)), text))
                    else:
                        await websocket.send_text(f"{text}")
                        log_file.write(text + " ")
                    sec += chunk_duration
                log_file.close()
  

api = API()

if __name__=='__main__':
    uvicorn.run('250h:api.app', port=9090, reload=True)
