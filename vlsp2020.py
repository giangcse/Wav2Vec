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
from transformers.file_utils import cached_path, hf_bucket_url
from importlib.machinery import SourceFileLoader
from transformers import Wav2Vec2ProcessorWithLM

from pydub import AudioSegment
from pydub.silence import split_on_silence

from speechbrain.pretrained import SepformerSeparation as separator
import torchaudio

class LargeVLSP_Model:
    def __init__(self) -> None:
        # Khoi tao model
        self.model_name = "models/wav2vec2-large-vi-vlsp2020"
        self.model = SourceFileLoader("model", "models/wav2vec2-large-vi-vlsp2020/model_handling.py").load_module().Wav2Vec2ForCTC.from_pretrained(self.model_name, local_files_only=True)
        self.processor = Wav2Vec2ProcessorWithLM.from_pretrained(self.model_name, local_files_only=True)
        print('[INFO] \t{}'.format("Model wav2vec2-large-vi-vlsp2020 has been loaded"))

    def load(self, audio):
        input_data = self.processor.feature_extractor(audio, sampling_rate=16000, return_tensors='pt')
        output = self.model(**input_data)
        woLM = self.processor.tokenizer.decode(output.logits.argmax(dim=-1)[0].detach().cpu().numpy())
        wLM = self.processor.decode(output.logits.cpu().detach().numpy()[0], beam_width=100).text
        return {"LM": wLM, "notLM": woLM}

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
        self.LVM = LargeVLSP_Model()
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

                buffer = chunk_duration * 16000
                samples_total = len(y_mono)
                samples_wrote = 0

                # Ghi file log
                log_file =  open((data['audio'])[:-4] + '_VLSP2020.txt', 'a', encoding='utf8')
                sec = 0
                while samples_wrote < samples_total:
                    #check if the buffer is not exceeding total samples 
                    if buffer > (samples_total - samples_wrote):
                        buffer = samples_total - samples_wrote

                    block = y_mono[samples_wrote : (samples_wrote + buffer)]
                    samples_wrote += buffer

                    result = self.LVM.load(block)
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
    uvicorn.run('vlsp2020:api.app', port=9091, reload=True)