# -*- coding: utf-8 -*-

import json
import os, zipfile, sys
import uvicorn
import torch
import kenlm
import librosa
import numpy as np
import datetime

from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware

sys.path.insert(1,os.path.abspath('py37/pyctcdecode1'))
# from pyctcdecode import Alphabet, BeamSearchDecoderCTC, LanguageModel
from py37.pyctcdecode1.pyctcdecode import Alphabet, BeamSearchDecoderCTC, LanguageModel
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC

from speechbrain.pretrained import SepformerSeparation as separator

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

class DenoiseAudio:
    def __init__(self) -> None:
        self.model = separator.from_hparams(source="speechbrain/sepformer-wham16k-enhancement", savedir='models/sepformer-wham16k-enhancement')
        print('[INFO]\t{}'.format('Denoise model has been loaded'))

    def denoise(self, audio_path):
        if(os.path.exists(audio_path)):
            est_sources = self.model.separate_file(path=audio_path)
            try:
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
                return_data = None
                for start in range(input_padding_len, len(y_mono)-input_padding_len, chunk_len):
                    result = self.BVM.wav2vec(y_mono, start, input_padding_len, chunk_len)
                    if data['LM']==1:
                        text = result['LM']
                    else:
                        text = result['notLM']

                    if data['keyframe']==1:
                        return_data = {"time": str(datetime.timedelta(seconds=sec)), "text": text}
                        await websocket.send_text((return_data))
                        log_file.write("{:>12} {}\n".format(str(datetime.timedelta(seconds=sec)), text))
                    else:
                        return_data = text
                        await websocket.send_text(return_data)
                        log_file.write(text + " ")
                    sec += chunk_duration
                log_file.close()
  

api = API()

if __name__=='__main__':
    uvicorn.run('vietnamesemodel:api.app', host="0.0.0.0", port=9090, reload=True)
