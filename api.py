# -*- coding: utf-8 -*-
import json
import os, zipfile
import uvicorn
import shutil
import sqlite3
import datetime
import websockets
import asyncio
import difflib
import re

from fastapi import FastAPI, Form, File, UploadFile, WebSocket
from fastapi.requests import Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from starlette.responses import FileResponse
from pydantic import BaseModel

from vietnamesemodel import BaseVietnamese_Model
from vlsp2020 import LargeVLSP_Model

from infer import infer

class Delete_audio(BaseModel):
    username: str
    audio_name: str

class Get_audio(BaseModel):
    username: str

class API:
    def __init__(self) -> None:
        # Khởi tạo thông tin kết nối đến Database
        self.database = 'database.db'
        self.connection_db = sqlite3.connect(self.database, check_same_thread=False)
        self.cursor = self.connection_db.cursor()
        print('[INFO]\t{}'.format('Connected to database'))
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
        
        self.BVM = BaseVietnamese_Model()
        
        self.VLSP = LargeVLSP_Model()

        @self.app.get("/")
        async def root(request: Request):
            return self.templates.TemplateResponse('index.html', context={'request': request})
        # Endpoint get list audio
        @self.app.post("/get_list")
        async def get_list(request: Request, getAudio: Get_audio):
            find = self.cursor.execute("SELECT * FROM audios WHERE USERNAME = ?", (str(getAudio.username), ))
            audios = []
            for i in find.fetchall():
                audios.append(i[0])
            return JSONResponse(status_code=200, content={"data": audios})
        # Endpoint xoá audio
        @self.app.post("/delete")
        async def delete(request: Request, body: Delete_audio):
            deleted = self.cursor.execute("DELETE FROM audios WHERE USERNAME = ? AND AUDIO_NAME = ?", (str(body.username), str(body.audio_name), ))
            self.cursor.commit()
            if self.cursor.rowcount > 0:
                return JSONResponse(status_code=200, content={"result": "Xoá thành công"})
            else:
                return JSONResponse(status_code=500, content={"result": "Xoá không thành công"})
        # Enpoint upload audio
        @self.app.post("/")
        async def upload(request: Request, file: UploadFile = File (...)):
            username = self.cursor.execute('SELECT USERNAME FROM users WHERE USERNAME = ?', ('admin', )).fetchone()[0]
            if not os.path.exists(os.path.join('audio', username)):
                os.mkdir(os.path.join('audio', username))
            with open(os.path.join('audio', username, file.filename), 'wb') as audio:
                shutil.copyfileobj(file.file, audio)
            try:    
                storage = self.cursor.execute("INSERT INTO audios(AUDIO_NAME, USERNAME, DATETIME) VALUES (?, ?, ?)", (str(os.path.join('audio', username, file.filename)), str(username), datetime.datetime.now().strftime("YYYY-MM-DD hh:mm:ss.xxxxxx")))
                self.connection_db.commit()
            except Exception:
                pass
            return JSONResponse(status_code=200, content={'audios': [x[0] for x in self.cursor.execute("SELECT AUDIO_NAME FROM audios WHERwebsocketsE USERNAME = ?", (str(username),))]})
        # Endpoint allow to download audio
        @self.app.post("/download_audio")
        async def download_audio(request: Request, audio: Delete_audio):
            return FileResponse(audio.audio_name, media_type='application/octet-stream', filename=str(audio.audio_name).split('/')[-1])

        # Endpoint allow to download result
        @self.app.post("/download_text")
        async def download_text(request: Request, audio: Delete_audio):
            name = str(audio.audio_name)[:-4].split('/')[-1]
            return FileResponse(str(audio.audio_name)[:-4]+'.txt', media_type='application/octet-stream',filename=(name + '.txt'))

        # endpoint websocket
        @self.app.websocket("/ws")
        async def websocket_endpoint(websocket: WebSocket):
            await websocket.accept()
            while True:
                data = await websocket.receive_text()
                data = json.loads(data)
                return_string_1 = ''
                return_string_2 = ''
                return_data = None
                if(data['model']=='vlsp'):
                    return_data = self.VLSP.speech_to_text(data)
                    for i in return_data:
                        await websocket.send_text(str(i))
                elif(data['model']=='250h'):
                    return_data = self.BVM.speech_to_text(data)
                    for i in return_data:
                        await websocket.send_text(str(i))
                else:
                    for i in self.VLSP.speech_to_text(data):
                        return_string_1 += (str(i)+' ')
                    for j in self.BVM.speech_to_text(data):
                        return_string_2 += (str(j)+' ')
                    # self.show_comparison(return_string_1, return_string_2, sidebyside=False)
                    await websocket.send_text(self.show_comparison(return_string_1, return_string_2, sidebyside=False))

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

api = API()

if __name__=='__main__':
    uvicorn.run('api:api.app', host='0.0.0.0', port=9089, reload=True)