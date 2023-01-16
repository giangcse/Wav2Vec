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
import hashlib
import string, random

from fastapi import FastAPI, Form, File, UploadFile, WebSocket, Header
from fastapi.requests import Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from starlette.responses import FileResponse
from pydantic import BaseModel
from typing import Union

from vietnamesemodel import BaseVietnamese_Model
from vlsp2020 import LargeVLSP_Model

from normalize_text.infer import infer
from vfastpunct import VFastPunct

class Delete_audio(BaseModel):
    token: str
    audio_name: str

class Get_audio(BaseModel):
    token: str

class User(BaseModel):
    username: str
    password: str

class Convert(BaseModel):
    token: str
    audio: str
    denoise: Union[int, 0]
    keyframe: Union[int, 0]
    LM: Union[int, 1]
    model: Union[str, None]

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

        self.punc = VFastPunct(model_name='mBertPunctCap', no_cuda=False)

        @self.app.get("/")
        async def root(request: Request):
            return JSONResponse(status_code=200, content={"success": "Login at /login to continue"})
        # Endpoint get list audio
        @self.app.post("/get_list")
        async def get_list(request: Request, audio: Get_audio):
            res = self.check_token(audio.token)
            if res is not False:
                find = self.cursor.execute("SELECT * FROM audios WHERE username = ?", (str(res), ))
                audios = []
                for i in find.fetchall():
                    audios.append(i[2])
                return JSONResponse(status_code=200, content={"data": audios})
            else:
                return JSONResponse(content={"error": "Please login"})
        # Endpoint xoá audio
        @self.app.post("/delete")
        async def delete(request: Request, body: Delete_audio):
            res = self.check_token(body.token)
            if res is not False:
                deleted = self.cursor.execute("DELETE FROM audios WHERE username = ? AND audio_name = ?", (str(res), str(body.audio_name), ))
                self.connection_db.commit()
                # if self.cursor.execute("SELECT EXISTS (SELECT * FROM audios WHERE username = ? AND  audio_name = ?)", (body.username, body.audio_name, )) == 0:
                return JSONResponse(status_code=200, content={"result": "Xoá thành công"})
            else:
                return JSONResponse(status_code=500, content={"result": "Xoá không thành công"})
        # Enpoint upload audio
        @self.app.post("/upload")
        async def upload(request: Request, file: UploadFile = File (...), token: str = Form(...)):
            res = self.check_token(token)
            if res is not False:

                if not os.path.exists(os.path.join('audio', res)):
                    os.mkdir(os.path.join('audio', res))
                with open(os.path.join('audio', res, file.filename), 'wb') as audio:
                    shutil.copyfileobj(file.file, audio)
                
                find = self.cursor.execute("SELECT EXISTS (SELECT * FROM audios WHERE username = ? AND  audio_name = ?)", (res, os.path.join('audio', res, file.filename), ))
                if find.fetchone()[0] == 0:
                    insert = self.cursor.execute("INSERT INTO audios(username, audio_name, created_at, updated_at) VALUES (?, ?, ?, ?)", (res, os.path.join('audio', res, file.filename), datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
                    self.connection_db.commit()
                else:
                    update = self.cursor.execute("UPDATE audios SET updated_at = ? WHERE username = ? AND audio_name = ?", (datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), res, os.path.join('audio', res, file.filename)))
                    self.connection_db.commit()
                return JSONResponse(status_code=200, content={'audios': [x[0] for x in self.cursor.execute("SELECT audio_name FROM audios WHERE username = ?", (str(res),))]})
            else:
                return JSONResponse(content={"error": "Please login"})
        # Endpoint allow to download audio
        @self.app.post("/download_audio")
        async def download_audio(request: Request, audio: Delete_audio):
            if self.check_token(audio.token) is not False:
                if os.path.exists(str(audio.audio_name)):
                    return FileResponse(audio.audio_name, media_type='application/octet-stream', filename=str(audio.audio_name).split('/')[-1])
                else:
                    return JSONResponse(content={"error": "File not found"})
            else:
                return JSONResponse(content={"error": "Please login"})

        # Endpoint allow to download result
        @self.app.post("/download_text")
        async def download_text(request: Request, audio: Delete_audio):
            if self.check_token(audio.token) is not False:
                name = str(audio.audio_name)[:-4].split('/')[-1]
                if os.path.exists(str(audio.audio_name)[:-4]+'.txt'):
                    return FileResponse(str(audio.audio_name)[:-4]+'.txt', media_type='application/octet-stream',filename=(name + '.txt'))
                else:
                    return JSONResponse(content={"error": "File not found"})
            else:
                return JSONResponse(content={"error": "Please login"})

        # Endpoint login
        @self.app.post("/login")
        async def login(request: Request, info: User):
            token = self.create_token(info.username, info.password)
            if token is None:
                return JSONResponse(content={"error": "Username/Password is incorrect!"})
            else:
                return JSONResponse(content={"status": "Login success", "token": str(token)}, status_code=200)

        # Endpoint register
        @self.app.post("/register")
        async def register(request: Request, info: User):
            username = info.username
            password = info.password
            try:
                insert = self.cursor.execute("INSERT INTO users(username, password) VALUES ('{}', '{}')".format(str(username), hashlib.sha512(password.encode()).hexdigest()))
                self.connection_db.commit()
                return JSONResponse(content={"success": "Created"})
            except sqlite3.IntegrityError:
                return JSONResponse(content={"error": "Username is exist"})
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
                if self.check_token(data['token']) is not False:
                    if os.path.exists(data['audio']):
                        if(data['model']=='vlsp'):
                            return_data = self.VLSP.speech_to_text(data)
                            for i in return_data:
                                await websocket.send_text(str(i)+' ')
                        elif(data['model']=='250h'):
                            return_data = self.BVM.speech_to_text(data)
                            for i in return_data:
                                await websocket.send_text(str(i)+' ')
                        else:
                            for i in self.VLSP.speech_to_text(data):
                                return_string_1 += (str(i)+' ')
                            for j in self.BVM.speech_to_text(data):
                                return_string_2 += (str(j)+' ')
                            last_result = self.punc(self.show_comparison(return_string_1, return_string_2, sidebyside=False))
                            log_file =  open((data['audio'])[:-4] + '.txt', 'w', encoding='utf8')
                            log_file.write(last_result)
                            log_file.close()
                            await websocket.send_text(last_result)
                    else:
                        await websocket.send_text("File not found")
                else:
                    await websocket.send_text("Please login")

        # Endpoint for eCabinet
        @self.app.post("/stt")
        async def speech_to_text(request: Request, body: Convert):
            return_string_1 = ''
            return_string_2 = ''
            return_data = None
            if self.check_token(body['token']) is not False:
                if os.path.exists(body['audio']):
                    if(body['model']=='vlsp'):
                        return_data = self.VLSP.speech_to_text(body)
                    elif(body['model']=='250h'):
                        return_data = self.BVM.speech_to_text(body)
                    else:
                        for i in self.VLSP.speech_to_text(body):
                            return_string_1 += (str(i)+' ')
                        for j in self.BVM.speech_to_text(body):
                            return_string_2 += (str(j)+' ')
                        last_result = self.punc(self.show_comparison(return_string_1, return_string_2, sidebyside=False))
                        log_file =  open((body['audio'])[:-4] + '.txt', 'w', encoding='utf8')
                        log_file.write(last_result)
                        log_file.close()
                        return last_result
                else:
                    return "File not found"
            else:
                return "Please login"
            

    def create_token(self, username, password):
        find = self.cursor.execute("SELECT username FROM users WHERE username='{}' AND password='{}'".format(str(username), hashlib.sha512(password.encode()).hexdigest()))
        token = ''.join(random.choices(string.ascii_lowercase + string.digits, k = 50)) 
        res = find.fetchone()
        if res is None:
            return None
        else:
            update = self.cursor.execute("UPDATE users SET token = ? WHERE username = ?", (str(token), str(username), ))
            self.connection_db.commit()
            return token

    def check_token(self, token):
        find = self.cursor.execute("SELECT username FROM users WHERE token = ?", (str(token), ))
        res = find.fetchone()
        if res is None:
            return False
        else:
            return res[0]

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
    uvicorn.run('api:api.app', host='0.0.0.0', port=9090, reload=True)