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
import audioread

from fastapi import FastAPI, Form, File, UploadFile, WebSocket, Header, BackgroundTasks
from fastapi.requests import Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from starlette.responses import FileResponse
from starlette import status
from pydantic import BaseModel
from typing import Union

import stt

class Delete_audio(BaseModel):
    token: str
    audio_name: str

class User_token(BaseModel):
    token: str

class User(BaseModel):
    username: str
    password: str

class Convert(BaseModel):
    audio_path: str
    token: str
    enable_lm: int
    key_frame: int

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
        
        # Khoi tao Speech to text
        self.STT = stt.STT()

        @self.app.get("/")
        async def root(request: Request):
            return JSONResponse(status_code=200, content={"success": "Login at /login to continue"})

        # Endpoint register
        @self.app.post("/register")
        async def register(request: Request, info: User):
            username = info.username
            password = info.password
            try:
                insert = self.cursor.execute("INSERT INTO users(username, password) VALUES ('{}', '{}')".format(str(username), hashlib.sha512(password.encode()).hexdigest()))
                self.connection_db.commit()
                return JSONResponse(content={"Success": "Created"}, status_code=status.HTTP_201_CREATED)
            except sqlite3.IntegrityError:
                return JSONResponse(content={"Error": "Username is exist"}, status_code=status.HTTP_409_CONFLICT)

        # Enpoint Login
        @self.app.post("/login")
        async def login(request: Request, user: User):
            username = str(user.username)
            password = str(user.password)
            result = self.create_token(username, password)

            if result=='0':
                return JSONResponse(status_code=status.HTTP_401_UNAUTHORIZED, content={"Error": "Wrong Password"})
            elif result=='-1':
                return JSONResponse(status_code=status.HTTP_401_UNAUTHORIZED, content={"Error": "Username does not exists"})
            else:
                return JSONResponse(status_code=status.HTTP_200_OK, content={"Success": "Login success", "token": result})

        # Endpoint logout
        @self.app.post("/logout")
        async def logout(request: Request, info: User_token):
            result = self.check_token(info.token)
            if(result=='0'):
                return JSONResponse(status_code=status.HTTP_401_UNAUTHORIZED, content={"Error": "Token does not exists"})
            elif(result=='-1'):
                return JSONResponse(status_code=status.HTTP_401_UNAUTHORIZED, content={"Error": "Token expired"})
            else:
                try:
                    update = self.cursor.execute("UPDATE users SET token_exp = ? WHERE token = ?", ((datetime.datetime.now()).strftime("%Y-%m-%d %H:%M:%S"), str(info.token), ))
                    self.connection_db.commit()
                    return JSONResponse(status_code=status.HTTP_200_OK, content={"Success": "Logout success"})
                except Exception:
                    return JSONResponse(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, content={"Error": "Database error"})

        # Endpoint get list audio
        @self.app.post("/get_list")
        async def get_list(request: Request, user: User_token):
            res = self.check_token(user.token)

            if res=='0':
                return JSONResponse(status_code=status.HTTP_401_UNAUTHORIZED, content={"Error": "Token does not exists"})
            elif res=='-1':
                return JSONResponse(status_code=status.HTTP_401_UNAUTHORIZED, content={"Error": "Token expired"})
            else:
                find = self.cursor.execute("SELECT * FROM audios WHERE username = ?", (str(res), ))
                audios = []
                for i in find.fetchall():
                    audios.append(i[2])
                return JSONResponse(status_code=status.HTTP_200_OK, content={"data": audios})

        # Endpoint xoá audio
        @self.app.post("/delete")
        async def delete(request: Request, body: Delete_audio):
            res = self.check_token(body.token)

            if res=='0':
                return JSONResponse(status_code=status.HTTP_401_UNAUTHORIZED, content={"Error": "Token does not exists"})
            elif res=='-1':
                return JSONResponse(status_code=status.HTTP_401_UNAUTHORIZED, content={"Error": "Token expired"})
            else:
                try:
                    deleted = self.cursor.execute("DELETE FROM audios WHERE username = ? AND audio_name = ?", (str(res), str(body.audio_name), ))
                    self.connection_db.commit()
                    # if self.cursor.execute("SELECT EXISTS (SELECT * FROM audios WHERE username = ? AND  audio_name = ?)", (body.username, body.audio_name, )) == 0:
                    return JSONResponse(status_code=status.HTTP_200_OK, content={"Success": "Audio has been deleted"})
                except Exception:
                    return JSONResponse(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, content={"Error": "Internal server error"})

        # Enpoint upload audio
        @self.app.post("/upload")
        async def upload(request: Request, file: UploadFile = File (...), token: str = Form(...)):
            res = self.check_token(token)

            if res=='0':
                return JSONResponse(status_code=status.HTTP_401_UNAUTHORIZED, content={"Error": "Token does not exists"})
            elif res=='-1':
                return JSONResponse(status_code=status.HTTP_401_UNAUTHORIZED, content={"Error": "Token expired"})
            else:
                try:
                    if not os.path.exists(os.path.join('audio', res)):
                        os.mkdir(os.path.join('audio', res))
                    with open(os.path.join('audio', res, file.filename), 'wb') as audio:
                        shutil.copyfileobj(file.file, audio)
                    audio_length = self.get_audio_length(os.path.join('audio', res, file.filename))
                    if audio_length != (-1):
                        find = self.cursor.execute("SELECT EXISTS (SELECT * FROM audios WHERE username = ? AND  audio_name = ?)", (res, os.path.join('audio', res, file.filename), ))
                        if find.fetchone()[0] == 0:
                            insert = self.cursor.execute("INSERT INTO audios(username, audio_name, audio_length, created_at, updated_at) VALUES (?, ?, ?, ?, ?)", (res, os.path.join('audio', res, file.filename), int(audio_length), datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
                            self.connection_db.commit()
                        else:
                            update = self.cursor.execute("UPDATE audios SET updated_at = ? WHERE username = ? AND audio_name = ?", (datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), res, os.path.join('audio', res, file.filename)))
                            self.connection_db.commit()
                        return JSONResponse(status_code=status.HTTP_200_OK, content={"Success": "Uploaded", 'Audios': [x[0] for x in self.cursor.execute("SELECT audio_name FROM audios WHERE username = ?", (str(res),))]})
                    else:
                        return JSONResponse(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, content={"Error": "File does not support"})
                except Exception:
                    return JSONResponse(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, content={"Error": "Internal server error"})

        # Endpoint allow to download audio
        @self.app.post("/download_audio")
        async def download_audio(request: Request, audio: Delete_audio):
            res = self.check_token(audio.token)

            if res=='0':
                return JSONResponse(status_code=status.HTTP_401_UNAUTHORIZED, content={"Error": "Token does not exists"})
            elif res=='-1':
                return JSONResponse(status_code=status.HTTP_401_UNAUTHORIZED, content={"Error": "Token expired"})
            else:
                if os.path.exists(str(audio.audio_name)) and (res in str(audio.audio_name)):
                    return FileResponse(audio.audio_name, media_type='application/octet-stream', filename=str(audio.audio_name).split('/')[-1])
                else:
                    return JSONResponse(content={"error": "Audio file not found"}, status_code=status.HTTP_404_NOT_FOUND)

        # Endpoint allow to download result
        @self.app.post("/download_text")
        async def download_text(request: Request, audio: Delete_audio):
            res = self.check_token(audio.token)

            if res=='0':
                return JSONResponse(status_code=status.HTTP_401_UNAUTHORIZED, content={"Error": "Token does not exists"})
            elif res=='-1':
                return JSONResponse(status_code=status.HTTP_401_UNAUTHORIZED, content={"Error": "Token expired"})
            else:
                select = self.cursor.execute("SELECT content, updated_at FROM audios WHERE username = ? AND audio_name = ?", (str(res), str(audio.audio_name), ))
                result = select.fetchone()
                if result is not None:
                    return JSONResponse(content={"Text": result[0], "Time": result[1]}, status_code=status.HTTP_200_OK)
                else:
                    return JSONResponse(content={"Error": "Text not found"}, status_code=status.HTTP_404_NOT_FOUND)
            
                
        # Endpoint check token exp
        @self.app.post("/check_token")
        async def check_token_endpoint(request: Request, info: User_token):
            res = self.check_token(info.token)
            if res=='0':
                return JSONResponse(status_code=status.HTTP_401_UNAUTHORIZED, content={"Error": "Token does not exists"})
            elif res=='-1':
                return JSONResponse(status_code=status.HTTP_401_UNAUTHORIZED, content={"Error": "Token expired"})
            else:
                return JSONResponse(status_code=status.HTTP_200_OK, content={"Success": "Token is vailid"})

        # Endpoint get detail
        @self.app.post("/get_detail")
        async def get_detail(request: Request, info: User_token):
            res = self.check_token(info.token)
            if res=='0':
                return JSONResponse(status_code=status.HTTP_401_UNAUTHORIZED, content={"Error": "Token does not exists"})
            elif res=='-1':
                return JSONResponse(status_code=status.HTTP_401_UNAUTHORIZED, content={"Error": "Token expired"})
            else:
                # Get total word
                _word_count_query = self.cursor.execute("SELECT (length(content) - length(replace(content, ' ', ''))) + 1 AS word_count FROM audios WHERE username = ?", (res, ))
                _word_count_result = _word_count_query.fetchall()
                _total_word = 0
                for i in _word_count_result:
                    _total_word += int(i[0])
                # Get total length
                _duration_audio_query = self.cursor.execute("SELECT (length(content) - length(replace(content, ' ', ''))) + 1 AS word_count FROM audios WHERE content IS NOT NULL AND username = ?", (res, ))
                _duration_audio_result = _duration_audio_query.fetchall()
                _total_length = 0
                for i in _duration_audio_result:
                    _total_length += int(i[0])
                return JSONResponse(status_code=status.HTTP_200_OK, content={"Total": {"Words": _total_word, "Duration": _total_length}})

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

        # Endpoint convert Speech to text
        @self.app.post("/convert")
        async def convert_speech_to_text(request: Request, background_tasks: BackgroundTasks, info: Convert):
            res = self.check_token(info.token)

            if res=='0':
                return JSONResponse(status_code=status.HTTP_401_UNAUTHORIZED, content={"Error": "Token does not exists"})
            elif res=='-1':
                return JSONResponse(status_code=status.HTTP_401_UNAUTHORIZED, content={"Error": "Token expired"})
            else:
                _audio_exist = self.cursor.execute("SELECT EXISTS(SELECT 1 FROM audios WHERE audio_name = ?)", (os.path.join('audio', res, info.audio_path), ))
                _result_audio_exist = _audio_exist.fetchone()

                if int(_result_audio_exist[0]) > 0:
                    background_tasks.add_task(self.convert_stt, os.path.join('audio', res, info.audio_path), info.enable_lm, info.key_frame)
                    return JSONResponse(content={"Success": "Please wait"}, status_code=status.HTTP_200_OK)
                else:
                    return JSONResponse(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, content={"Error": "File audio not found", "Result": _result_audio_exist})



        # Endpoint for eCabinet
        @self.app.post("/stt")
        async def upload(request: Request, file: UploadFile = File (...), token: str = Form(...), enable_lm: int = Form(...), keyframe: int = Form(...), model: str = Form(...)):
            res = self.check_token(token)

            if res=='0':
                return JSONResponse(status_code=status.HTTP_401_UNAUTHORIZED, content={"Error": "Token does not exists"})
            elif res=='-1':
                return JSONResponse(status_code=status.HTTP_401_UNAUTHORIZED, content={"Error": "Token expired"})
            else:
                    if not os.path.exists(os.path.join('audio', res)):
                        os.mkdir(os.path.join('audio', res))
                    with open(os.path.join('audio', res, file.filename), 'wb') as audio:
                        shutil.copyfileobj(file.file, audio)
                    
                    audio_length = self.get_audio_length(os.path.join('audio', res, file.filename))
                    if audio_length != (-1):
                        find = self.cursor.execute("SELECT EXISTS (SELECT * FROM audios WHERE username = ? AND  audio_name = ?)", (res, os.path.join('audio', res, file.filename), ))
                        if find.fetchone()[0] == 0:
                            insert = self.cursor.execute("INSERT INTO audios(username, audio_name, audio_length, created_at, updated_at) VALUES (?, ?, ?, ?, ?)", (res, os.path.join('audio', res, file.filename), int(audio_length), datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
                            self.connection_db.commit()
                        else:
                            update = self.cursor.execute("UPDATE audios SET updated_at = ? WHERE username = ? AND audio_name = ?", (datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), res, os.path.join('audio', res, file.filename)))
                            self.connection_db.commit()

                        # upload is done, convert speech to text phase
                        return_string_1 = ''
                        return_string_2 = ''
                        return_data = ''
                        data = {'audio': os.path.join('audio', res, file.filename), 'keyframe': int(keyframe), 'LM': int(enable_lm)}

                        return_data = self.STT.convert(file_path=os.path.join('audio', res, file.filename), key_frame=int(keyframe), enable_lm=int(enable_lm), model='')
                        update_result = self.cursor.execute("UPDATE audios SET content = ?, updated_at = ? WHERE username = ? AND audio_name = ?", (return_data, datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), res, os.path.join('audio', res, file.filename)))
                        self.connection_db.commit()
                        return JSONResponse(status_code=status.HTTP_200_OK, content={"Result": return_data})
                    else:
                        return JSONResponse(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, content={"Error": "File does not support"})

    def convert_stt(self, audio_path, enable_lm, key_frame):
        result = self.STT.convert(audio_path, enable_lm, key_frame, '')
        update_result = self.cursor.execute("UPDATE audios SET content = ?, updated_at = ? WHERE audio_name = ?", (result, datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), audio_path))
        self.connection_db.commit()
        return JSONResponse(content={"Result": result}, status_code=status.HTTP_200_OK) 

    def create_token(self, username, password):
        '''
        CREATE TOKEN FUNC.
        -----
        - input: username (str): Tên người dùng; password (str): Mật khẩu

        - output (str): token: Thành công; 0: Username không tồn tại; -1: Lỗi database
        '''
        find = self.cursor.execute("SELECT username FROM users WHERE username='{}' AND password='{}'".format(str(username), hashlib.sha512(password.encode()).hexdigest()))
        token = ''.join(random.choices(string.ascii_lowercase + string.digits, k = 50)) 
        res = find.fetchone()
        if res is None:
            return str(0)
        else:
            try: 
                update = self.cursor.execute("UPDATE users SET token = ?, token_exp = ? WHERE username = ?", (str(token), (datetime.datetime.now() + datetime.timedelta(hours=24)).strftime("%Y-%m-%d %H:%M:%S"), str(username), ))
                self.connection_db.commit()
                return token
            except Exception:
                return str(-1)
                
    def check_token(self, token):
        '''
        CHECK TOKEN FUNC.
        -----------------------------
        - input (str): token: Token cần check được cung cấp từ hàm login
        
        - ouput (str): username: thành công, -1: Token hết hạn, 0: Token không tồn tại 
        '''
        if token!='':
            find = self.cursor.execute("SELECT * FROM users WHERE token = ?", (str(token), ))
            res = find.fetchall()
            if res is None:
                return str(0)
            else:
                try:
                    now = datetime.datetime.now()
                    exp = datetime.datetime.strptime(res[0][3], "%Y-%m-%d %H:%M:%S")
                    est = exp - now
                    if(est.total_seconds() > 0):
                        return res[0][0]
                    else:
                        return str(-1)
                except Exception:
                    return str(-1)
        else:
            return str(-1)
            
    # Get duration of audio
    def get_audio_length(self, audio_path):
        '''
        GET AUDIO LENGTH FUNC.
        -----
        - input (str): audio path
        
        - output (float): độ dài audio (s) nếu đúng, nếu không đọc được thì return lỗi là str(-1)
        '''
        try:
            with audioread.audio_open(audio_path) as f:
                return (f.duration)
        except Exception:
            return (-1)
        
api = API()

if __name__=='__main__':
    uvicorn.run('api:api.app', host='0.0.0.0', port=9090, reload=True)