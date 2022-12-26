# -*- coding: utf-8 -*-
import json
import os, zipfile
import uvicorn
import shutil
import sqlite3
import datetime
import websockets
import asyncio

from fastapi import FastAPI, Form, File, UploadFile, WebSocket
from fastapi.requests import Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from starlette.responses import FileResponse
from pydantic import BaseModel

from vietnamesemodel import BaseVietnamese_Model
from vlsp2020 import LargeVLSP_Model

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
                return_list_1 = []
                return_list_2 = []
                return_data = []
                if(data['model']=='vlsp'):
                    return_data = self.VLSP.speech_to_text(data)
                    for i in return_data:
                        return_list_1.append([str(i)])
                elif(data['model']=='250h'):
                    return_data = self.BVM.speech_to_text(data)
                    for i in return_data:
                        return_list_2.append([str(i)])
                else:
                    for i in self.VLSP.speech_to_text(data):
                        return_list_1.append([str(i)])
                    for j in self.BVM.speech_to_text(data):
                        return_list_2.append([str(j)])

                return_data.append(return_list_1)
                return_data.append(return_list_2)
                await websocket.send_text(str(return_data))



api = API()

if __name__=='__main__':
    uvicorn.run('api:api.app', host='0.0.0.0', port=9089, reload=True)