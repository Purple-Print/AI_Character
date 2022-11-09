# import mediapipe as mp
# import cv2
from fastapi import FastAPI
# from AI_Character.detect_shape import face_classifi
# from AI_Character.skincolor import *

app = FastAPI()


print(3)
@app.get("/")
async def root():
    return {"message":"hello World"}

@app.get("/hello/{name}")
async def say_hello(name:str):
    return {"message":f"hello {name}"}

# @app.post("/character")
# async def autocharacter():
#     try:
#         face_shape = face_classifi()
#         skin_color = skin_detect()
#     except:
#         print('no file')
#     return {"shape": face_shape,
#             "color": skin_color}