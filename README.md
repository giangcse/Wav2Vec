# Công cụ chuyển giọng nói thành văn bản (Speech2Text)

Cài đặt trên ubuntu 20.04 LTS, python 3.7
Sử dụng model Wav2Vec để chuyển giọng nói thành văn bản. Ứng dụng trong cuộc họp.

> **Khuyến nghị:** Sử dụng  **Ubuntu 20.04**.


## Đánh giá

Đánh giá mô hình:

|Model           |With  ngrams                   |With out ngrams              |
|----------------|-------------------------------|-----------------------------|
|wav2vec2-large-vi-vlsp2020         |`95%`				             |87%                          |
|wav2vec2-base-vietnamese-250h        |`91%`            				 |86%				           |

## Hướng dẫn cài đặt
`sudo setup.sh`

## Cần chỉnh sửa các file sau
> **py37/lib/python3.7/site-packages/transformers/models/wav2vec2_with_lm/processing_wav2vec2_with_lm.py**

Sửa các dòng 31, 85, 139, 188 từ

`pyctcdecode`

thành

`py37.pyctcdecode4.pyctcdecode`

> **py37/lib/python3.7/site-packages/vfastpunct/constants.py**

Sửa dòng 59 từ

`BASE_PATH = 'cache/'`

thành

`BASE_PATH = 'models/'`