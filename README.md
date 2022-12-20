# Công cụ chuyển giọng nói thành văn bản (Speech2Text)

Cài đặt trên ubuntu 20.04 LTS, môi trường Anaconda Python 3.7
Cài thêm ffmpeg để cắt audio: sudo apt install -y ffmpeg
Sử dụng model Wav2Vec để chuyển giọng nói thành văn bản. Ứng dụng trong cuộc họp.

> **Khuyến nghị:** Sử dụng  **Ubuntu 20.04** và tạo môi trường ảo với **Anaconda**.


## Đánh giá

Đánh giá mô hình:

|                |With  ngrams                   |With out ngrams              |
|----------------|-------------------------------|-----------------------------|
|abc.wav         |`91%`				             |87%                          |
|def.mp3         |`95%`            				 |86%				           |

## Hướng dẫn cài đặt
`sudo setup.sh`