from flask import Flask, render_template, request, jsonify
import speech_recognition as sr
import whisper
import os
import tempfile
import threading
from datetime import datetime
import uuid
import io
from gtts import gTTS
from playsound import playsound
from train_scroing_model import training_model
from pydub import AudioSegment
from pydub.playback import play
import torch

app = Flask(__name__)

# Khởi tạo các components
r = sr.Recognizer()
model = whisper.load_model("base")
os.environ["PATH"] += os.pathsep + "C:\\Users\\Admin\\Downloads\\ffmpeg-7.1.1-essentials_build\\ffmpeg-7.1.1-essentials_build\\bin"

# Biến global để quản lý trạng thái
is_recording = False
accumulated_audio_data = []
recording_thread = None
lock = threading.Lock()

def record_audio():
    global is_recording, accumulated_audio_data
    
    try:
        print("Đang nghe... Nói gì đó!")
        with sr.Microphone() as source2:
            r.adjust_for_ambient_noise(source2, duration=1.0)
            r.pause_threshold = 1.5
            r.energy_threshold = 300
            print("Đã sẵn sàng, bắt đầu nói...")
            
            while is_recording:
                try:
                    audio_chunk = r.listen(source2, timeout=1, phrase_time_limit=3)
                    
                    with lock:
                        if is_recording: 
                            accumulated_audio_data.append(audio_chunk.get_wav_data())
                            print("Đã thu được một đoạn audio")
                except sr.WaitTimeoutError:
                    continue
                except Exception as e:
                    print(f"Lỗi khi ghi âm: {e}")
                    break
                    
    except Exception as e:
        print(f"Lỗi không xác định trong record_audio: {e}")

def combine_audio_chunks(audio_chunks):
    if not audio_chunks:
        return None
    record_input_dir = "recordInput"
    if not os.path.exists(record_input_dir):
        os.makedirs(record_input_dir)
        print(f"Đã tạo thư mục: {record_input_dir}")
        
    try:
        print("Dang ket hop {} chunks audio".format(len(audio_chunks)))
        combined = AudioSegment.empty()
        for i, chunk in enumerate(audio_chunks):
            try:
                audio_segment  = AudioSegment.from_wav(io.BytesIO(chunk))
                combined += audio_segment
                print(f"Đã kết hợp chunk {i+1}/{len(audio_chunks)}")
            except Exception as e:
                 print(f"Lỗi khi xử lý chunk {i}: {e}")
                 continue
        if len(combined)>0:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"recording_{timestamp}_{uuid.uuid4().hex[:8]}.wav"
            filepath = os.path.join(record_input_dir, filename)
            combined.export(filepath, format="wav")
        else:
            print("Không có audio data hợp lệ để kết hợp")
            return None
        return filepath
        
    except Exception as e:
        print(f"Lỗi khi kết hợp audio: {e}")
        return None

def transcribe_audio(audio_filename):
    """Chuyển audio thành text"""
    try:
        print("Đang xử lý audio...")
        result = model.transcribe(audio_filename)
        text = result["text"].strip()
        return text
        
    except Exception as e:
        print(f"Lỗi khi transcribe: {e}")
        return None

def save_text(text):
    """Lưu text vào file"""
    try:
        with open("output.txt", "a", encoding="utf-8") as f:
            f.write(text)
            f.write("\n")
        print(f"Đã ghi: {text}")
        return True
    except Exception as e:
        print(f"Lỗi khi ghi file: {e}")
        return False

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/text_to_speech', methods=['POST'])
def change_text_to_speech():
    try:
        text = request.json.get("text", "")
        if not text:
            return jsonify({"error": "No text provided"}), 400            
        
        if not os.path.exists('static'):
            os.makedirs('static')        
        
        import uuid
        filename = f"static/output_{uuid.uuid4().hex[:8]}.mp3"
        tts = gTTS(text, lang='en', slow=False) 
        tts.save(filename)
        
        if not os.path.exists(filename):
            return jsonify({"error": "Failed to create audio file"}), 500
            
        print(f"Audio file created: {filename}")        
        
        return jsonify({
            "status": "success", 
            "audio_url": f"/{filename}",
            "message": "Audio created successfully"
        })
        
    except Exception as e:
        print(f"Error in text_to_speech: {e}")
        return jsonify({"error": f"TTS Error: {str(e)}"}), 500
@app.route('/scoring', methods=['POST'])
def scoring():
    model = torch.load('D:\\AI_assistance\\best_pronunciation_model_with_word_numbers.pth')
    try:
        record = combine_audio_chunks()
        word_count, accuracy = model.prediction(record)
        return word_count, accuracy
    except Exception as e:
        print(f"Error: {e}")
        
        
@app.route('/start_recording', methods=['POST'])
def start_recording():
    global is_recording, recording_thread, accumulated_audio_data
    
    if not is_recording:
        is_recording = True
        accumulated_audio_data = []
        recording_thread = threading.Thread(target=record_audio)
        recording_thread.daemon = True
        recording_thread.start()
        combine_audio_chunks()

        print("Bắt đầu ghi âm liên tục...")
        return jsonify({
            "status": "success", 
            "message": "Bắt đầu ghi âm liên tục",
            "timestamp": datetime.now().isoformat()
        })    
    else:
        return jsonify({"status": "error", "message": "Đang ghi âm rồi"})

@app.route('/stop_recording', methods=['POST'])
def stop_recording():
    global is_recording, accumulated_audio_data, recording_thread
    if not is_recording:
        return jsonify({
            "status": "error", 
            "message": "Không có ghi âm nào đang chạy"
        })
    print("Stopping record")
    is_recording = False
    if recording_thread and recording_thread.is_alive():
        recording_thread.join(timeout=10)
        if recording_thread.is_alive():
             print("Warning: Recording thread không thoát trong thời gian cho phép")
    with lock:
        if not accumulated_audio_data:
            return jsonify({
                "status": "error",
                "message": "Khong co du lieu nao duoc thu"
            })
        print("Da thu duoc {} doan audio".format(len(accumulated_audio_data)))
        combined_audio_file = combine_audio_chunks(accumulated_audio_data)
        if not combined_audio_file:
            return jsonify({
                "status": "error",
                "message": "Khong the ket hop audio"
            })
        text = transcribe_audio(combined_audio_file)
        if not text:
            return jsonify({
                "status": "error",
                "message":"Khong nhan dien duoc giong noi",
                "chunks_count": len(accumulated_audio_data)
            })
        if save_text(text):
            accumulated_audio_data = []
            
            return jsonify({
                "status": "success", 
                "message": "Dừng ghi âm và lưu thành công",
                "text": text,
                "chunks_count": len(accumulated_audio_data),
                "audio_file": combined_audio_file,
                "timestamp": datetime.now().isoformat()
            })
        else:
            return jsonify({
                "status": "partial_success", 
                "message": "Transcribe thành công nhưng lưu file thất bại",
                "text": text,
                "chunks_count": len(accumulated_audio_data)
            })
@app.route('/get_status')
def get_status():
    with lock:
        chunks_count = len(accumulated_audio_data) if accumulated_audio_data else 0
    return jsonify({
        "is_recording": is_recording,
        "chunks_recorded": chunks_count
    })


if __name__ == '__main__':
    app.run(debug=True, port=5000)