import cv2
import pyaudio
import wave
import threading
import time
import subprocess
import os
import argparse

name= "your_name"
global video_thread
global audio_thread

class VideoRecorder():  
    def __init__(self):
        self.open = True
        self.device_index = 0
        self.fps = 6      
        self.fourcc = "MJPG" 
        self.frameSize = (640,480) 
        self.video_filename = "path_to_save/" + name + ".avi"
        self.video_cap = cv2.VideoCapture(0)
        self.video_writer = cv2.VideoWriter_fourcc(*'XVID')
        self.video_out = cv2.VideoWriter(self.video_filename, self.video_writer, self.fps, self.frameSize)
        self.frame_counts = 1
        self.start_time = time.time()


    def record(self):
        timer_start = time.time()
        timer_current = 0
        while(self.open==True):
            ret, video_frame = self.video_cap.read()
            if (ret==True):
                    self.video_out.write(video_frame)
                    self.frame_counts += 1
                    time.sleep(0.16)
            else:
                break

    def stop(self):
        if self.open==True:
            self.open=False
            self.video_out.release()
            self.video_cap.release()
            cv2.destroyAllWindows()
        else: 
            pass

    def start(self):
        video_thread = threading.Thread(target=self.record)
        video_thread.start()


class AudioRecorder():
    def __init__(self):
        self.open = True
        self.rate = 32000
        self.frames_per_buffer = 32000
        self.channels = 2
        self.format = pyaudio.paInt16
        self.audio_filename = "path_to_save/" + name + ".wav"
        self.audio = pyaudio.PyAudio()
        self.stream = self.audio.open(format=self.format,
                                      channels=self.channels,
                                      rate=self.rate,
                                      input_device_index=11,
                                      frames_per_buffer = self.frames_per_buffer,
                                      input=True)
        self.audio_frames = []


    def record(self):
        self.stream.start_stream()
        while(self.open == True):
            data = self.stream.read(self.frames_per_buffer)
            self.audio_frames.append(data)
            if self.open==False:
                break


    def stop(self):
        if self.open==True:
            self.open = False
            self.stream.stop_stream()
            self.stream.close()
            self.audio.terminate()
            waveFile = wave.open(self.audio_filename, 'wb')
            waveFile.setnchannels(self.channels)
            waveFile.setsampwidth(self.audio.get_sample_size(self.format))
            waveFile.setframerate(self.rate)
            waveFile.writeframes(b''.join(self.audio_frames))
            waveFile.close()
        pass

    def start(self):
        audio_thread = threading.Thread(target=self.record)
        audio_thread.start()




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--record_audio', action="store_true")
    parser.add_argument("--record_video", action="store_true")
    parser.add_argument('--record_video_and_audio', action="store_true")
    parser.add_argument('--time', type=int, default=0, help='Time in seconds to record')
    args = parser.parse_args()
    if args.record_audio:
        audio_thread = AudioRecorder()
        audio_thread.start()
        print("Recording...")
        time.sleep(args.time)
        audio_thread.stop()
    else:
        if args.record_video:
            video_thread = VideoRecorder()
            video_thread.start()
            print("Recording...")
            time.sleep(args.time)
            video_thread.stop()
        else:
            if args.record_video_and_audio:
                video_thread = VideoRecorder()
                audio_thread = AudioRecorder()
                audio_thread.start()
                video_thread.start()
                print("Recording...")
                time.sleep(args.time)
                video_thread.stop()
                audio_thread.stop()
    while threading.active_count() > 1:
        time.sleep(1)
