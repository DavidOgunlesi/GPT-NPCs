import io
import pyaudio
import keyboard
import wave

# set the parameters for the audio recording
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
CHUNK = 1024

def ChooseDeviceIndex():
    # create an instance of the Pyaudio class
    audio = pyaudio.PyAudio()

    # get a list of available audio devices and their indices
    for i in range(audio.get_device_count()):
        device = audio.get_device_info_by_index(i)
        print(f"{i}: {device['name']}")

    # select a device index to use for recording
    return int(input("Enter device index: "))

def StartRecording(device_index=0):
    # create an instance of the Pyaudio class
    audio = pyaudio.PyAudio()

    # open the microphone and start recording
    stream = audio.open(format=FORMAT, channels=CHANNELS,
                        rate=RATE, input=True,
                        frames_per_buffer=CHUNK,
                        input_device_index=device_index)

    frames = []

    print("Recording started. Press 's' to stop recording.")
    # loop until the user presses the 's' key to stop recording
    while True:
        data = stream.read(CHUNK)
        frames.append(data)
        if keyboard.is_pressed('s'):
            break

    # stop the audio stream and close the Pyaudio instance
    stream.stop_stream()
    stream.close()
    audio.terminate()

    #save the recorded audio to a WAV file
    wf = wave.open("temp.wav", 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(audio.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b''.join(frames))
    wf.close()

    # save the recorded audio to a byte object
    # with io.BytesIO() as buffer:
    #     with wave.open(buffer, 'wb') as wf:
    #         wf.setnchannels(CHANNELS)
    #         wf.setsampwidth(audio.get_sample_size(FORMAT))
    #         wf.setframerate(RATE)
    #         wf.writeframes(b''.join(frames))
    #     audio_bytes = buffer.getvalue()

    print("Recording stopped.")
    return "temp.wav"
