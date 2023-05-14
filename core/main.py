from classes import Character
from mic import StartRecording, ChooseDeviceIndex
print("Start")

c = Character("Dane")

idx = ChooseDeviceIndex()
#audio_file_path = '1.mp3'
while True:
    audio_file_name = StartRecording(idx) #open(audio_file_path, 'rb')
    audio_file = open(audio_file_name, 'rb')
    c.SpeakTo(audio_file)
    input("Press Enter to continue...")


