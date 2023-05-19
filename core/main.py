from classes import Character
from mic import StartRecording
print("Start")

c = Character("Dane")

idx = 2
while True:
    audio_file_name = StartRecording(idx)
    audio_file = open(audio_file_name, 'rb')
    c.SpeakTo(audio_file)
    input("Press Enter to continue...")


