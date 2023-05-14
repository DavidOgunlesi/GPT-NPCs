from classes import DatabaseModule, SpeechToTextModule, TextToSpeechModule, SpeechConstructionModule
import time
print("Start")

char = DatabaseModule("CHAR")
char.Load("Dane", "characters")
print("Loaded Dane")
#############################################
# HEAR
#############################################
start = time.time()
##
stt = SpeechToTextModule()
t = TextToSpeechModule()

audio_file_path = '1.mp3'
audio_data = open(audio_file_path, 'rb')
i = stt.Transcribe(audio_data)
print(i)
##
end = time.time()
print(f"Time taken to hear: {end - start}")
#############################################
# THINK
#############################################
start = time.time()
##
#filler = ["hmm", "Let me think", "uhhh", "huhhh", "hmmmm"]
#t.Speak("Hmmm. Let me think")
results = char.QueryDatabase(i)
print(results)
##
end = time.time()
print(f"Time taken to think: {end - start}")
#############################################
# SPEAK
#############################################
#start = time.time()
#
t = TextToSpeechModule()
print("Constructing Speech")
s = SpeechConstructionModule("Dane speaks like a stereotypical fantasy bartender. Stoic and jolly. He is a commoner. Can be a bit of a drunk. Potty mouth.")
prev_sentences = ""
genCount = 0
while genCount < 10:
    sres = s.ConstructResponse(results, i, prev_sentences)
    print(sres)
    prev_sentences += sres
    t.Speak(sres)
    genCount += 1
    print(genCount)
    if "NULL" in sres:
        break
# make it so GPT processing happens whilst TTS is speaking
#print(sres)
#
#end = time.time()
#print(f"Time taken to construct speech: {end - start}")
#############################################
# GEN VOICE
#############################################
#start = time.time()
#
#t.Speak(sres)
#
#end = time.time()
#print(f"Time taken to gen voice: {end - start}")
#############################################

