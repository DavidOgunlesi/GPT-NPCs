from __future__ import annotations
import openai
from elevenlabs import generate, play, set_api_key
import numpy as np
import json
import os
import time
import threading

from keys import OPENAI_KEY, ELEVENLABS_KEY
openai.api_key = OPENAI_KEY

class ListSerializer:
    @staticmethod
    def serialize(lst, filepath):
        with open(filepath, 'w') as f:
            json.dump(lst, f)
    
    @staticmethod
    def deserialize(filepath):
        with open(filepath, 'r') as f:
            lst = json.load(f)
        return lst


class GPTInstance:
    class Model:
        CHATGPT = "gpt-3.5-turbo"
        CURIE = "text-curie-001"

    def __init__(self, prompt) -> None:
        self.prompt = prompt
        self.conversation = []

    def CreateChat(self, confirmString="OK", message = None) -> str:
        self.conversation = [{"role": "system", "content": self.prompt}]
        chat_completion = openai.ChatCompletion.create(model=self.Model.CHATGPT, messages=self.conversation)
        response = chat_completion.choices[0].message.content
        self.conversation.append({"role": "assistant", "content": response})
        if confirmString != None and response != confirmString:
            raise Exception("Conversation failed to initialize")
        
        if message != None:
            self.conversation.append({"role": "user", "content": message})
            chat_completion = openai.ChatCompletion.create(model=self.Model.CHATGPT, messages=self.conversation)
            response = chat_completion.choices[0].message.content
            self.conversation.append({"role": "assistant", "content": response})

        return response
    
    def CreateCompletion(self) -> str:
        response = openai.Completion.create(model=self.Model.CURIE, prompt=self.prompt)
        return response.choices[0].text

class DatabaseModule:
    def __init__(self, name, speech_personality = "", initState = []) -> None:
        self.embeddings = []
        self.entries = []
        self.speech_personality = speech_personality

        for entry in initState:
            self.AddEmbedding(entry)

        print(f"Initialised {name}")
    

    def AddEmbedding(self, sentence):
        # Encode sentences into vectors using OpenAI's text-embedding-ada-002 model
        response = openai.Embedding.create(
                input=sentence,
                model="text-embedding-ada-002"
            )
        self.embeddings.append(response['data'][0]['embedding'])
        self.entries.append(sentence)

    def QueryDatabase(self, query):
        # Encode query into vector using OpenAI's text-embedding-ada-002 model
        response = openai.Embedding.create(
            input=query,
            model="text-embedding-ada-002"
        )
        query_embedding = response['data'][0]['embedding']
        query_embedding = np.array(query_embedding)

        embeddings = np.array(self.embeddings)
        # Calculate cosine similarity between query vector and sentence vectors
        similarities = np.dot(embeddings, query_embedding) / (np.linalg.norm(embeddings, axis=1) * np.linalg.norm(query_embedding))

        # Get indices of top N most similar sentences
        N = 5  # number of similar sentences to return
        top_N_indices = np.argsort(similarities)[::-1][:N]

        # Get top N most similar sentences
        top_N_sentences = [self.entries[i] for i in top_N_indices]

        return top_N_sentences

    def Save(self, characterName, dirPath):
        # If the directory doesn't exist
        if not os.path.exists(dirPath):
            # Create a directory to store the database
            os.mkdir(dirPath)
        # Save the entries to a JSON file
        ListSerializer.serialize(self.entries, os.path.join(dirPath, f'{characterName}.entries'))
        # Save the embeddings to a JSON file
        ListSerializer.serialize(self.embeddings, os.path.join(dirPath, f'{characterName}.embeddings'))
        # Save the speech personality to a JSON file
        ListSerializer.serialize(self.speech_personality, os.path.join(dirPath, f'{characterName}.speech_personality'))

    def Load(self, characterName, dirPath):
        # Load the entries from a JSON file
        self.entries = ListSerializer.deserialize(os.path.join(dirPath, f'{characterName}.entries'))
        # Load the embeddings from a JSON file
        self.embeddings = ListSerializer.deserialize(os.path.join(dirPath, f'{characterName}.embeddings'))
        # Load the speech personality from a JSON file
        self.speech_personality = ListSerializer.deserialize(os.path.join(dirPath, f'{characterName}.speech_personality'))

class SpeechConstructionModule:
    def __init__(self, speech_personality) -> None:
        self.speech_personality = speech_personality

    def ConstructResponse(self, initial_queries, originalInput, db, previousSentences, conversation = []):
        # prompt =  f'''
        # INSTRUCTIONS:
        # You are roleplaying as a character. Given the information from your thoughts, which have been retrieved from your thoughts, about you and/or the world, construct an in-character response to the original input.        
        # Only construct the next sentence. Do not construct the entire response. Do this by predicting the next sentence from the previous sentences. If no previous sentences exist, construct the first sentence.
        # If your previous sentences make a complete answer/response, you should stop generating.

        # Make it sound like a human.
        # Speech Personality: {self.speech_personality}
        # Do not say anything else. Only say the response. And only one sentence. Do not repeat things.

        # DATA:
        # Current Conversation History: {conversation}
        # Current Input to respond to: {originalInput}
        # Your Previous Sentences: {"".join(previousSentences)}
        # Your retrieved Thoughts: {queries}

        # EXAMPLE: 
        # Current Conversation History: ["role": "system", "content": "You are under arrest!"]
        # Current Input to respond to: "You are under arrest!"
        # Your Previous Sentences: Who are you? I don't recognise you around these parts.
        # Your retrieved Thoughts: ["Do I know this person?", "Are they new here?"]
        # Output Example: Are you new here? NULL.

        # When you are finished predicting sentences say "NULL".
        # '''
        # prompt=f'''
        # INSTRUCTIONS:
        # You are roleplaying as a character. Given the information from your thoughts, which have been retrieved from your thoughts, about you and/or the world, construct an in-character response to the original input.
        # Make it sound like a human.
        # Speech Personality: {self.speech_personality}

        # DATA:
        # Current Conversation History: {conversation}

        # Your retrieved Thoughts: {queries}
        # '''
        prompt = open("core/prompts/speech_construction.txt", "r").read()
        previousSentencesString = "\n".join(previousSentences).replace('"', '')

        # Think about that you are saying too
        thoughts = initial_queries if len(previousSentences) == 0 else db.QueryDatabase(previousSentencesString) + initial_queries
        thoughts = "\n".join(thoughts)

        def CleanString(string):
            return string.replace('"', '')
        
        convoString = "\n".join([f'{x["role"]}: "{CleanString(x["content"])}"' for i, x in enumerate(conversation)])
        previousSentencesString = "\n".join([f'{CleanString(s)}' for i, s in enumerate(previousSentences)])

        prompt = prompt.format(
            speech_personality= self.speech_personality, 
            conversation= convoString, 
            thoughts= thoughts,
            sequence= previousSentencesString,
            current_input = originalInput
            )
        #print("".join(previousSentences).replace("\"", ""))
        #print("------------------------------------------------------------------------------------------------------------------------")
        #print(prompt)
        #print("------------------------------------------------------------------------------------------------------------------------")
        gpt = GPTInstance(prompt)
        return gpt.CreateChat(confirmString=None)

class SpeechToTextModule:
    def Transcribe(self, audio_file) -> None:
        transcript = openai.Audio.transcribe("whisper-1", audio_file)
        return transcript['text']

class TextToSpeechModule:
    def Speak(self, text) -> None:
        #set_api_key(ELEVENLABS_KEY)

        audio = generate(
        text=text,
        voice="f5UejLGNKubShJkTy5UO",
        model="eleven_monolingual_v1"
        )
        
        play(audio)

    def SpeakBytes(self, data) -> None:
        #set_api_key(ELEVENLABS_KEY)
        play(data)

    def GenerateAudioBytes(self, text) -> bytes:
        #set_api_key(ELEVENLABS_KEY)

        audio = generate(
        text=text,
        #voice="f5UejLGNKubShJkTy5UO",
        voice="Adam",
        model="eleven_monolingual_v1"
        )
        
        return audio
    
    # def Speak_async(self, text):
    #     process = mp.Process(target=self.Speak, args=(text,))
    #     process.start()

class Character:
    def __init__(self, name) -> None:
        self.name = name
        self.conversation = []

    def SpeakTo(self, audio_file):
        char = DatabaseModule("CHAR")
        char.Load(self.name, "characters")
        print("Loaded Dane")

        #############################################
        # HEAR
        #############################################
        stt = SpeechToTextModule()
        tts = TextToSpeechModule()
        inp_text = stt.Transcribe(audio_file)
        print(f"You said: {inp_text}")
        self.conversation.append({"role": "other", "content": inp_text})
        #############################################
        # THINK
        #############################################
        results = char.QueryDatabase(inp_text)
        print(f"Thoughts: {results}")
        #############################################
        # SPEAK
        #############################################
        tts = TextToSpeechModule()
        s = SpeechConstructionModule(char.speech_personality)
        prev_sentences = []
        genCount = 0

        audioBytesBuffer = []

        speaking = True
        audio_playing = False
        def speak():
            global audio_playing
            while speaking:
                if len(audioBytesBuffer) > 0:
                    data = audioBytesBuffer.pop(0)
                    audio_playing = True
                    tts.SpeakBytes(data)
                    audio_playing = False

        thread = threading.Thread(target=speak)
        thread.start()
        fullText = ""
        maxSentences = 10
        while genCount < maxSentences:
            #print(prev_sentences)
            sres = s.ConstructResponse(results, inp_text, char, prev_sentences, conversation=self.conversation)
            print(f"Response: {sres}")
            speechText = sres.replace("NULL", "")
            speechText = speechText.replace("CONTINUE", "")
            fullText += speechText
            #print(f"Speech: {speechText}")
            prev_sentences.append(sres)
            genbytes = tts.GenerateAudioBytes(speechText)
            audioBytesBuffer.append(genbytes)
            genCount += 1
            if "NULL" in sres:
                print("NULL")
                break
        
        while audio_playing or audioBytesBuffer != []:
            time.sleep(0.1)
        speaking = False

        self.conversation.append({"role": "you", "content": fullText})