from __future__ import annotations
import openai
from elevenlabs import generate, play, set_api_key
import numpy as np
import json
import os
import multiprocessing as mp
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
    def __init__(self, name, initState = []) -> None:
        self.embeddings = []
        self.entries = []

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
        # Create a directory to store the database
        os.mkdir(dirPath)
        # Save the entries to a JSON file
        ListSerializer.serialize(self.entries, os.path.join(dirPath, f'{characterName}.entries'))
        # Save the embeddings to a JSON file
        ListSerializer.serialize(self.embeddings, os.path.join(dirPath, f'{characterName}.embeddings'))

    def Load(self, characterName, dirPath):
        # Load the entries from a JSON file
        self.entries = ListSerializer.deserialize(os.path.join(dirPath, f'{characterName}.entries'))
        # Load the embeddings from a JSON file
        self.embeddings = ListSerializer.deserialize(os.path.join(dirPath, f'{characterName}.embeddings'))

class SpeechConstructionModule:
    def __init__(self, speech_personality) -> None:
        self.speech_personality = speech_personality

    def ConstructResponse(self, queries, originalInput, previousSentences):
        prompt = f'''
        Given the information from the queries, which have been retrieved from your thoughts and are about you or the world, construct an in-character response to the original input.
        Only construct the next sentence. Do not construct the entire response. Do this by predicting the next sentence from the previous sentences. If no previous sentences exist, construct the first sentence.
        When you are finished predicting sentences (when it gets too long) say "NULL".
        Original Input: {originalInput}
        Previous Sentences: {previousSentences}
        Queries: {queries}
        
        Ouput Example: "Oh what the hell!"

        Make it sound like a human.
        Speech Personality: {self.speech_personality}
        Do not say anything else. Only say the response. And only one sentence.
        '''
        gpt = GPTInstance(prompt)
        return gpt.CreateChat(confirmString=None)

class SpeechToTextModule:
    def Transcribe(self, audio) -> None:
        transcript = openai.Audio.transcribe("whisper-1", audio)
        return transcript['text']

class TextToSpeechModule:
    def Speak(self, text) -> None:
        set_api_key(ELEVENLABS_KEY)

        audio = generate(
        text=text,
        voice="f5UejLGNKubShJkTy5UO",
        model="eleven_monolingual_v1"
        )
        
        play(audio)

    def Speak_async(self, text):
        process = mp.Process(target=self.Speak, args=(text,))
        process.start()

