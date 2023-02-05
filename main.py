import subprocess
import discord
import os
import json
import nltk
import requests
import scipy
import re
import networkx as nx
import subprocess
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.classify import NaiveBayesClassifier
from nltk.classify.util import accuracy
from nltk.corpus import wordnet
from nltk.cluster.util import cosine_distance
from nltk.tokenize import sent_tokenize
import numpy as np
from dotenv import load_dotenv
from discord.ext import commands, tasks
from discord.utils import get
from discord import FFmpegPCMAudio
from discord import TextChannel
from youtube_dl import YoutubeDL
from gensim.models import Word2Vec
import ast
load_dotenv()
intents = discord.Intents().all()
bot = discord.Client(intents=intents)
client = commands.Bot(command_prefix='!', intents=intents, activity = discord.Activity(type=discord.ActivityType.listening, name="Hodevs"))  # prefix our commands with '.'
bad_imports = ["os", "subprocess"]
user_imports = []

players = {}

if not os.path.isfile("config.json"):
    sys.exit("'config.json' not found! Please add it and try again.")
else:
    with open("config.json") as file:
        config = json.load(file)

@client.event  # check if bot is ready
async def on_ready():
    activity = discord.Activity(type=discord.ActivityType.listening, name="Musica Senape")
    await bot.change_presence(status=discord.Status.idle, activity=activity)
    print('Bot online')
    
@tasks.loop(minutes=1.0)
async def status_task() -> None:
    statuses = ["Musica Senaposa"]
    await bot.change_presence(activity=discord.Streaming.Game(random.choice(statuses))) 
    
@client.event
async def on_command(ctx):
    print(f"Command used: {ctx.command}")
    
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')
stop_words_english = set(stopwords.words('english'))
stop_words_italian = set(stopwords.words('italian'))
stop_words_french = set(stopwords.words('french'))
stop_words_spanish = set(stopwords.words('spanish'))
lemmatizer = WordNetLemmatizer()
data = [
    ("Hi, how are you?", "greeting"),
    ("What's up?", "greeting"),
    ("How's it going?", "greeting"),
    ("I'm good, thanks for asking.", "good"),
    ("I'm fine, thanks for asking.", "good"),
    ("I'm not good, thanks for asking.", "not good"),
    ("I'm not fine, thanks for asking.", "not good"),
]

def preprocess_text(text):
    text = text.lower()
    tokens = nltk.word_tokenize(text)
    tokens = [lemmatizer.lemmatize(token) for token in tokens if token not in stop_words]
    return tokens

def checkimport(tree):
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
           user_imports.append("".join([n.name for n in node.names]))
        elif isinstance(node, ast.ImportFrom):
            user_imports.append(node.module)
    temp = [x for x in user_imports if x in bad_imports]
    if temp:
        return True, temp[0]

def train_classifier(data):
    features = [(({word: True for word in preprocess_text(text)}, label) for text, label in data)]
    classifier = NaiveBayesClassifier.train(features)
    return classifier

classifier = None

def jaccard_similarity(list1, list2):
    s1 = set(list1)
    s2 = set(list2)
    return len(s1 & s2) / len(s1 | s2)

@client.hybrid_command(name="summarize")
async def summarize(ctx, num_sentences: int, language: str, *, text: str):
    if language == "english":
        stop_words = stop_words_english
    elif language == "french":
        stop_words = stop_words_french
    elif language == "italian":
        stop_words = stop_words_italian
    elif language == "spanish":
        stop_words = stop_words_spanish
    sentences = sent_tokenize(text)
    words = [word_tokenize(sent) for sent in sentences]
    words = [[word.lower() for word in sent if word.isalpha()] for sent in words]
    words = [[word for word in sent if word not in stop_words] for sent in words]

    similarity_matrix = np.zeros((len(sentences), len(sentences)))
    for i in range(len(sentences)):
        for j in range(len(sentences)):
            if i != j:
                similarity_matrix[i][j] = jaccard_similarity(words[i], words[j])

    similarity_graph = nx.from_numpy_array(similarity_matrix)
    scores = nx.pagerank(similarity_graph)
    ranked_sentences = sorted(((scores[i], sent) for i, sent in enumerate(sentences)), reverse=True)

    summary = " ".join([ranked_sentences[i][1] for i in range(num_sentences)])
    await ctx.send(summary)


@client.command()
async def checkcode(ctx, *, code: str):
    data = ast.parse(code)
    if not checkimport(data)[0]:
        process = subprocess.Popen(['python', '-c', code], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stdout, stderr = process.communicate()

        if process.returncode == 0:
            await ctx.send("Code is correct!")
        else:
            await ctx.send(f"Error: {stderr.decode()}")
    else:
        await ctx.send("bad code! Cannot use {}".format(checkimport(data)[1]))
    del user_imports[:]


lemmatizer = WordNetLemmatizer()

def find_synonym(word):
    synonyms = []
    for syn in wordnet.synsets(word):
        for lemma in syn.lemmas():
            synonyms.append(lemma.name())
    if len(synonyms) > 0:
        return synonyms[0]
    else:
        return word

@client.command()
async def changewords(ctx, *, text: str):
    words = word_tokenize(text)
    new_words = []
    for word in words:
        try:
            pos = nltk.pos_tag([word])[0][1]
            if pos.startswith('J'):
                pos = wordnet.ADJ
            elif pos.startswith('V'):
                pos = wordnet.VERB
            elif pos.startswith('N'):
                pos = wordnet.NOUN
            elif pos.startswith('R'):
                pos = wordnet.ADV
            else:
                pos = None
            lemma = lemmatizer.lemmatize(word, pos)
            synonym = find_synonym(lemma)
            if synonym:
                new_words.append(synonym)
            else:
                new_words.append(word)
        except KeyError:
            new_words.append(word)
    new_text = ' '.join(new_words)
    await ctx.send(new_text)
    
    
@client.command()
async def chat(ctx, *, text):
    global classifier
    if classifier is None:
        classifier = NaiveBayesClassifier.train([({word: True for word in preprocess_text(text)}, label) for text, label in data])
    features = {word: True for word in preprocess_text(text)}
    label = classifier.classify(features)
    if label == "greeting":
        await ctx.send("Hello! How are you?")
    elif label == "good":
        await ctx.send("That's great to hear!")
    elif label == "not good":
        await ctx.send("I'm sorry to hear that.")
        
@client.command()
async def wikipedia(ctx, *, query: str):
    url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{query}"
    response = requests.get(url)
    data = json.loads(response.text)
    if response.status_code == 200:
        await ctx.send(data['extract'])
    else:
        await ctx.send(f"Sorry, I couldn't find a Wikipedia page for {query}.")

# command for bot to join the channel of the user, if the bot has already joined and is in a different channel, it will move to the channel the user is in
@client.command()
async def join(ctx):
    channel = ctx.message.author.voice.channel
    voice = get(client.voice_clients, guild=ctx.guild)
    if voice and voice.is_connected():
        await voice.move_to(channel)
    else:
        voice = await channel.connect()
        
@client.command(name='leave', help='To make the bot leave the voice channel')
async def leave(ctx):
    voice_client = ctx.message.guild.voice_client
    if voice_client.is_connected():
        await voice_client.disconnect()
    else:
        await ctx.send("The bot is not connected to a voice channel.")

# command to play sound from a youtube URL
@client.command()
async def play(ctx, url):
    YDL_OPTIONS = {'format': 'bestaudio', 'noplaylist': 'True'}
    FFMPEG_OPTIONS = {
        'before_options': '-reconnect 1 -reconnect_streamed 1 -reconnect_delay_max 5', 'options': '-vn'}
    voice = get(client.voice_clients, guild=ctx.guild)

    if not voice.is_playing():
        with YoutubeDL(YDL_OPTIONS) as ydl:
            info = ydl.extract_info(url, download=False)
        URL = info['url']
        voice.play(FFmpegPCMAudio(URL, **FFMPEG_OPTIONS))
        voice.is_playing()
        await ctx.send('Bot is playing')

# check if the bot is already playing
    else:
        await ctx.send("Bot is already playing")
        return


# command to resume voice if it is paused
@client.command()
async def resume(ctx):
    voice = get(client.voice_clients, guild=ctx.guild)

    if not voice.is_playing():
        voice.resume()
        await ctx.send('Bot is resuming')


# command to pause voice if it is playing
@client.command()
async def pause(ctx):
    voice = get(client.voice_clients, guild=ctx.guild)

    if voice.is_playing():
        voice.pause()
        await ctx.send('Bot has been paused')


# command to stop voice
@client.command()
async def stop(ctx):
    voice = get(client.voice_clients, guild=ctx.guild)

    if voice.is_playing():
        voice.stop()
        await ctx.send('Stopping...')
        


@commands.has_permissions(kick_members=True)

@client.command()
async def clear(ctx, amount=5):
    await ctx.channel.purge(limit=amount)
    await ctx.send("Messages have been cleared")
    
@client.command()
async def weather(ctx, *, location):
    API_key = "Api_Key"
    weather_url = f"http://api.openweathermap.org/data/2.5/weather?q={location}&appid={API_key}&units=metric"
    weather_data = requests.get(weather_url)
    weather_data = weather_data.json()
    
    try:
        temp = weather_data["main"]["temp"]
        weather_description = weather_data["weather"][0]["description"]
        city = weather_data["name"]
        country = weather_data["sys"]["country"]
        await ctx.send(f"Weather in {city}, {country}: {temp}°C, {weather_description}")
    except KeyError as e:
        await ctx.send("An error occurred: "+str(e))
  
client.run(config["token"])
