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

load_dotenv()
intents = discord.Intents().all()
bot = discord.Client(intents=intents)
client = commands.Bot(command_prefix='!', intents=intents, activity = discord.Activity(type=discord.ActivityType.listening, name="Hodevs"))  # prefix our commands with '.'


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
    
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))
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

def train_classifier(data):
    features = [(({word: True for word in preprocess_text(text)}, label) for text, label in data)]
    classifier = NaiveBayesClassifier.train(features)
    return classifier

classifier = None

def jaccard_similarity(sent1, sent2):
    a = set(sent1) 
    b = set(sent2)
    c = a.intersection(b)
    return float(len(c)) / (len(a) + len(b) - len(c))


def summarize_text(text):
    stop_words = set(stopwords.words('english'))
    sentences = nltk.sent_tokenize(text)
    tokenized_sentences = [nltk.word_tokenize(sent) for sent in sentences]
    cleaned_sentences = [[word for word in sent if word.lower() not in stop_words] for sent in tokenized_sentences]
    sentence_similarity_matrix = np.zeros((len(sentences), len(sentences)))

    for i in range(len(sentences)):
        for j in range(len(sentences)):
            if i != j:
                sentence_similarity_matrix[i][j] = jaccard_similarity(cleaned_sentences[i], cleaned_sentences[j])
    sentence_similarity_graph = nx.from_numpy_array(sentence_similarity_matrix)
    scores = nx.pagerank(sentence_similarity_graph)
    ranked_sentences = sorted(((scores[i], s) for i, s in enumerate(sentences)), reverse=True)
    summarize_text = []
    for i in range(3):
        if i < len(ranked_sentences):
            summarize_text.append(ranked_sentences[i][1])
        else:
            break
    return ". ".join(summarize_text)


@client.command()
async def summarize(ctx, *, text: str):
    summary = summarize_text(text)
    await ctx.send(f"Summary: {summary}")
    
@client.command()
async def checkcode(ctx, *, code: str):
    process = subprocess.Popen(['python', '-c', code], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = process.communicate()

    if process.returncode == 0:
        await ctx.send("Code is correct!")
    else:
        await ctx.send(f"Error: {stderr.decode()}")
        
@client.command()
async def changewords(ctx, *, text: str):
    nltk.download('wordnet')
    words = nltk.word_tokenize(text)
    new_words = []
    for word in words:
        try:
            synsets = wordnet.synsets(word)
            if synsets:
                new_word = synsets[0].lemmas()[0].name()
                new_words.append(new_word)
        except Exception as e:
            await ctx.send(f"An error occurred: {e}")
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
