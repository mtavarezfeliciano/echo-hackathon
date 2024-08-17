import discord
from discord.ext import commands
from dotenv import load_dotenv, find_dotenv
import os
from langchain.prompts import SystemMessagePromptTemplate, PromptTemplate
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import TextLoader
from langchain.schema import HumanMessage
import traceback

#hi nathan and peter lol
load_dotenv(find_dotenv())

loader = TextLoader('./listings.txt')
documents = loader.load()
text_splitter = CharacterTextSplitter(chunk_size=300, chunk_overlap=100)
texts = text_splitter.split_documents(documents)

embeddings = OpenAIEmbeddings()
retriever = Chroma.from_documents(texts, embeddings).as_retriever()
chat = ChatOpenAI(temperature=1)

prompt_template = """You are a helpful discord bot that helps users with looking for listings in the Bronx, New York.

{context}

Please provide the most suitable response for the users question. If asked a question regarding meetings, viewing, or sign ups give them their respective links
Answer:"""

prompt = PromptTemplate(
    template=prompt_template, input_variables=["context"]
)
system_message_prompt = SystemMessagePromptTemplate(prompt=prompt)


intents = discord.Intents.default()
intents.message_content = True

bot = commands.Bot(command_prefix='!', intents=intents)

#Signup Stuff
SIGNUP_CHANNEL = int(os.getenv("SIGNUP_CHANNEL"))
DISCORD_TOKEN = os.getenv("SIGNUP_CHANNEL")
GUILDS_ID = int(os.getenv("GUILDS_ID"))

@bot.command()
async def question(ctx, *, question):
    try:
        docs = retriever.get_relevant_documents(query=question)
        formatted_prompt = system_message_prompt.format(context=docs)

        messages = [formatted_prompt, HumanMessage(content=question)]
        result = chat(messages)
        await ctx.send(result.content)
    except Exception as e:
        print(f"Error occurred: {e}")
        await ctx.send("Sorry, I was unable to process your question.")

bot.run(os.environ.get("DISCORD_TOKEN"))