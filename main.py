import os
from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from beanie import Document, init_beanie
from motor.motor_asyncio import AsyncIOMotorClient
from datetime import datetime
import logging
import json
import http.client
from typing import List

from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser

# For running blocking calls in an async endpoint
from starlette.concurrency import run_in_threadpool

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define a Beanie document to store the film suggestion history
class FilmHistory(Document):
    user_input: str
    email: str
    film_name: str
    created_at: datetime = Field(default_factory=datetime.utcnow)

    class Settings:
        name = "film_history"

# Define a Pydantic model for the film suggestion output
class Film(BaseModel):
    name: str
    trailer: str
    director: str
    year: str
    description: str
    genre: str
    runtime: str
    cast: str
    boxOffice: str
    awards: str
    language: str

# MongoDB connection setup
mongo_url = os.getenv("MONGO_URL")
if not mongo_url:
    raise Exception("MONGO_URL not found in environment variables.")
client = AsyncIOMotorClient(mongo_url)

# Load the Groq API key from environment variables
groq_api_key = os.getenv("GROQ_API_KEY")
if not groq_api_key:
    raise Exception("GROQ_API_KEY not found in environment variables.")

# Initialize the ChatGroq model for film suggestions
filmChatModel = ChatGroq(
    temperature=1,
    groq_api_key=groq_api_key,
    model="llama3-70b-8192"  # Adjust the model if needed
)

# Set up a parser that converts the AI response to a Film object
parser = JsonOutputParser(pydantic_object=Film)

# Define a prompt template including format instructions from the parser.
prompt = PromptTemplate(
    template=(
        "You are an expert film critic. Based on the user's input mood, suggest a film that matches the mood. "
        "Provide a JSON response with the following keys: name, trailer, director, year, description, genre, "
        "runtime, cast, boxOffice, awards, language. Keep the response under 150 words.\n"
        "{format_instructions}\n"
        "Input: {text}"
    ),
    input_variables=["text"],
    partial_variables={"format_instructions": parser.get_format_instructions()}
)

# Create a chain by combining the prompt, the model, and the parser
chain = prompt | filmChatModel | parser

# Initialize FastAPI
app = FastAPI(title="FilmFinder")

from fastapi.middleware.cors import CORSMiddleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
async def startup_event():
    await init_beanie(
        database=client.filmfinder,  # Adjust database name if needed
        document_models=[FilmHistory],
    )

# Asynchronous helper to get the trailer link using the YouTube API
async def get_trailer_link(film_name: str) -> str:
    def blocking_trailer_call():
        # Create the query by URL-encoding spaces and appending 'trailer'
        query = film_name.replace(" ", "%20") + "%20trailer"
        conn = http.client.HTTPSConnection("youtube138.p.rapidapi.com")
        headers = {
            'x-rapidapi-key': "9e9fa423a4msh87afb3bb7efdb74p15d16ajsn5f94843732b9",
            'x-rapidapi-host': "youtube138.p.rapidapi.com"
        }
        endpoint = f"/search/?q={query}&hl=en&gl=US"
        conn.request("GET", endpoint, headers=headers)
        res = conn.getresponse()
        data = res.read()
        result = json.loads(data.decode("utf-8"))
        # Extract the video ID from the first result
        videoId = result["contents"][0]["video"]["videoId"]
        return f"https://www.youtube.com/embed/{videoId}"
    
    return await run_in_threadpool(blocking_trailer_call)

@app.post("/film-suggestion", response_model=Film)
async def film_suggestion(input: str, email: str):
    """
    Accepts a mood input and user email. Returns a film suggestion in JSON format and saves the query history.
    """
    try:
        # Invoke the chain to get film details
        film_info_dict = await run_in_threadpool(chain.invoke, {"text": input})
        film_instance = Film(**film_info_dict)
        
        # Update the trailer field by fetching the trailer link from YouTube API
        trailer_link = await get_trailer_link(film_instance.name)
        film_instance.trailer = trailer_link
        
        # Save the history record to MongoDB
        film_history = FilmHistory(user_input=input, email=email, film_name=film_instance.name)
        await film_history.insert()
        
        return film_instance
    except Exception as e:
        logger.error("Error in /film-suggestion: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/film-history", response_model=List[FilmHistory])
async def get_film_history(email: str):
    """
    Returns all film suggestion history records for the provided user email.
    """
    try:
        film_histories = await FilmHistory.find(FilmHistory.email == email).to_list()
        return film_histories
    except Exception as e:
        logger.error("Error in /film-history: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8001)
