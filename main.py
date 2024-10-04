import concurrent.futures
import asyncio
import os
from pydub import AudioSegment
from pyannote.audio import Pipeline
os.environ['KMP_DUPLICATE_LIB_OK']='True'

from faster_whisper import WhisperModel
from fastapi import FastAPI, UploadFile, File, HTTPException, Request, Form, Depends, status
from fastapi.security import HTTPAuthorizationCredentials
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, RedirectResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from typing import List

# Constants
from constants import device, compute_type, security, MAX_THREADS

# Responses
from responses import SUCCESSFUL_RESPONSE, BAD_REQUEST_RESPONSE
from responses import VALIDATION_ERROR_RESPONSE, INTERNAL_SERVER_ERROR_RESPONSE

# Logging configuration
from logging_config import get_logger
logger = get_logger()

app = FastAPI()

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Diarization pipeline (use pre-trained model)
diarization_pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization")

# Helper functions
from utils import authenticate_user
from utils import process_file, validate_parameters

# Routes
@app.get("/", response_class=RedirectResponse)
async def redirect_to_docs():
    return "/docs"

@app.get('/info')
def home():
    return HTMLResponse(content=f"""
        <h1>FastWhisperAPI is running on <span style="color: blue;">{device}</span>!</h1>
        <p>Version: <strong>1.0</strong></p>
        <p>Author: <strong>Edoardo Cilia</strong></p>
        <p>License: <strong>Apache License 2.0</strong></p>
        <h2>Endpoints:</h2>
        <ul>
            <li>
                <h3>/v1/transcriptions</h3>
                <p>Method: POST</p>
                <p>Description: API designed to transcribe audio files leveraging the Faster Whisper library and FastAPI framework.</p>
                <h4>Parameters:</h4>
                <ul>
                    <li>file: A list of audio files to transcribe. This is a required parameter.</li>
                    <li>model: The size of the model to use for transcription. This is an optional parameter. The options are 'large', 'medium', 'small', 'base', 'tiny'. Default is 'base'.</li>
                    <li>language: This parameter specifies the language of the audio files. It is optional, with accepted values being lowercase ISO-639-1 format. (e.g., 'en' for English). If not provided, the system will automatically detect the language.</li>
                    <li>initial_prompt: This optional parameter provides an initial prompt to guide the model's transcription process. It can be used to pass a dictionary of the correct spellings of words and to provide context for better understanding speech, thus maintaining a consistent writing style.</li>
                    <li>vad_filter: Whether to apply a voice activity detection filter. This is an optional parameter. Default is False.</li>
                    <li>min_silence_duration_ms: The minimum duration of silence to be considered as a pause. This is an optional parameter. Default is 1000.</li>
                    <li>response_format: The format of the response. This is an optional parameter. The options are 'text', 'verbose_json'. Default is 'text'.</li>
                    <li>timestamp_granularities=segment</li>
                </ul>
                <h4>Example curl request:</h4>
                <ul style="list-style-type:none;">
                    <li>curl -X POST "http://localhost:8000/v1/transcriptions" \</li>
                    <li>-H  "accept: application/json" \</li>
                    <li>-H  "Content-Type: multipart/form-data" \</li>
                    <li>-F "file=@audio1.wav;type=audio/wav" \</li>
                    <li>-F "file=@audio2.wav;type=audio/wav" \</li>
                    <li>-F "model=base" \</li>
                    <li>-F "language=en" \</li>
                    <li>-F "initial_prompt=RoBERTa, Mixtral, Claude 3, Command R+, LLama 3." \</li>
                    <li>-F "vad_filter=False"</li>
                    <li>-F "min_silence_duration_ms=1000"</li>
                    <li>-F "response_format=text"</li>
                </ul>
            </li>
            <li>
                <h3>/</h3>
                <p>Method: GET</p>
                <p>Description: Redirects to the /docs endpoint.</p>
            </li>
        </ul>
    """)

@app.post('/v1/transcriptions',
          responses={
              200: SUCCESSFUL_RESPONSE,
              400: BAD_REQUEST_RESPONSE,
              422: VALIDATION_ERROR_RESPONSE,
              500: INTERNAL_SERVER_ERROR_RESPONSE,
          }
)
async def transcribe_audio(credentials: HTTPAuthorizationCredentials = Depends(security),
                           file: List[UploadFile] = File(...),
                           model: str = Form("base"),
                           language: str = Form(None),
                           initial_prompt: str = Form(None),
                           vad_filter: bool = Form(False),
                           min_silence_duration_ms: int = Form(1000),
                           response_format: str = Form("text"),
                           timestamp_granularities: str = Form("segment")):
    user = authenticate_user(credentials)
    validate_parameters(file, language, model, vad_filter, min_silence_duration_ms, response_format, timestamp_granularities)
    word_timestamps = timestamp_granularities == "word"
    whisper_model = WhisperModel(model, device=device, compute_type=compute_type)

    transcriptions = {}

    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_THREADS) as executor:
        futures = []
        
        for audio_file in file:
            file_path = f"./{audio_file.filename}"
            with open(file_path, "wb") as buffer:
                buffer.write(await audio_file.read())

            # Diarize the audio and split based on speakers
            speaker_segments = diarize_audio(file_path)
            speaker_files = split_audio(file_path, speaker_segments)

            # Transcribe the speaker-labeled segments
            future = executor.submit(transcribe_segments, speaker_files, whisper_model, language, word_timestamps, vad_filter, min_silence_duration_ms)
            futures.append(future)

        # Collect and format the results
        for i, future in enumerate(concurrent.futures.as_completed(futures), start=1):
            try:
                result = future.result()
                transcriptions[f"File {i}"] = {"text": result}
            except Exception as e:
                logger.error(f"An error occurred during transcription: {str(e)}")
                raise HTTPException(status_code=500, detail=str(e))

    logger.info(f"Transcription completed for {len(file)} file(s).")
    
    # Return the formatted transcription JSON response
    return JSONResponse(content=transcriptions)

# Helper functions
def diarize_audio(file_path):
    """
    Perform speaker diarization on an audio file using pyannote-audio and return speaker-labeled segments.
    """
    diarization = diarization_pipeline(file_path)
    
    # Extract speaker segments
    speaker_segments = []
    for segment in diarization.itertracks(yield_label=True):
        speaker_segments.append({
            'speaker': segment[2],  # Speaker label
            'start': segment[0].start,  # Start time in seconds
            'end': segment[0].end  # End time in seconds
        })
    return speaker_segments

def split_audio(file_path, segments):
    """
    Split audio into speaker-labeled segments based on diarization results.
    """
    audio = AudioSegment.from_file(file_path)
    
    if not os.path.exists('speaker_segments'):
        os.makedirs('speaker_segments')

    speaker_files = []
    for i, segment in enumerate(segments):
        start_ms = segment['start'] * 1000
        end_ms = segment['end'] * 1000
        
        # Extract speaker segment from the audio file
        speaker_audio = audio[start_ms:end_ms]
        speaker_file_path = f"speaker_segments/speaker_{segment['speaker']}_{i}.wav"
        speaker_audio.export(speaker_file_path, format="wav")
        speaker_files.append((speaker_file_path, segment['speaker']))
    
    return speaker_files

def transcribe_segments(speaker_files, model, language, word_timestamps, vad_filter, min_silence_duration_ms):
    """
    Transcribe each speaker segment using the Whisper model.
    """
    transcriptions = []
    for speaker_file, speaker in speaker_files:
        segments, info = model.transcribe(speaker_file, language=language, word_timestamps=word_timestamps, vad_filter=vad_filter, min_silence_duration_ms=min_silence_duration_ms)
        
        # Collect transcription text with speaker label
        for segment in segments:
            transcriptions.append(f"{speaker}: {segment.text.strip()}")  # Simple chat format
    
    return transcriptions

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content=exc.detail,
    )

@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception):
    status_code = 500
    error_type = type(exc).__name__
    if isinstance(exc, ValueError) or isinstance(exc, TypeError):
        status_code = 400
    return JSONResponse(
        status_code=status_code,
        content={
            "error": {
                "message": str(exc),
                "type": error_type,
                "param": "",
                "code": status_code
            }
        },
    )

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request, exc):
    details = exc.errors()[0]['msg']
    loc = exc.errors()[0]['loc']  
    return JSONResponse(
        status_code=422,
        content={
            "error": {
                "message": details,
                "type": "invalid_request_error",
                "param": loc[-1] if loc else "",
                "code": 422
            }
        },
    )
