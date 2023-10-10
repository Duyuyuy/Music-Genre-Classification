import numpy as np
import librosa as lbs
import pandas as pd
import os
import shutil
from pydub import AudioSegment
from pydub.playback import play
from pydub.utils import make_chunks


metadata= pd.read_csv("data/output1.csv")
genres = metadata['TAGS'].unique()
onesample=['electronic', 'classical', 'pop', 'ambient', 'poprock', 'rock', "bossanova"]
twosample=["orchestral", "jazz", "house", "alternative", "techno", "dance", "chillout", "metal", "blues", "hiphop", "latin", "newage", "country", "rnb"]
threesample=["reggae", "folk", "funk", "world","rap"]
chunk_length_ms = 30000 # Interval length in milliseconds (10 seconds)


# Create an Audio object and display it

# for genre in genres:
#     os.makedirs(f'data/process/{genre}', exist_ok=True)

for audio_filename in os.listdir("data/20"):
    if audio_filename.endswith(".mp3"):
        # # Load the audio file
        # audio_file = AudioSegment.from_file(f"data/00mp3/{audio_filename}", format="mp3")
        track_id = f'data/20/{audio_filename}'
        # print(f'00/{track_id}')

        # Look up the corresponding genre in the metadata file
        genre = metadata.loc[metadata['PATH'] ==f'20/{audio_filename}', 'TAGS'].values
        if len(genre) == 0 or genre[0] == "":
             shutil.move(track_id, f"data/process/Test/{audio_filename}")

        elif genre[0] in ["No", "nan", None]:
            print("Invalid")

        else:
            # Load the audio file and split it into chunks
         audio = AudioSegment.from_file(track_id, format="mp3")
         chunks = make_chunks(audio, chunk_length_ms)
         chunk_index = 0
         if genre[0] in onesample:
            chunk_index = 1
         elif genre[0] in twosample:
            chunk_index = 2
         elif genre[0] in threesample:
            chunk_index = 3

         if chunk_index < len(chunks) & chunk_index>1:
           for i in range(1,chunk_index+1):
                x = audio_filename.split(".")
                a=f"{x[0]}({i}).mp3"
                chunk_name = f"data/process/{genre[0]}/{a}"
                chunks[i].export(chunk_name, format="mp3")
         else:
             chunk_name = f"data/process/{genre[0]}/{audio_filename}"
             chunks[0].export(chunk_name, format="mp3")

print("sucess")

