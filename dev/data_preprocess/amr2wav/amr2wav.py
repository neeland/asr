# Databricks notebook source
# MAGIC %md
# MAGIC ## Introduction
# MAGIC 
# MAGIC Workbook contains loop to convert all `.amr` files in selected folder to `.wav` files.
# MAGIC 
# MAGIC Adapted from [this](https://gist.github.com/Kronopath/c94c93d8279e3bac19f2) process
# MAGIC 
# MAGIC ### Note
# MAGIC 
# MAGIC We create a [Notebook-scoped python library](https://docs.microsoft.com/en-us/azure/databricks/libraries/notebooks-python-libraries) - we create a custom environment with magic `%sh` and no other notebooks attached to the cluster are effected. 

# COMMAND ----------

# MAGIC %md
# MAGIC We follow [these](https://stackoverflow.com/questions/64425119/pyspark-use-ffmpeg-on-the-driver-and-workers) instruction to use `ffmpeg` on databricks

# COMMAND ----------

import os, pyspark, subprocess, ffmpeg, sox, audiofile
from pydub import AudioSegment

# COMMAND ----------

# MAGIC  %sh
# MAGIC which ffmpeg

# COMMAND ----------

src = "/dbfs/mnt/asr/1/2.amr"
dst =  "/dbfs/mnt/asr/1/converted/2.wav"

# convert wav to mp3                                                            
audSeg = AudioSegment.from_file(src)
wavFile = audSeg.export(dst, format="wav")

# COMMAND ----------

audiofile.convert_to_wav('noise.flac', 'noise.wav')
audiofile.samples('noise.wav')

# COMMAND ----------

1# check if script exists else create
try:
  display(dbutils.fs.ls("dbfs:/databricks/scripts/install-sox.sh"))
except:
 #create a new dir
  dbutils.fs.mkdirs("dbfs:/databricks/scripts/") 

  #create an init script into the newly created dir
  dbutils.fs.put("/databricks/scripts/install-sox.sh","""
  #!/bin/bash
  sudo apt-get install sox
  """, True)

# COMMAND ----------

1# check if script exists else create
try:
  display(dbutils.fs.ls("dbfs:/databricks/scripts/install-ffmpeg.sh"))
except:
 #create a new dir
  dbutils.fs.mkdirs("dbfs:/databricks/scripts/") 

  #create an init script into the newly created dir
  dbutils.fs.put("/databricks/scripts/install-ffmpeg.sh","""
  #!/bin/bash
  sudo apt-get update
  sudo apt-get -y install ffmpeg""", True)


# COMMAND ----------

# list all dbfs mounts
for mount in dbutils.fs.mounts():
  print (mount.mountPoint)

# COMMAND ----------

mount_name = 'asr'
# list files in mounted folder
os.listdir(f'/dbfs/mnt/{mount_name}')

# COMMAND ----------

def convert_amr_file(input_dir='/dbfs/mnt/asr/1', input_file_name='2.amr', verbose=True, remove_intermediate=False):

    """
    converts single to intermediate .aud file before converting to .wav
    """
    # create an additional folder to store converted .wav files
    output_dir = os.path.join(input_dir, 'converted' )
    if not os.path.isdir(output_dir): os.mkdir(output_dir)

    # find absolute file path & append .amr file to path
    input_file_path = os.path.join(input_dir, input_file_name)
    if verbose: print(input_file_path)

    # open this input .amr file
    input_file = open(input_file_path, 'rb')
    if verbose: print(input_file)

    # replace input .amr file name w/ intermediatiary .aud file
    intermediate_file_name = input_file_name.replace(".amr",".mp3")
    if verbose: print(f"intermediate_file_name {intermediate_file_name}")

    # create a path for the .aud file in the "/converted" folder space
    intermediate_file_path = os.path.join(output_dir, intermediate_file_name)
    if verbose: print(f"intermediate_file_path {intermediate_file_path}")

    # open the intermediate file
    intermediate_file = open(intermediate_file_path, 'wb')
    if verbose: print(f"intermediate_file {intermediate_file}")

    # write the input file to the intermediate file and close both
    intermediate_file.write(input_file.read())
    input_file.close()
    intermediate_file.close()

    # replace .amr with .wav naming
    output_file_name = input_file_name.replace(".amr", ".wav")
    if verbose: print(f"output_file_name {output_file_name}")

    # join this .wav file path to the autput directory path
    output_file_path = os.path.join(output_dir, output_file_name)
    if verbose: print(f"output_file_path {output_file_path}")

    # create a file to dump the .aud files
    black_hole_file = open("black_hole", "w")

    # convert .aud files to .wav files w/ specific sampling rate
    # sampling rate alteration follows this method: https://stackoverflow.com/questions/63793137/convert-mp3-to-wav-with-custom-sampling-rate
    subprocess.call([ "ffmpeg", "-i", intermediate_file_path, "-ar", "16k", output_file_path],
                    stdout = black_hole_file, stderr = black_hole_file)
    
    # close dump file
    black_hole_file.close()

    if remove_intermediate:
        # delete junk files
        os.remove("black_hole")
        os.remove(intermediate_file_path)

        ###~
convert_amr_file()

# COMMAND ----------

os.listdir('/dbfs/mnt/asr/1')

# COMMAND ----------

import os, argparse, subprocess
from datetime import datetime

def bulk_convert(audio_src='1', verbose=True, remove_intermediate=False):
    """
    converts all .amr file is audio_src to intermediate .aud file before converting to .wav in audio_src/converted
    
    ***Args:
    - audio_src: folder containing .amr files that need conversion
               : assumes this is in same folder as workbook
    - verbose: will print progress if True
    - remove_intermediate: will remove intermediate .aud if True
    
    ***Returns:
    - folder audio_src/converted: contains converted .wav versions of the .amr files in audio_src
    
    """
    # find the absolute file path to your audio files
    audio_src = os.path.join(os.getcwd(), audio_src)
    ERRORS = []
    # loop over each file in directory & convert to .wav
    for dirname, dirnames, filenames in os.walk(audio_src):
        for filename in filenames:
            input_path = os.path.join(dirname, filename)
            if verbose: print(f'input_path: {input_path}')
            try: 
                convert_amr_file(dirname, filename,verbose=verbose)
                if verbose: print(f"!!! DONE CONVERTING {filename}!")
            except:
                if verbose:  print(f"!!! ERROR on {filename}")
                ERRORS.append(filename)
                
bulk_convert()

# COMMAND ----------

os.listdir('/dbfs/mnt/asr/1/converted')

# COMMAND ----------

bulk_convert(audio_src='/dbfs/mnt/asr/1', verbose=True, remove_intermediate=False)

# COMMAND ----------


