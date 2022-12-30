"""
Creates json, csv manifest for huggingface datasets of files on mount for speech recognition with TCRS dataset.
Authors: neelan@elucidate.ai
"""

SRC="./sample/"
DST="./"



import os
import json
import shutil
import logging
import pandas as pd
import torchaudio
import math

logger = logging.getLogger(__name__)

def get_aud(txt_file):
    """
    Helper function to return full audio file path from full text path
    Args:
    - txt_file: full path .txt 
    Returns:
    - aud_file: path to audio file
    """
    aud_file = txt_file.replace('assembly_ai/','').replace('.txt','')

    return aud_file

def get_text(aud_file, location):
    """
    Helper function to return text transcription path from aud path
    Args:
    - aud_file: full path .wav 
    Returns:
    - txt_file: path to text transcriptio file
    """
    aud_file = aud_file.replace('.wav','.wav.txt').replace(mount_path,location)

    return aud_file

def read_txt(file_):
    """
    Helper function to read .txt file
    Args:
    - file_: .txt file that results from assembly_ai
    Returns:
    - file_contents: rendering of file
    """
    f = open(file_, 'r')
    file_contents = f.read()
    return file_contents


def asr_parse_to_json(txt_files, wav_files, wav_source, txt_source):

    """
    json produced is fed into hf datasets
    """
    # in this dataset files names are spk_id-chapter_id-utterance_id.flac
    # we build a dictionary with words for each utterance
    words_dict = {}
    # we now build JSON examples
    examples = {}
    # n counts valid data output 
    n = 0
    #for i, txtf in enumerate(txt_files):

    print("parsing...")
    for i, txtf in enumerate(txt_files[:5]):
        print(txtf)
        id_ = txtf.replace(txt_source, "").replace(".wav.txt","")
        with open(txtf, "r") as f:
            lines = f.readlines()
        

        for l in lines:
            l = l.strip("\n")
            #utt_id = l.split(" ")[0]
            words = " ".join(l.split(" ")[1:])
            #words_dict[id_] = words
    #print("words dict parsed")


        wavf = txtf.replace(txt_source, wav_source).replace(".txt","")
        # define data file features
        id_ = wavf.replace(wav_source, "").replace(".wav","")
        words_write = words #words_dict[id_]
        # blank training data can produce no supervision
        if words_write == '' : continue
        if words_write == None: continue
        spkID = ''.join([i for i in id_ if not i.isdigit()]).replace('/','').replace('-','').replace('##','#')
        word_count = len(words_write.split())
        duration_seconds =  torchaudio.info(wavf).num_frames / torchaudio.info(wavf).sample_rate
        words_per_second = word_count/duration_seconds
        
        # applying filtering of raw data
        #if words_per_second < 1: continue

        # start counting after filtering
        n =+ 1
        # random split & partioning

        examples[id_] = {"file_path": wavf,
                            "words": words_write,
                            "word_count": word_count,
                            "spkID": spkID,
                            "bits_per_sample": torchaudio.info(wavf).bits_per_sample,
                            "encoding": torchaudio.info(wavf).encoding,
                            "num_channels": torchaudio.info(wavf).num_channels,
                            "num_frames": torchaudio.info(wavf).num_frames,
                            "sample_rate": torchaudio.info(wavf).sample_rate,
                            "duration_seconds": duration_seconds,
                            "words_per_second": words_per_second        
                            }
    
    # return examples as csv that may be ingest via huggingface datasets
    examples_pd = pd.DataFrame(examples).transpose()
    examples_pd.to_csv('data.csv')

    #return examples as json that may be ingest via huggingface datasets
    examples_pd[0].to_json('data.json')

def parse2csv(wav_source=SRC,wav_dest=DST, detail=True, limit=math.inf, csv_out_name="data"):

    """
    csv produced is fed into hf datasets
    """
    # in this dataset files names are spk_id-chapter_id-utterance_id.flac
    # we build a dictionary with words for each utterance
    words_dict = {}
    # we now build JSON examples
    examples = {}
    # n counts valid data output 
    n = 0
    #for i, txtf in enumerate(txt_files):
    source_files = os.listdir(wav_source)
    print("parsing...")
    for i, wavf in enumerate(source_files):
      if i < limit:
        if ".wav" in wavf:
          # define data file features
          id_ = wavf.replace(wav_source, "").replace(".wav","")
        
        wavf = os.path.join(wav_source, wavf)
        duration_seconds =  torchaudio.info(wavf).num_frames / torchaudio.info(wavf).sample_rate
        if detail:
          examples[id_] = {"file_path": wavf,
                        "bits_per_sample": torchaudio.info(wavf).bits_per_sample,
                        "encoding": torchaudio.info(wavf).encoding,
                        "num_channels": torchaudio.info(wavf).num_channels,
                        "num_frames": torchaudio.info(wavf).num_frames,
                        "sample_rate": torchaudio.info(wavf).sample_rate,
                        "duration_seconds": duration_seconds,     
                          "bits_per_sample": torchaudio.info(wavf).bits_per_sample,
                          "encoding": torchaudio.info(wavf).encoding,
                          "num_channels": torchaudio.info(wavf).num_channels,
                          "num_frames": torchaudio.info(wavf).num_frames,
                          "sample_rate": torchaudio.info(wavf).sample_rate,
                          "duration_seconds": duration_seconds,     
                              }
        else:
          examples[id_] = {"file_path": wavf}

    # return examples as csv that may be ingest via huggingface datasets
    examples_pd = pd.DataFrame(examples).transpose()
    examples_pd.to_csv(csv_out_name+'.csv')


def main():
  parse2csv()



if __name__ == "__main__":
  main() 
