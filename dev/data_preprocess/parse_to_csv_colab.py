"""
Creates json, csv manifest for huggingface datasets of files on mount for speech recognition with TCRS dataset.
Authors: neelan@elucidate.ai
"""
import os
import json
import shutil
import logging
import pandas as pd
import torchaudio

def parse_to_csv_colab(wav_files, wav_source):

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
    for i, wavf in enumerate(wav_files):
        # define data file features
        id_ = wavf.replace(wav_source, "").replace(".wav","")

        duration_seconds =  torchaudio.info(wavf).num_frames / torchaudio.info(wavf).sample_rate

        examples[id_] = {"file_path": wavf,
                        "bits_per_sample": torchaudio.info(wavf).bits_per_sample,
                        "encoding": torchaudio.info(wavf).encoding,
                        "num_channels": torchaudio.info(wavf).num_channels,
                        "num_frames": torchaudio.info(wavf).num_frames,
                        "sample_rate": torchaudio.info(wavf).sample_rate,
                        "duration_seconds": duration_seconds,     
                            }
    
    # return examples as csv that may be ingest via huggingface datasets
    examples_pd = pd.DataFrame(examples).transpose()
    examples_pd.to_csv('data.csv')



if __name__ == "__main__":
  import glob
  import pickle

  wav_files = []
  wav_source = '/content/drive/MyDrive/tcrs/'
  for file_ in os.listdir(wav_source):
    if file_.endswith(".wav"):
        wav_files.append(wav_source + file_)
  # parsing in time consuming so we only want to do it once per machine
  with open("wav_files.pkl", "wb") as fp:   #Pickling
      pickle.dump(wav_files, fp)

  parse_to_csv_colab(wav_files, wav_source)