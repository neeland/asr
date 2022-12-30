"""
Creates manifest of files on mount for speech recognition with TCRS dataset.
Authors: neelan@elucidate.ai

"""

import os
import json
import shutil
import logging
import pandas as pd
import torchaudio

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


def asr_parse_to_json(txt_files, wav_files):

    """
    json produced is fed into speechbrain.IO
    """

    # indexes
    n_train = 567
    n_valid = 850

    # in this dataset files names are spk_id-chapter_id-utterance_id.flac
    # we build a dictionary with words for each utterance
    words_dict = {}
    print("parsing...")
    for txtf in txt_files:
        with open(txtf, "r") as f:
            lines = f.readlines()
        utt_id = txtf.replace("/home/azureuser/cloudfiles/code/Users/neelan/asr/pipeline/data/assembly_ai_0", "").replace(".wav","").replace(".txt", "").replace('/','')

        for l in lines:
            l = l.strip("\n")
            #utt_id = l.split(" ")[0]
            words = " ".join(l.split(" ")[1:])
            words_dict[utt_id] = words
    print("building json...")
    # we now build JSON examples
    examples = {}
    # n counts valid data output
    n = 0
    for i, file_ in enumerate(wav_files):
        if i//100 == 0:
            print(f'i: {i}')
        
        # define data file features
        id_ = file_.replace("/home/azureuser/cloudfiles/code/Users/neelan/asr/pipeline/data/", "").replace(".wav","").replace(".txt", "").replace('/','')
        words_write = words_dict[id_]
        # blank training data can produce no supervision
        if words_write == '' : continue
        if words_write == None: continue
        spkID = file_.replace(os.path.dirname(file_),'').replace('.wav','')
        spkID = ''.join([i for i in spkID if not i.isdigit()]).replace('/','').replace('-','').replace('##','#')
        word_count = len(words_write.split())
        duration_seconds =  torchaudio.info(file_).num_frames / torchaudio.info(file_).sample_rate
        words_per_second = word_count/duration_seconds
        
        # applying filtering of raw data
        #if words_per_second < 1: continue

        # start counting after filtering
        n =+ 1

        examples[id_] = {"file_path": file_,
                            "words": words_write,
                            "word_count": word_count,
                            "spkID": spkID,
                            "bits_per_sample": torchaudio.info(file_).bits_per_sample,
                            "encoding": torchaudio.info(file_).encoding,
                            "num_channels": torchaudio.info(file_).num_channels,
                            "num_frames": torchaudio.info(file_).num_frames,
                            "sample_rate": torchaudio.info(file_).sample_rate,
                            "duration_seconds": duration_seconds,
                            "words_per_second": words_per_second,
                            "outfile:": outfile              
                            }

    examples_pd = pd.DataFrame(examples).transpose()
    examples_pd.to_csv('data.csv')
    #return examples
    for file_ in ["train.json", "valid.json", "test.json"]:
        examples_pd[examples_pd["outfile"]==file_].to_json(file_)
        print(f"written {file_}")

    """
    with open("train.json", "w") as f:
        json.dump(examples_pd[examples_pd["outfile"]=="train.json"], f, indent=4)
    print("train.json written")

    with open("valid.json", "w") as g:
        json.dump(examples_pd[examples_pd["outfile"]=="valid.json"], g, indent=4)
    print("valid.json written")
    
    with open("test.json", "w") as h:
        json.dump(examples_pd[examples_pd["outfile"]=="test.json"], h, indent=4)
    print("test.json written")
    """
if __name__ == "__main__":
    import glob
    txt_files = [] 
    wav_files = [] 
    print('parsing...')
    for file in os.listdir('./data/'):
        if file.endswith(".txt"):
            txt_files.append(file)
        elif file.endswith(".wav"):
            wav_files.append(file)

    asr_parse_to_json(txt_files, wav_files)
