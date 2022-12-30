"""
Creates csv transcripts from wav files, running inference using huggingface pipeline with chunking
Authors: neelan@elucidate.ai
"""
####################################################################
### setting up global variables ####################################
####################################################################

MOUNT = "./10k_sample_3" # relative path data wavs are stored
PARSE="./parse/split/csv_out_0_500.csv" # where parse csv is located

OUT = "./transcripts/0_500/"
# controls if logit_score and lm_score will be produced; loads additional model for this so increasing GPU overhead
LOGITS= True 

####################################################################
### import needed pacakges...   ####################################
####################################################################

import os
if not os.path.exists(OUT.replace("./","")):
     os.mkdir(OUT.replace("./",""))
import warnings
import pandas as pd
import torch
from tqdm.auto import tqdm, trange
warnings.filterwarnings("ignore")
from datasets import load_dataset, load_metric, Audio
import transformers
from transformers import pipeline
from transformers.pipelines.pt_utils import KeyDataset
from transformers import AutoModelForCTC, AutoProcessor

####################################################################
### defining processes...       ####################################
####################################################################



def load_data(data_csv=PARSE, verbose=True):
    """
    loads data from parsed .wavs from .csv for ingestion into huggingface models
    """


    data = load_dataset('csv', data_files=data_csv, download_mode='force_redownload')
    dataset = data['train']
    dataset = dataset.cast_column("file_path", Audio(sampling_rate=16000))
    #checking that mapping worked correctly
    if verbose: print(dataset[0])
    return dataset

def load_pipeline(dataset,model_id="patrickvonplaten/hubert-xlarge-ls960-ft-4-gram", task="automatic-speech-recognition"):
    """
    loads inference pipeline for processing data with chunking for long .wav files
    """
    pipe = pipeline(task, model=model_id, device=0, chunk_length_s=5, stride_length_s=(2, 1))
    # KeyDataset (only *pt*) will simply return the item in the dict returned by the dataset item
    # as we're not interested in the *target* part of the datasetn
    transcripts =  tqdm(pipe(KeyDataset(dataset, "file_path")))
    return transcripts

def load_model_processor(model_id= "patrickvonplaten/hubert-xlarge-ls960-ft-4-gram"):
    """
    loads model & processor needed for logit_score & lm_score
    """
    
    model = AutoModelForCTC.from_pretrained(model_id).to("cuda")
    processor = AutoProcessor.from_pretrained(model_id)
    return model, processor


def run(dataset, transcripts, model, processor, mount=MOUNT ,start=0, save_itt=50, shutdown=True, limit=999999):
    """
    runs transcription process
    """

    out = []
    for i, transcript in enumerate(transcripts, start=start):
        if i>limit: break
        if not os.path.isdir(OUT): os.mkdir(OUT)
        if (model != None) and (processor !=None): 
            inputs = processor.feature_extractor(dataset[i]["file_path"]["array"], sampling_rate=16_000, return_tensors="pt")
            inputs = {k: v.to("cuda") for k,v in inputs.items()}
            with torch.no_grad():
                # model is used to create logits
                logits = model(**inputs).logits
            logits_out = processor.batch_decode(logits.cpu().numpy())
            out.append({"file":dataset[i]["file_path"]["path"].replace(mount,"").replace(".wav","") , "pipe_text": transcript["text"], "logits_text": logits_out["text"], "logit_score": logits_out["logit_score"], "lm_score": logits_out["lm_score"]}) #, "word_offsets": logits_out["word_offsets"] })
        else:
            out.append({"file":dataset[i]["file_path"]["path"].replace(mount,"").replace(".wav","") , "pipe_text": transcript["text"]})
        if save_itt == None:
            dataframe = pd.DataFrame(out)
            dataframe.to_csv(f'{OUT}/transcripts_{i}.csv')
        elif i%save_itt == 0:
            dataframe = pd.DataFrame(out)
            if i==0: continue
            print(f"saving csv to blob at {i}")
            dataframe.to_csv(f'{OUT}/transcripts_{i}.csv')
    if shutdown:
        # shutdown machine (GPUs are expensive)
        os.system('sudo shutdown now')

def main():
    print("\n §loading data... \n")  
    data = load_data()
    print("\n data loaded! \n")
    print("\n §loading pipeline... \n")
    trans = load_pipeline(data)
    print("\n §pipeline loaded! \n")
    if LOGITS:
        print("\n §loading model, processor... \n")
        hubert_model, hubert_processor = load_model_processor()
        print("\n §model, processor loaded! \n")
        print("\n §running... \n")
        run(dataset=data, transcripts=trans, model=hubert_model, processor=hubert_processor, mount=MOUNT ,start=0, save_itt=None, shutdown=True, limit=999999)
    else:
        run(dataset=data, transcripts=trans, model=None, processor=None, mount=MOUNT ,start=0, save_itt=None, shutdown=True, limit=999999)

    

if __name__ == "__main__":
  main() 
