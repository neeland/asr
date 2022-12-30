Authors: neelan.pather@gmail.com

![149741504-67ef2b85-8e12-47fd-a681-adefe3a4ec5e-removebg-preview](https://user-images.githubusercontent.com/97616162/158749972-4d86af31-7f4a-4d0b-b5a3-d0bda799a0f3.png)

# 🎙 ASR
Automatic Speech Recognition 

<img width="442" alt="156428778-ed57f87b-3c4f-4af4-88b9-f31360553a8e-removebg-preview" src="https://user-images.githubusercontent.com/97616162/158749997-1305c19c-a2bd-4d04-aa96-1a51fffa3ff6.png">

Code to implement Hugging Face (🤗) `pipeline` on Azure machines, transcribing `.wav` (converted from client's `.amr`)

**TODO:** _items are throughout code & documentation_



### Client data challenges

* South African accents are very thick → need for fine-tuning
* Code-switching between English & other African languages
* Some audio is completely inaudible
* Some audio is completely in a different language - need for classification here

### Solution chosen

* Currently very good ASR Transformer based solutions are being open-sourced. For example [Wav2Vec2](https://huggingface.co/blog/fine-tune-wav2vec2-english), [Whisper](https://openai.com/blog/whisper/), with [open-source competitions](https://discuss.huggingface.co/t/open-to-the-community-robust-speech-recognition-challenge/13614) being held, resulting in open access to [high quality models](https://paperswithcode.com/sota/automatic-speech-recognition-on-librispeech-2).

    ### Best performing open source model tested on client data

    * [`patrickvonplaten/hubert-xlarge-ls960-ft-4-gram`](https://huggingface.co/patrickvonplaten/hubert-xlarge-ls960-ft-4-gram)

    * **TODO:** Haven't tried current state-of- art: [`openai/whisper-large`](https://huggingface.co/openai/whisper-large) 
# Contents
Each folder has a markdown `.md` file explaining each file in the folder

```
asr
|
├── dev       : development
│ 
├── mount     : mounted input files
│ 
├── prod      : production process
│ 
└── README.md : >> you are here <<

```
## Folder `README` links
Describes relevant folder's files
* [`./dev/README.md`](dev/README.md)
* [`./mount/README.md`](mount/README.md)
* [`./prod/README.md`](prod/README.md)

# ▶️ Inference instructions

* [📽 `demo_videos` 2️⃣ → 4️⃣](https://user-images.githubusercontent.com/97616162/196622195-37d9438b-3c21-4477-b979-b501899a3cb8.mp4)
* [📽 `demo_videos` 5️⃣](https://user-images.githubusercontent.com/5680639/196697073-3aa41c80-2c34-4d0c-a71d-c39786f9335d.mp4)



To transcribe a new batch of client data...

1️⃣ Connect to client SFTP, download necessary data locally, convert `.amr` to `.wav` using:

* [Cyber Duck](https://cyberduck.io/download/) is a stand-alone app for SFTP connection. Download files locally using SFTP 

    * **TODO:** _automate client SFTPT → blob process, triggering conversion & inference when new data appears_
    
*  [`prod/inference/env/setup_amr2wav.sh`](https://github.com/elucidate-ai/asr/blob/main/prod/inference/env/setup_amr2wav.sh) to set up environment
* [`prod/inference/amr2wav.py`](https://github.com/elucidate-ai/asr/blob/main/prod/inference/amr2wav.py) to convert `.amr` → `.wav`
    ```
    > git clone https://github.com/elucidate-ai/asr
    > cd asr/prod/inference
    > bash env/setup_inference2csv.sh
    python inference2csv.py
    ```


Then upload to Azure blob storage

* [Storage Explore](https://azure.microsoft.com/en-us/products/storage/storage-explorer/) is a stand-alone app for interacting with Azure blob storage

* [`mount/connection.cfg`](mount/connection.cfg) is connection config to current storage blob used


2️⃣ Create appropriate Azure GPU machine for inference
 * [Azure ML Portal]([https://ml.azure.com) used to create machines (_Compute > + New_)
    
    **Note on GPU needed**
    
    Only following Azure ML machines will work for such large models: 
    > 1 x NVIDIA Tesla P100 

    > 1 x NVIDIA Tesla V100 
    

3️⃣ Connect to terminal of the machine you just created & clone repo
    
* [VS Code Azure extensions make this easy](https://code.visualstudio.com/docs/azure/extensions)
    ```
    > git clone https://github.com/elucidate-ai/asr
    > cd asr
    ```
4️⃣ Mount Azure storage blob using [`mount/mount_blob.py`](mount/mount_blob.py)
* See [`mount/mount_README.md`](mount/README.md) for more information

5️⃣ Run production  

* See [`prod/prod_README.md`](prod/prod_README.md) for more details

* First parse the `.wav` files into an index `.csv` using:
    * [`prod/inference/env/setup_parse2csv.sh`](prod/inference/parse2csv.py) to set up environment
    * [`prod/inference/parse2csv.py`](prod/inference/parse2csv.py) to parse input `.wav` files to `.csv`

* Run inference to create output `.csv` of transcriptions
    * [`prod/inference/env/setup_inference2csv.sh`](prod/inference/inference2csv.py) to set up environment
    * [`prod/inference/inference2csv.py`](prod/inference/inference2csv.py) to transcribe `.wav` files to a bulk transcription `.csv` output using following command to allow process to run in background & create log `logs/inference2csv.out`:
        ```
        > nohup python inference2csv.py > logs/inference2csv.out&
        ```
        **TODO** _Currently cuts out after 2058 interations. Seems blob unmounts at that point. As such I tried to do blob mounting best practices & mount at home [`mount/mount_blob_home.py`](mount/mount_blob_home.py); this does not help at still cuts off at 2058 interations 😓 



## Useful links
* [A primer on audio data in notebooks](https://musicinformationretrieval.com/ipython_audio.html)

* [Fine-Tune Wav2Vec2 for English ASR with 🤗 Transformers](https://huggingface.co/blog/fine-tune-wav2vec2-english)

* [Boosting Wav2Vec2 with n-grams in 🤗 Transformers](https://huggingface.co/blog/wav2vec2-with-ngram)

* [Fine-tuning XLS-R for Multi-Lingual ASR with 🤗 Transformers](https://huggingface.co/blog/fine-tune-xlsr-wav2vec2)

* [Making automatic speech recognition work on large files with Wav2Vec2 in 🤗 Transformers](https://huggingface.co/blog/asr-chunking)

* [Pipelines for inference tutorial](https://huggingface.co/docs/transformers/pipeline_tutorial)

* [Robust Speech Challenge 🤗 Discord announcement](https://discord.com/channels/879548962464493619/897387888663232554/933348413985132596) 

* [Robust Speech Challenge 🤗 GitHub](https://github.com/huggingface/transformers/tree/main/examples/research_projects/robust-speech-event)

* [Introducing Whisper](https://openai.com/blog/whisper/) current state-of-the-art ⚡️
