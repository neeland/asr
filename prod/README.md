# `./asr/prod/*`

# Contents
Each folder has a markdown `.md` file explaining each file in the folder

```
asr
│ 
├── dev            : development
│ 
├── mount          : mounted input files
│ 
├── prod           : production process
│    │ 
│    └── README.md : >> you are here <<
│ 
└── README.md      : project documentation

```

---
## 5️⃣ Run production

---
* Folder contains all components needed to run  🤗 inference production `pipeline`:

* _Step assumes you have already_:

    *  Downloaded all client `.amr` files locally before converting them to `.wav` & then uploaded onto Azure blob storage
    * Mounted blob storage to Azure GPU machine being used
    * Parsed input`.wav` files must be parsed into a `.csv` index files

* **TODO:** _very crude method is used to scale; in `./inference/helpers` we split original `.csv` index file into 3 files and manually run in parallel_


## `./asr/prod/inference/*`
* `env` folder contains files needed to set up production environment and packages for each script 
    * [`inference/env/requirements.txt`](inference/env/requirements.txt) : Run following command to set up production environment called  `<env_name>`
    
        ```
        conda create --name <env_name> --file requirements.txt  
        ``` 
    
    * [`inference/env/setup_amr2wav.sh`](inference/env/setup_amr2wav.sh) : Run following command to set up environment to run `prod/inference/amr2wav.py`
        ```  
        bash env/setup_amr2wav.sh
        python amr2wav.py
        ```  
    * [`inference/env/setup_parse2csv.sh`](inference/env/setup_parse2csv.sh) : Run following command to set up environment to run to set up environment to run `prod/inference/parse2csv.py`

        ```
        bash env/setup_parse2csv.sh
        python parse2csv.py
        ```
    * [`inference/env/setup_inference2csv.sh`](inference/env/setup_inference2csv.sh) : Run following command to set up environment to run `prod/inference/inference2csv.py` in background & create log `logs/inference2csv.out`
        ```
        bash env/setup_inference2csv.sh
        nohup python inference2csv.py > logs/inference2csv.out&
        ``` 

## `./asr/prod/inference/helpers/*`
* Folder contains files that help in inference production
    * [`inference/helpers/helper_split_parse.py`](inference/helpers/helper_split_parse.py) used to split parsed `.csv` into smaller parsed input pieces for "parallel" process 
    
        **TODO:** _not a great scaling "parallel" process_**
