
#!/usr/bin/env/python3

# content 
"""Recipe for finetuning wav2vec-based ctc ASR system trained on librispeech.
The system employs wav2vec as its encoder. Decoding is performed with
ctc greedy decoder.
To run this recipe, do the following:
> python finetune_wav2vec_wandb.py hparams.yml
The neural network is trained on CTC likelihood target and character units
are used as basic recognition tokens. 

* Initial training is performed on the full LibriSpeech dataset (960 h). 
* Finetuning performed on transcriptions produced by running Trasaction Capital call recording through assembly.ai API to produce transcriptions for supervision


Authors
* neelan@elucidate.ai

* structure like https://github.com/speechbrain/speechbrain/blob/develop/recipes/LibriSpeech/ASR/CTC/train_with_wav2vec.py
* reference for checkpointing, wer https://github.com/speechbrain/speechbrain/blob/develop/templates/speech_recognition/ASR/train.py

"""

############################################################################
###################### importing needed packages ###########################
############################################################################

# basics
import os # to handle ocal files
os.environ["KALDI_ROOT"] = "/home/azureuser"
print(os.environ["KALDI_ROOT"])
import time
import string
import numpy as np
import logging
from pathlib import Path
import pandas as pd
import glob # unix style pathname pattern expansion
import json
import requests # to handle apis
from datetime import datetime as dt # timers 

# azure
import azureml.core, os
from azureml.core import Workspace, Datastore, Dataset, Run
from azureml.data.datapath  import DataPath
from azureml.data.data_reference import DataReference
from azureml.exceptions import UserErrorException

# load subscription info from config.json
ws = Workspace.from_config()

# to be able to write on mounted datastore
import fuse

# deep learning
import torch

# audio
import torchaudio
import speechbrain as sb
from speechbrain.utils.distributed import run_on_main

logger = logging.getLogger(__name__)


import sys
import logging
from hyperpyyaml import load_hyperpyyaml
from pathlib import Path
import wandb

# Define finetuning procedure

from speechbrain.lobes.features import Fbank

# Define fine-tuning procedure 
class EncDecFineTune(sb.Brain):
     
    def compute_forward(self, batch, stage):
        """Forward computations from the waveform batches to the output probabilities.
        
        Runs all the computation of the CTC + seq2seq ASR. It returns the
        posterior probabilities of the CTC and seq2seq networks.
         Arguments
        ---------
        batch : PaddedBatch
            This batch object contains all the relevant tensors for computation.
        stage : sb.Stage
            One of sb.Stage.TRAIN, sb.Stage.VALID, or sb.Stage.TEST.
        Returns
        -------
        predictions : dict
            At training time it returns predicted seq2seq log probabilities.
            If needed it also returns the ctc output log probabilities.
            At validation/test time, it returns the predicted tokens as well.
        """
        # Forcing cuda
        self.device = 'cuda'

        # We first move the batch to the appropriate device.
        batch = batch.to(self.device)
        wavs, wav_lens = batch.signal
        wavs, wav_lens = wavs.to(self.device), wav_lens.to(self.device)

        tokens_bos, _ = batch.tokens_bos

        # Forward pass
        feats = self.modules.compute_features(wavs)
        feats = self.modules.normalize(feats, wav_lens)
        feats.requires_grad = True


        # Running the encoder (prevent propagation to feature extraction)
        #encoded_signal = self.modules.encoder(feats.detach())
        x = self.modules.enc(feats).to(self.device)
        
        # Embed tokens and pass tokens & encoded signal to decoder
        e_in = self.modules.emb(tokens_bos) # y_in bos + tokens
        h, _ = self.modules.dec(e_in, x, wav_lens)
        
        e_in = e_in.to(self.device)
        h = h.to(self.device)

        # Output layer for seq2seq log-probabilities
        logits = self.modules.seq_lin(h).to(self.device)
        predictions = {"seq_logprobs": self.hparams.log_softmax(logits).to(self.device)}
        
        print("~~~~~",self.device)

        if stage == sb.Stage.VALID:
            valid_search = sb.decoders.S2SRNNBeamSearcher(
                embedding= self.modules.emb,
                decoder= self.modules.dec,
                linear= self.modules.seq_lin, #,
                #ctc_linear: !ref <ctc_lin>,
                bos_index=0,
                eos_index=0,
                blank_index=0,
                min_decode_ratio=0.0,
                max_decode_ratio=1.0,
                beam_size=8,
                eos_threshold=1.5,
                using_max_attn_shift=True,
                max_attn_shift=240,
                coverage_penalty=1.5,
                temperature=1.25).to(self.device)
            predictions["tokens"], _ = valid_search(x, wav_lens)
            
        elif stage == sb.Stage.TEST:
            test_search  = sb.decoders.S2SRNNBeamSearchLM(
                embedding = self.modules.emb,
                decoder  = self.modules.dec,
                linear   = self.modules.seq_lin, #,
                #ctc_linear: !ref <ctc_lin>,
                #language_model: !ref <lm_model>,
                bos_index = 0,
                eos_index = 0,
                blank_index = 0,
                min_decode_ratio=0.0,
                max_decode_ratio=1.0,
                beam_size=8,
                eos_threshold=1.5,
                using_max_attn_shift=True,
                max_attn_shift=240,
                coverage_penalty=1.5,
                #lm_weight:=0.50,
                #ctc_weight: !ref <ctc_weight_decode>, 
                temperature=1.25,
                temperature_lm=1.25).to(self.device)
            predictions["tokens"], _ = test_search(x, wav_lens)
            
          
        return predictions

    def compute_objectives(self, predictions, batch, stage):
        """Computes the loss (CTC+NLL) given predictions and targets."""
        self.device = 'cuda'
        
        p_seq = predictions["seq_logprobs"]

        ids = batch.id
        tokens_eos, tokens_eos_lens = batch.tokens_eos
        tokens, tokens_lens = batch.tokens

        loss = self.hparams.seq_cost(
            p_seq, tokens_eos, tokens_eos_lens)
        
        if stage != sb.Stage.TRAIN:
            # Converted predicted tokens from indexes to words
            predicted_words = [
                self.tokenizer.decode_ids(predictions).split(" ")
                for predictions in predictions["tokens"] 
            ]
            target_words = [words.split(" ") for words in batch.words]

            # Monitor word error rate and character error rated at
            # valid and test time.
            self.wer_metric.append(batch.id, predicted_words, target_words)
            print(self.wer_metric)
            wandb.log(self.wer_metric.summarize())
            self.cer_metric.append(batch.id, predicted_words, target_words)
            wandb.log(self.cer_metric.summarize())        
        return loss

    def fit_batch(self, batch):
        """Train the parameters given a single batch in input"""
        predictions = self.compute_forward(batch, sb.Stage.TRAIN)

        loss = self.compute_objectives(predictions, batch, sb.Stage.TRAIN)
        wandb.log({'loss': loss})
        loss.backward()

        if self.check_gradients(loss):
            self.optimizer.step()

        self.optimizer.zero_grad()

        return loss.detach()
    
    def evaluate_batch(self, batch, stage):
        """Computations needed for validation/test batches"""
        batch = batch.to(self.device)
        predictions = self.compute_forward(batch, stage=stage)

        with torch.no_grad():
            loss = self.compute_objectives(predictions, batch, stage=stage)
        return loss.detach()

    def on_stage_start(self, stage, epoch):
        # enable grad for all modules we want to fine-tune
        
        if stage == sb.Stage.TRAIN:
            for module in [self.modules.enc, self.modules.emb, self.modules.dec, self.modules.seq_lin]:
                for p in module.parameters():
                    p.requires_grad = True

        if stage != sb.Stage.TRAIN:
            self.cer_metric = self.hparams.cer_computer()
            self.wer_metric = self.hparams.error_rate_computer()

def on_stage_end(self, stage, stage_loss, epoch):
        """Gets called at the end of an epoch.
        Arguments
        ---------
        stage : sb.Stage
            One of sb.Stage.TRAIN, sb.Stage.VALID, sb.Stage.TEST
        stage_loss : float
            The average loss for all of the data processed in this stage.
        epoch : int
            The currently-starting epoch. This is passed
            `None` during the test stage.
        """

        # Store the train loss until the validation stage.
        stage_stats = {"loss": stage_loss}
        if stage == sb.Stage.TRAIN:
            self.train_stats = stage_stats

        # Summarize the statistics from the stage for record-keeping.
        else:
            stage_stats["CER"] = self.cer_metric.summarize("error_rate")
            stage_stats["WER"] = self.wer_metric.summarize("error_rate")

        # Perform end-of-iteration things, like annealing, logging, etc.
        if stage == sb.Stage.VALID:

            # Update learning rate
            old_lr, new_lr = self.hparams.lr_annealing(stage_stats["WER"])
            sb.nnet.schedulers.update_learning_rate(self.optimizer, new_lr)

            # The train_logger writes a summary to stdout and to the logfile.
            self.hparams.train_logger.log_stats(
                stats_meta={"epoch": epoch, "lr": old_lr},
                train_stats=self.train_stats,
                valid_stats=stage_stats,
            )

            # Save the current checkpoint and delete previous checkpoints.
            self.checkpointer.save_and_keep_only(
                meta={"WER": stage_stats["WER"]}, min_keys=["WER"],
            )

        # We also write statistics about test data to stdout and to the logfile.
        elif stage == sb.Stage.TEST:
            self.hparams.train_logger.log_stats(
                stats_meta={"Epoch loaded": self.hparams.epoch_counter.current},
                test_stats=stage_stats,
            )
            with open(self.hparams.wer_file, "w") as w:
                self.wer_metric.write_stats(w)

def dataio_prepare(hparams, asr_model):
    """This function prepares the datasets to be used in the brain class.
    It also defines the data processing pipeline through user-defined functions."""

    # Define audio pipeline. 
    #In this case, we simply read the path contained in the variable wav with the audio reader.

    @sb.utils.data_pipeline.takes("file_path")
    @sb.utils.data_pipeline.provides("signal")
    def audio_pipeline(file_path):
        """Load the audio signal. This is done on the CPU in the `collate_fn`."""
        signal = sb.dataio.dataio.read_audio(file_path)
        return signal

    # Define text pipeline:
    @sb.utils.data_pipeline.takes("words")
    @sb.utils.data_pipeline.provides(
            "words", "tokens_list", "tokens_bos", "tokens_eos", "tokens")
    def text_pipeline(words):
        yield words
        tokens_list = asr_model.tokenizer.encode_as_ids(words)
        yield tokens_list
        tokens_bos = torch.LongTensor([asr_model.hparams.bos_index] + (tokens_list))
        yield tokens_bos
        tokens_eos = torch.LongTensor(tokens_list + [asr_model.hparams.eos_index]) # we use same eos and bos indexes as in pretrained model
        yield tokens_eos
        tokens = torch.LongTensor(tokens_list)
        yield tokens

    # build text pipeline object
    from speechbrain.dataio.dataset import DynamicItemDataset

    # Define datasets from json data manifest file
    # Define datasets sorted by ascending lengths for efficiency
    datasets = {}
    data_folder = hparams["data_folder"]
    for dataset in ["train", "valid", "test"]:
        datasets[dataset] = sb.dataio.dataset.DynamicItemDataset.from_json(
            json_path=hparams[f"{dataset}_annotation"],
            replacements={"data_root": data_folder},
            dynamic_items=[audio_pipeline, text_pipeline],
            output_keys=["id","duration_seconds", "word_count", "words_per_second", "signal", "words", "tokens_list", "tokens_bos", "tokens_eos", "tokens"],
        )

    # Sorting training data with ascending order makes the code  much
    # faster  because we minimize zero-padding. In most of the cases, this
    # does not harm the performance.
    if hparams["sorting"] == "ascending":
        datasets["train"] =  datasets["train"].filtered_sorted(
            #sort_key=hparams["data"]["sort_key"],
            select_n=hparams["data"]["select_n"],
            key_min_value={hparams["data"]["sort_key"]: hparams["data"]["key_min_value"]}, 
            key_max_value={hparams["data"]["sort_key"]: hparams["data"]["key_max_value"]})
        hparams["train_dataloader_opts"]["shuffle"] = False

    elif hparams["sorting"] == "descending":
        datasets["train"] =  datasets["train"].filtered_sorted(
            sort_key=hparams["data"]["sort_key"],
            reverse=True,
            select_n=hparams["data"]["select_n"],
            key_min_value={hparams["data"]["sort_key"]: hparams["data"]["key_min_value"]}, 
            key_max_value={hparams["data"]["sort_key"]: hparams["data"]["key_max_value"]})
        hparams["train_dataloader_opts"]["shuffle"] = False

    elif hparams["sorting"] == "random":
        datasets["train"] =  datasets["train"].filtered_sorted(
            select_n=hparams["data"]["select_n"],
            key_min_value={hparams["data"]["sort_key"]: hparams["data"]["key_min_value"]}, 
            key_max_value={hparams["data"]["sort_key"]: hparams["data"]["key_max_value"]})
        hparams["train_dataloader_opts"]["shuffle"] = True
        pass

    else:
        raise NotImplementedError(
            "sorting must be random, ascending or descending"
        )
    return datasets

   

if __name__ == "__main__":

    #os.environ["CUDA_VISIBLE_DEVICES"]="0,1"

    # Reading command line arguments
    # CLI: load hyperparameters file with command-line overrides
    hparams_file, run_opts, overrides = sb.parse_arguments(sys.argv[1:])

    # Initialize ddp (useful only for multi-GPU DDP training)
    # If distributed_launch=True then
    # create ddp_group with the right communication protocol
    sb.utils.distributed.ddp_init_group(run_opts)
    
    # Load hyperparameters file with command-line overrides
    with open(hparams_file) as fin:
        hparams = load_hyperpyyaml(fin, overrides)
    
    # Start wandb
    wandb.init(project=hparams["wandb"]["project"],name=hparams["wandb"]["name"], entity = hparams["wandb"]["entity"], config=hparams)


    # Create experiment directory
    sb.create_experiment_directory(
        experiment_directory=hparams["output_folder"],
        hyperparams_to_save=hparams_file,
        overrides=overrides,
    )

    # Dataset prep (parsing Librispeech)
    #from librispeech_prepare import prepare_librispeech  # noqa

    # multi-gpu (ddp) save data preparation
    """
    run_on_main(
        prepare_librispeech,
        kwargs={
            "data_folder": hparams["data_folder"],
            "tr_splits": hparams["train_splits"],
            "dev_splits": hparams["dev_splits"],
            "te_splits": hparams["test_splits"],
            "save_folder": hparams["output_folder"],
            "merge_lst": hparams["train_splits"],
            "merge_name": "train.csv",
            "skip_prep": hparams["skip_prep"],
        },
    )
    """
    
    # here we create the datasets objects as well as tokenization and encoding

    from speechbrain.pretrained import EncoderDecoderASR
    asr_model =  EncoderDecoderASR.from_hparams(source=hparams["finetune_model"]["source"], savedir=hparams["finetune_model"]["save_model"])
    wandb.watch(asr_model)
    

    #print("~~~~~~~~~~")
    #print(os.environ["CUDA_VISIBLE_DEVICES"])
 
    datasets = dataio_prepare(hparams, asr_model)
    #print(datasets["train"][0])

    modules = {"enc": asr_model.mods.encoder.model, 
           "emb": asr_model.hparams.emb,
           "dec": asr_model.hparams.dec,
           "compute_features": asr_model.mods.encoder.compute_features, # we use the same features 
           "normalize": asr_model.mods.encoder.normalize,
           "seq_lin": asr_model.hparams.seq_lin, 
          }

    hparams_ ={"seq_cost": lambda x, y, z: sb.nnet.losses.nll_loss(x, y, z, label_smoothing=hparams['label_smoothing']),
               "log_softmax": sb.nnet.activations.Softmax(apply_log=True), "device": "cuda"}
    hparams_.update(hparams)
    # Trainer initialization
    brain = EncDecFineTune(modules, 
                           hparams=hparams_, 
                           opt_class=hparams["opt_class"],
                           run_opts=run_opts,
                           #checkpointer=hparams["checkpointer"],
                           )

    # We dynamicaly add the tokenizer to our brain class.
    # NB: This tokenizer corresponds to the one used for the LM!!
    brain.tokenizer = asr_model.tokenizer

    # Training
        # The `fit()` method iterates the training loop, calling the methods
    # necessary to update the parameters of the model. Since all objects
    # with changing state are managed by the Checkpointer, training can be
    # stopped at any point, and will be resumed on next call.
    brain.fit(
        hparams["epoch_counter"],
        datasets["train"],
        datasets["valid"],
        train_loader_kwargs=hparams["train_dataloader_opts"],
        valid_loader_kwargs=hparams["valid_dataloader_opts"])
    # Load best checkpoint for evaluation
    test_stats = brain.evaluate(
        test_set=datasets["test"],
        min_key="WER",
        test_loader_kwargs=hparams["test_dataloader_opts"],
    )

             