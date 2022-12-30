"""
Creates .wav files from .amr files; .wav files are needed for inference on huggingface models (wav2vec2 is a central model used)
Authors: neelan@elucidate.ai
"""
#import os,  subprocess, ffmpeg, sox, audiofile
#from pydub import AudioSegment
import os,  subprocess
from datetime import datetime

# change SRC & DST as appropriate
SRC = os.path.join(os.getcwd(),"amr/10k_sample_3/")
DST = os.path.join(os.getcwd(), "wav/10k_sample_3/")

def convert_amr_file(input_dir=SRC, output_dir=DST ,input_file_name='000000004328552#306#MASELAELOS#TCRLENA6-05#20220616092701830.amr', verbose=True, remove_intermediate=True):

    """
    converts single to intermediate .aud file before converting to .wav
    """
    # create an additional folder to store converted .wav files if doesn't already exist
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



def bulk_convert(audio_src=SRC, verbose=True, remove_intermediate=False):
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
    ERRORS = []
    # loop over each file in directory & convert to .wav
    for dirname, dirnames, filenames in os.walk(audio_src):
        for filename in filenames:
            input_path = os.path.join(dirname, filename)
            if verbose: print(f'input_path: {input_path}')
            try:
                convert_amr_file(input_dir=SRC, output_dir=DST ,input_file_name=filename, verbose=True, remove_intermediate=True)
              
                if verbose: print(f"\n !!! DONE CONVERTING {filename}!!! \n")
            except:
                if verbose:  print(f"\n !!! ERROR on {filename} !!! \n")
                ERRORS.append(filename)

def main():
    
    #uncomment out single file version to test 
    #convert_amr_file()
    bulk_convert()

if __name__ == "__main__":
    main()


