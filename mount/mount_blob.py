# azure
import azureml.core, os
from azureml.core import Workspace, Datastore, Dataset, Run
from azureml.data.datapath  import DataPath
from azureml.data.data_reference import DataReference
from azureml.exceptions import UserErrorException
import subprocess

# load subscription info from config.json
ws = Workspace.from_config()

# to be able to write on mounted datastore
import fuse

user = os.popen('whoami').read()[:-1]
#mount_path must be empty
mount_path   = "/home/azureuser/cloudfiles/code/Users/neelan/asr/pipeline/data" #📌 must change for user 
if not os.path.exists(mount_path): os.mkdir(mount_path)
if os.listdir(mount_path) != []:
    print("mount_path not empty")

#make sure user has write access to cache_path - if not, create & chown to your user.
cache_path   = "/home/azureuser/cloudfiles/data/tmp"
if not os.path.exists(cache_path): os.mkdir(cache_path)

config_path  = "/home/azureuser/cloudfiles/code/Users/neelan/asr/pipeline/connection.cfg"
if not os.path.exists(cache_path):
    print("get connection file from https://github.com/elucidate-ai/asr/blob/main/pipeline/connection.cfg")


if __name__ == "__main__":
    


    bashCommand = "blobfuse /home/azureuser/cloudfiles/code/Users/neelan/asr/pipeline/data --tmp-path=/home/azureuser/cloudfiles/data/tmp  --config-file=/home/azureuser/cloudfiles/code/Users/neelan/asr/pipeline/connection.cfg"
    process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
    output, error = process.communicate()
