# azure

#run these commands in cli first from https://docs.microsoft.com/en-us/azure/storage/blobs/storage-how-to-mount-container-linux
#wget https://packages.microsoft.com/config/ubuntu/18.04/packages-microsoft-prod.deb
#sudo dpkg -i packages-microsoft-prod.deb
#sudo apt-get update
#sudo apt-get install blobfuse
#sudo apt-get install fuse
#pip install -U fusepy --user



import azureml.core, os
from azureml.core import Workspace, Datastore, Dataset, Run
from azureml.data.datapath  import DataPath
from azureml.data.data_reference import DataReference
from azureml.exceptions import UserErrorException
import subprocess

# load subscription info from config.json
ws = Workspace.get(name='asr-colab', subscription_id='1dc6758e-9473-4f1a-b378-b9247db6b6f1', resource_group='asr')

# to be able to write on mounted datastore
import fuse

user = os.popen('whoami').read()[:-1]
#mount_path must be empty
mount_path   = "/content/data" #📌 must change for user 
if not os.path.exists(mount_path): os.mkdir(mount_path)
if os.listdir(mount_path) != []:
    print("mount_path not empty")

#make sure user has write access to cache_path - if not, create & chown to your user.
cache_path   = "/content/data/tmp"
if not os.path.exists(cache_path): os.mkdir(cache_path)

config_path  = "/content/connection.cfg"
if not os.path.exists(cache_path):
    print("get connection file from https://github.com/elucidate-ai/asr/blob/main/pipeline/connection.cfg")


if __name__ == "__main__":
    


    bashCommand = "blobfuse /content/data --tmp-path=/content/tmp  --config-file=/content/connection.cfg -o nonempty"
    process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
    output, error = process.communicate()
