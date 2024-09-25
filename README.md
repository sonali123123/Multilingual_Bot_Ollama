## Download and Install  Python
https://www.python.org/downloads/ 

## Download and install anaconda 
https://www.anaconda.com/download

Set anaconda3/Scripts, anaconda3\Library\bin path in environment variable

## Download and Install Ollama
https://ollama.com/download/windows

Set Ollama folder path in environment variable

## Clone this  Multilingual_Bot_Ollama  repo in your vs code

### Create a conda environment using this command in command prompt inside project folder

conda create --name env

### Activate the environment using this command

conda activate env  (after this run all commands in this environment)

### Run this command 
conda install pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia   (change the cuda version if you have different cuda version in your device)

### Install the requirements.txt by running this command

pip install -r requirements.txt


### On the other hand , start other server in command prompt and pull ollama model, using this command

ollama pull llama3.1:8b   (After successfully pulling of this model , run the server)

### Using this command

ollama serve


### Now come back to your project repo, and start fastapi server using this command

fastapi run app.py

