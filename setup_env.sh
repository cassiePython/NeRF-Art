conda create -n nerfart python=3.8 -y
conda activate nerfart 

conda install pytorch==1.8.1 torchvision==0.9.1 torchaudio==0.8.1 cudatoolkit=10.2 -c pytorch -y
# if you use RTX 30XX GPUs, please use the following command instead
# conda install pytorch==1.8.1 torchvision==0.9.1 torchaudio==0.8.1 cudatoolkit=11.3 -c pytorch -c conda-forge 
pip install -r requirements.txt 
pip install git+https://github.com/openai/CLIP.git 

mkdir pretrained
