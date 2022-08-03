



import torch

import predict

from Audio_proc_lib.audio_proc_functions import load_music,sound_write

x,s = load_music() 

#target = "bass"
#est = predict.separate(audio=torch.tensor(x) , rate=s , model_str_or_path="/home/nnanos/Desktop/bass_dir" , targets=target , niter=1 , residual=True )
est = predict.separate(audio=torch.tensor(x) , rate=s , model_str_or_path="/home/nnanos/Desktop/Open_unmix_model_params/mymodel" , targets=["vocals","drums","bass","other"] ,  niter=1 )


#x_out = est[target].cpu().detach().numpy()[0].T


sound_write(x_out,s)