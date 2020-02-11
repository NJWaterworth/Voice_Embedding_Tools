import os
import time
import torch
import numpy as np
import argparse
from mfcc import MFCC
import webrtcvad

from config import NETWORKS_PARAMETERS
from torch.utils.data import DataLoader
from network import get_network
from vad import read_wave, write_wave, frame_generator, vad_collector
from utils import rm_sil, get_fbank
import torch.nn.functional as F

# Used to turn a wav file to an embedding
def get_embedding(e_net, voice_file, vad_obj, mfc_obj, GPU=True):
	
	# remove silence
	vad_voice = rm_sil(voice_file, vad_obj)
	
	#MFCC options 
	fbank = get_fbank(vad_voice, mfc_obj)
	fbank = fbank.T[np.newaxis, ...]
	fbank = torch.from_numpy(fbank.astype('float32'))

	if GPU:
		fbank = fbank.cuda()
	embedding = e_net(fbank)
	embedding = F.normalize(embedding)
	
	return embedding

def main():
	parser = argparse.ArgumentParser(description="Splitting parameters")
	parser.add_argument('voice_list_dir', type=str, metavar='v', help="recording list dir")
	parser.add_argument('save_dir', type=str, metavar='s', help="file save directory")
	args = parser.parse_args()

	print('Loading Voice Sample...')
	print("performing filters/mfcc")
	vad_obj = webrtcvad.Vad(2)
	mfc_obj = MFCC(nfilt=64, lowerf=20., upperf=7200., samprate=16000, nfft=1024, wlen=0.025)

	print('Initializing networks...')
	e_net, e_optimizer = get_network('e', NETWORKS_PARAMETERS, train=False)

	with open(args.voice_list_dir) as file_list:
		line = file_list.readline()
		line = line.rstrip()
		i = 1
		while line:
			print('making embedding for {}'.format(line))
			embedding = get_embedding(e_net, line, vad_obj, mfc_obj)
			i += 1
			stuff = line.split("/")
			output_name = "{}_{}".format(stuff[6], stuff[7])
			output_name = output_name.replace(".wav", "")
			output_name = output_name + ".npy"
			save_dir = os.path.join(args.save_dir, output_name)
			print('Saving embedding to: {}'.format(save_dir))
			np.save(save_dir,embedding.cpu().detach().numpy(), allow_pickle=True)
	
			line = file_list.readline()
			line =line.rstrip()
	print("complete")

if __name__ == "__main__":
	main()

