#!/usr/bin/env python
# -*- coding: utf-8 -*-


import sys
import numpy as np
import torch

from hparams import create_hparams
from model import Tacotron2
from layers import TacotronSTFT, STFT
from audio_processing import griffin_lim
from train import load_model
from tqdm import tqdm
from text import text_to_sequence
from argparse import ArgumentParser

from pytorch_inspector.extractor import ActivationsExtractor
from pytorch_inspector.structure import load_model_config, save_model_config
from pytorch_inspector.opts import Mode, check_opts, add_opts


def fileRead(fname):
		
		sentences=[]
		
		with open(fname) as f:
				for line in f:
					line = line.rstrip('\r\n')
					sentences.append(line)      
					
		return sentences


def main(opt):
	
	hparams = create_hparams()
	hparams.sampling_rate = 22050
	checkpoint_path = "tacotron2_myModel.pt"
	model = load_model(hparams)
	model.load_state_dict(torch.load(checkpoint_path)['state_dict'])
	_ = model.cuda().eval().half()

	check_opts(opt)

	if opt.mode == Mode.extract_structure:
		save_model_config(opt.config_file, model)
		return

	model_config = load_model_config(opt.config_file)
	extractor = ActivationsExtractor(model_config,
									 model,
									 opt.output_activations,
									 opt.activations_shard_size)

	sentences = []
	sentences = fileRead(opt.src)

	for i, text in tqdm(enumerate(sentences)):
		sequence = np.array(text_to_sequence(text, ['english_cleaners']))[None, :]
		sequence = torch.autograd.Variable(torch.from_numpy(sequence)).cuda().long()
		mel_outputs, mel_outputs_postnet, _, alignments = model.inference(sequence)
		extractor.save_activations()


if __name__ == "__main__":

	parser = ArgumentParser()
	add_opts(parser)
	parser.add_argument("-src", "--src", dest="src", help="file containing source text", metavar="FILE")
	parser.add_argument("-batch_size", "--batch_size", dest="batch_size", help="Batch size is required for nothing, just put 1", metavar="1")
	
	opt = parser.parse_args()
	main(opt)
