import os
import numpy as np
import cv2
import random
import json
import sys, getopt
import torch
from PIL import Image
from aic.dataloaders import get_loader

import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

path = "data"

max_length = 30
embed_size = 256
hidden_size = 512
encoder_size = 2048


def generate(image, image_trans, word_map):
	t = 3
	sentences = []
	vocab_size = len(word_map)
	rev_word_map = {v: k for k, v in word_map.items()}

	captions = {}

	image_trans = image_trans.unsqueeze(0)

	sent_lstm = inference_plain(word_map=word_map,
								vocab_size=vocab_size, 
								image=image_trans, 
								epoch=t,
								mode="lstm")
	final_lstm = words_from_indices(sent_lstm[1:], rev_word_map)
	final_lstm = final_lstm[:-1]
	captions['lstm'] = " ".join(final_lstm)

	sent_gru = inference_plain(word_map=word_map, 
								vocab_size=vocab_size, 
								image=image_trans, 
								epoch=t,
								mode="gru")
	final_gru = words_from_indices(sent_gru[1:], rev_word_map)
	final_gru = final_gru[:-1]
	captions['gru'] = " ".join(final_gru)

	sent_greedy, alphas = inference_beam(word_map=word_map, 
							vocab_size=vocab_size, 
							image=image_trans, 
							epoch=t,
							beam_size=1,
							mode="attention")
	final_greedy = words_from_indices(sent_greedy[1:], rev_word_map)
	final_greedy = final_greedy[:-1]
	captions['gru_attention'] = " ".join(final_greedy)

	sent_beam, alphas = inference_beam(word_map=word_map, 
							vocab_size=vocab_size, 
							image=image_trans, 
							epoch=t,
							beam_size=3,
							mode="attention")
	final_beam = words_from_indices(sent_beam[1:], rev_word_map)
	final_beam = final_beam[:-1]
	captions['gru_attention_beam'] = " ".join(final_beam)

	return captions

def save_results(image, cap_array):
	fig, (ax1, ax2) = plt.subplots(1, 2)
	cap_string = " ".join(cap_array)
	ax1.imshow(image[0])
	ax2.text(0, 0, cap_string, verticalalignment='center', horizontalalignment='center')
	plt.axis("off")
	plt.show()
	
def plot_attention(image, alphas, caption):
	alphas = alphas[1:-1]
	caplen = len(caption)
	fig1 = plt.gcf()
	plt.plot()
	sprite = np.float32(np.array(image))/255

	for t in range(caplen):
		canvas = np.array(alphas[t])
		plt.subplot(1+(caplen-1)//7, 7, t+1)
		plt.title(caption[t])

		sprite = cv2.resize(sprite, dsize=(96, 96), interpolation=cv2.INTER_LINEAR)
		canvas = cv2.resize((1/np.max(canvas))*canvas, dsize=(96, 96), interpolation=cv2.INTER_LINEAR)
		canvas = np.array([canvas, canvas, canvas]).transpose((1, 2, 0))
		mask = sprite*canvas
		plt.imshow(mask)
		plt.axis('off')
	
	plt.tight_layout()
	plt.show()


def words_from_indices(sent, word_map):
	return [word_map[w] for w in sent]

def inference_plain(word_map, vocab_size, image, epoch, mode, states=None):
	encoder, decoder = load_module(mode=mode, epoch=epoch, vocab_size=vocab_size)
	if mode == "lstm":
		decode_step = decoder.lstm
	else:
		decode_step = decoder.gru

	sampled_ids = []
	features = encoder(image)
	
	inputs = features.unsqueeze(1)
	for t in range(max_length):
		out, states = decode_step(inputs, states)
		out = decoder.fc(out.squeeze(1))
		_, predicted = out.max(1)
	   
		sampled_ids.append(predicted.item())
		if predicted.item() == word_map['<end>']:
			break
		inputs = decoder.embed(predicted)
		inputs = inputs.unsqueeze(1)
	
	return sampled_ids

def inference_beam(word_map, vocab_size, image, epoch, beam_size, mode, states=None):
	encoder, decoder = load_module(mode=mode, epoch=epoch, vocab_size=vocab_size)
	sampled_ids = []
	
	features = encoder(image)
	features = features.view(features.size(0), -1, features.size(-1))

	features = features.expand(beam_size, features.size(1), features.size(2))
	sent, alphas = beam_search(encoder_out=features, k=beam_size, decoder=decoder, word_map=word_map, vocab_size=vocab_size)

	return sent, alphas

def beam_search(encoder_out, k, decoder, word_map, vocab_size):
	enc_image_size = 14
		# Tensor to store top k previous words at each step; now they're just <start>
	k_prev_words = torch.LongTensor([[word_map['<start>']]] * k).to(device)  # (k, 1)

	# Tensor to store top k sequences; now they're just <start>
	seqs = k_prev_words  # (k, 1)

	# Tensor to store top k sequences' scores; now they're just 0
	top_k_scores = torch.zeros(k, 1).to(device)  # (k, 1)

	# Tensor to store top k sequences' alphas; now they're just 1s
	seqs_alpha = torch.ones(k, 1, enc_image_size, enc_image_size).to(device)  # (k, 1, enc_image_size, enc_image_size)

	# Lists to store completed sequences, their alphas and scores
	complete_seqs = list()
	complete_seqs_alpha = list()
	complete_seqs_scores = list()

	# Start decoding
	step = 1
	h = decoder.init_hidden_state(encoder_out)

	# s is a number less than or equal to k, because sequences are removed from this process once they hit <end>
	while True:
		embeddings = decoder.embed(k_prev_words).squeeze(1)  # (s, embed_dim)
		awe, alpha = decoder.attention(encoder_out, h)  # (s, encoder_dim), (s, num_pixels)
		alpha = alpha.squeeze(-1)  # (s, enc_image_size, enc_image_size)
		alpha = alpha.view(-1, enc_image_size, enc_image_size)

		gate = decoder.sigmoid(decoder.f_beta(h))  # gating scalar, (s, encoder_dim)
		awe = gate * awe
		h = decoder.gru(torch.cat([awe, embeddings], dim=1), h)  # (s, decoder_dim)

		scores = decoder.fc1(decoder.dropout(h))  # (s, vocab_size)

		# Add
		scores = top_k_scores.expand_as(scores) + scores  # (s, vocab_size)

		# For the first step, all k points will have the same scores (since same k previous words, h, c)
		if step == 1:
			top_k_scores, top_k_words = scores[0].topk(k, 0, True, True)  # (s)
		else:
			# Unroll and find top scores, and their unrolled indices
			top_k_scores, top_k_words = scores.view(-1).topk(k, 0, True, True)  # (s)

		# Convert unrolled indices to actual indices of scores
		prev_word_inds = top_k_words // vocab_size  # (s)
		next_word_inds = top_k_words % vocab_size  # (s)

		# Add new words to sequences, alphas
		seqs = torch.cat([seqs[prev_word_inds], next_word_inds.unsqueeze(1)], dim=1)  # (s, step+1)
		seqs_alpha = torch.cat([seqs_alpha[prev_word_inds], alpha[prev_word_inds].unsqueeze(1)],
							   dim=1)  # (s, step+1, enc_image_size, enc_image_size)

		# Which sequences are incomplete (didn't reach <end>)?
		incomplete_inds = [ind for ind, next_word in enumerate(next_word_inds) if
						   next_word != word_map['<end>']]
		complete_inds = list(set(range(len(next_word_inds))) - set(incomplete_inds))

		# Set aside complete sequences
		if len(complete_inds) > 0:
			complete_seqs.extend(seqs[complete_inds].tolist())
			complete_seqs_alpha.extend(seqs_alpha[complete_inds].tolist())
			complete_seqs_scores.extend(top_k_scores[complete_inds])
		k -= len(complete_inds)  # reduce beam length accordingly

		# Proceed with incomplete sequences
		if k == 0:
			break
		seqs = seqs[incomplete_inds]
		seqs_alpha = seqs_alpha[incomplete_inds]
		h = h[prev_word_inds[incomplete_inds]]
		encoder_out = encoder_out[prev_word_inds[incomplete_inds]]
		top_k_scores = top_k_scores[incomplete_inds].unsqueeze(1)
		k_prev_words = next_word_inds[incomplete_inds].unsqueeze(1)

		# Break if things have been going on too long
		if step > 50:
			break
		step += 1

	i = complete_seqs_scores.index(max(complete_seqs_scores))
	seq = complete_seqs[i]
	alphas = complete_seqs_alpha[i]

	return seq, alphas

def load_module(mode, epoch, vocab_size):
	if mode == "lstm":
		from aic.model import EncoderCNN, DecoderRNN
	elif mode == "gru":
		from aic.model_gru import EncoderCNN, DecoderRNN
	else:
		from aic.model_with_bahdanau import EncoderCNN, DecoderRNN

	encoder = EncoderCNN(embed_size=embed_size)
	decoder = DecoderRNN(embed_size=embed_size, vocab_size=vocab_size, hidden_size=512)

	encoder.load_state_dict(torch.load(os.path.join(path, 'pth', f"encoder-{mode}-{epoch}.pth"), map_location=torch.device('cpu')))
	decoder.load_state_dict(torch.load(os.path.join(path, 'pth', f"decoder-{mode}-{epoch}.pth"), map_location=torch.device('cpu')))

	encoder.to(device)
	decoder.to(device)

	encoder.eval()
	decoder.eval()

	return encoder, decoder


if __name__ == "__main__":
	opt_hint = "generate.py -p <dataset path>"
	question = "Una altra ronda? "
	try:
		opts, args = getopt.getopt(sys.argv[1:], "hp:", ["help", "path="])
	except getopt.err as err:
		print("Wrong combination of arguments. Expected:", opt_hint)
		sys.exit(2)
	
	for opt, arg in opts:
		if opt == "-h":
			print("Generate captions from all models, all training stages")
			print("Usage: ", opt_hint)
		elif opt in ("-p", "--path"):
			path = arg

	try:
		with open(os.path.join(path, 'json', 'wordmap.json'), 'r') as j:
			word_map = json.load(j)
	except FileNotFoundError:
		print(f"Word map file 'wordmap.json' not found in {path}/json. Did you run prepare.py first?")
		sys.exit(2)


	data_loader = get_loader(path, "test", 1)

	while True:
		i, (image, image_trans, caption, _, photoid) = next(enumerate(data_loader))

		generate(image=image, 
				image_trans=image_trans, 
				caption=caption, 
				word_map=word_map,
				photoid=photoid)
		
		ans = str(input(question+'(y): ')).lower().strip()
		if ans[:1] == 'y':
			continue
		else:
			sys.exit(1)