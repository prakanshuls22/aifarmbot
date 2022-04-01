# this function will have some dependencies of the model files 
# load the weights into the model 
# write the inference function 
# expect input to be numpy array 

import torch 
from torch.nn.utils.rnn import pack_padded_sequence
from models import *
from caption import *


emb_dim = 112  
attention_dim = 512  # dimension of attention linear layers
decoder_dim = 512  # dimension of decoder RNN
dropout = 0.5


def inference(img):

  '''
  img : numpy array received 

  '''

  with open("/content/gdrive/MyDrive/CSD_project/dataset/WORDMAP_1_cap_per_img_2_min_word_freq.json", 'r') as j:
    word_map = json.load(j)

  
  checkpoint = torch.load("/content/gdrive/MyDrive/CDD/new plant diseases dataset(augmented)/New Plant Diseases Dataset(Augmented)/checkpoint_five_disease_csd.pt", map_location = "cpu")

  encoder = Encoder()
  encoder.load_state_dict(checkpoint["encoder"])

  encoder = encoder.eval()

  decoder = DecoderWithAttention(attention_dim=attention_dim,embed_dim=emb_dim,decoder_dim=decoder_dim,vocab_size=len(word_map),dropout=dropout)
  decoder.load_state_dict(checkpoint["decoder"])

  decoder = decoder.eval()

  my_caption,_ = caption_image_beam_search(encoder, decoder, img, word_map, beam_size=3)
  


  reverse_map = {}
  for key in word_map:
    reverse_map[word_map[key]] = key


  caption = []
  for i,preds in enumerate(my_caption):
      caption.append(reverse_map[preds])

  return caption



  



