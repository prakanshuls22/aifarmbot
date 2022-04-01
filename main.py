from lib2to3.pytree import Base
import string
from pydantic import BaseModel
from fastapi import FastAPI, File, UploadFile
import pickle
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import numpy as np
from visions import Object
import json
import torch 
from torch.nn.utils.rnn import pack_padded_sequence
from models import *
from caption import *

app = FastAPI()
# to run fast API:- uvicorn main:app --reload
app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*'],
)

class Labels(BaseModel):
    symptoms: list

class img_arr(BaseModel):
    array: list

@app.on_event('startup')
def image_to_symptoms(data: img_arr):
    global emb_dim, attention_dim, decoder_dim, dropout
    
    '''
    img : numpy array received 

    '''
    temp = data.dict()
    img = np.asarray(temp["array"])
    emb_dim = 112
    attention_dim = 512
    decoder_dim= 512
    dropout = 0

    with open("WORDMAP_1_cap_per_img_2_min_word_freq.json", 'r') as j:
        word_map = json.load(j)

    
    checkpoint = torch.load("checkpoint_five_disease_csd.pt", map_location = "cpu")

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
    
    temp = {}
    temp["symptoms"] = caption
    return temp


@app.on_event('/preprocess')
def data_preprocess():
    global apple_scab_symptoms, apple_black_rot, tomato_bacterial_spot, tomato_late_blight, tomato_early_blight, final_set
    global asc_indexes, abr_indexes, tbs_indexes, tlb_indexes, teb_indexes, dis_sym_matrix
    
    apple_scab_symptoms = ['black-spots','circular-spots','spots-covering-entire-surface','pale-spots','fused-spots','yellow-spots','olive-green-spots','velvety-spots','yellow-leaf','twisted-leaf']
    apple_black_rot = ['dark-brown-lesions-purple-margin','irregular-lesions','chlorotic-leaf','circular-spots','purple-red-edge-light-tan-centers','small-purple-spots']
    tomato_bacterial_spot = ['greasy-spots','circular-spots','small-brown-spots','irregular-lesions','rough-lesions','yellowish-halo-around-spots']
    tomato_early_blight = ['yellow-tissue-around-spots','black-lesions','concentric-rings-inside-lesions']
    tomato_late_blight = ['dark-brown-patches','powdery-white-fungal-growth','curled-leaf']

    final_set = set(tomato_bacterial_spot)
    final_set = final_set.union(set(tomato_late_blight))
    final_set = final_set.union(set(tomato_early_blight))
    final_set = final_set.union(set(apple_scab_symptoms))
    final_set = final_set.union(set(apple_black_rot))

    asc_indexes = []
    abr_indexes = []
    teb_indexes = []
    tlb_indexes = []
    tbs_indexes = []

    for w,x in enumerate(sorted(final_set)):

        for i,j in enumerate(apple_scab_symptoms):
            if(x==j):
                asc_indexes.append(w)
                
        for i,j in enumerate(apple_black_rot):
            if(x==j):
                abr_indexes.append(w)
                
        for i,j in enumerate(tomato_early_blight):
            if(x==j):
                teb_indexes.append(w)
                
        for i,j in enumerate(tomato_late_blight):
            if(x==j):
                tlb_indexes.append(w)    
                
        for i,j in enumerate(tomato_bacterial_spot):
            if(x==j):
                tbs_indexes.append(w)

        dis_sym_matrix = np.zeros((25,5))

    for x in range(0,5):
        if(x==0):
            for i,y in enumerate(asc_indexes):
                dis_sym_matrix[y,x] = 1
        
        if (x==1):
            for i,y in enumerate(abr_indexes):
                dis_sym_matrix[y,x] = 1

        if(x==2):
            for i,y in enumerate(teb_indexes):
                dis_sym_matrix[y,x] = 1

        if(x==3):
            for i,y in enumerate(tlb_indexes):
                dis_sym_matrix[y,x] = 1

        if(x==4):
            for i,y in enumerate(tbs_indexes):
                dis_sym_matrix[y,x] = 1

# @app.post('/test')
# async def get_disease_from_label(data: img_arr):
#     print(type(data))
#     temp = data.dict()
#     print(type(temp))
#     numpy_arr = np.asarray(temp["array"])
#     print(type(numpy_arr),numpy_arr.shape)


@app.post('/predict')
async def get_disease_from_label(data: Labels):
    print(data)
    received = data.dict()
    print(type(received),received)
    received = received["symptoms"]
    print(received)
    q_indexes = []
    for j,y in enumerate(received):
        for i,x in enumerate(sorted(final_set)):
            if(y==x):
                q_indexes.append(i)

    single_col_dis = np.zeros((25,))

    for x in q_indexes:
        single_col_dis[x] = 1

    dis_conifdence = []
    single_col_norm = np.dot(single_col_dis,single_col_dis.T)

    for x in range(0,5):
        test_norm = np.dot(dis_sym_matrix[:,x],dis_sym_matrix[:,x].T)
        product = np.dot(single_col_dis,dis_sym_matrix[:,x])/(test_norm*single_col_norm)
        dis_conifdence.append(product)
    print(dis_conifdence)
    disease_info = {
        0:{'dis_name': 'apple_scab_symptoms',
        'dis_rem':'preventing this disease'},
        
        1:{'dis_name':'apple_black_rot',
        'dis_rem':'preventing this disease'},
        
        2:{'dis_name':'tomato_early_blight',
        'dis_rem':'preventing this disease'},
        
        3:{'dis_name':'tomato_late_blight',
        'dis_rem':'preventing this disease'},
        
        4:{'dis_name':'tomato_bacterial_spot',
        'dis_rem':'preventing this disease'},
        
    }

    maxi = dis_conifdence[0]
    maxi_2 = dis_conifdence[0]
    prev_id = 0
    max_id = 0

    for i,x in enumerate(dis_conifdence):
        if(x>=maxi_2 and x<maxi):
            maxi_2 = x
            prev_id = i        
        if(x>=maxi):
            maxi_2 = maxi
            prev_id = max_id
            maxi = x
            max_id = i

    diseases = {}
    diseases['first'] = disease_info[max_id]['dis_name']
    diseases['second'] = disease_info[prev_id]['dis_name']

    return diseases

# @app.on_event("startup")
# def load_model():
#     global model,dataset,X,Y,scores,dt,df_comb,df_norm
#     print(1)
#     model = pickle.load(open("model_dt.pkl", "rb"))
#     dataset = pickle.load(open("dataset_symptoms_list.pkl", "rb"))
#     df_comb = pd.read_csv("dis_sym_dataset_comb.csv") # Disease combination
#     df_norm = pd.read_csv("dis_sym_dataset_norm.csv") # Individual Disease
#     X = df_comb.iloc[:, 1:]
#     Y = df_comb.iloc[:, 0:1]
#     dt = DecisionTreeClassifier()
#     dt = dt.fit(X, Y)
#     scores = cross_val_score(dt,X,Y,cv=5)

@app.get('/')
async def index():
    print("called here")
    return {'message': 'This is the homepage of the API '}


# @app.post('/predict')
# async def get_diseases_probability(data: Symptoms):
#     received = data.dict()
#     print(received)
#     symptoms = received['symptoms']
#     symptoms = symptoms.split(',')
#     print(symptoms)

#     sample_x = [0 for x in range(0,len(dataset))]
#     for val in symptoms:
#         sample_x[dataset.index(val)]=1
#     pred_names = model.predict_proba([sample_x])
#     print(pred_names,type(Y))
#     k = 10
#     diseases = list(set(Y['label_dis']))
#     print(diseases)
#     diseases.sort()
#     topk = pred_names[0].argsort()[-k:][::-1]
#     print(topk)
#     print("hello till here done")


#     topk_dict = {}
#     # Show top 10 highly probable disease to the user.
#     for idx,t in  enumerate(topk):
#         match_sym=set()
#         row = df_norm.loc[df_norm['label_dis'] == diseases[t]].values.tolist()
#         row[0].pop(0)

#         for idx,val in enumerate(row[0]):
#             if val!=0:
#                 match_sym.add(dataset[idx])
#         prob = (len(match_sym.intersection(set(symptoms)))+1)/(len(set(symptoms))+1)
#         prob *= mean(scores)
#         topk_dict[t] = prob
#     j = 0
#     topk_index_mapping = {}
#     Probable_diseases = []
#     topk_sorted = dict(sorted(topk_dict.items(), key=lambda kv: kv[1], reverse=True))
#     for key in topk_sorted:
#       diseases_dict = {}
#       prob = topk_sorted[key]*100
#       print(str(j) + " Disease name:",diseases[key], "\tProbability:",str(round(prob, 2))+"%")
#       topk_index_mapping[j] = key
#       diseases_dict[diseases[key]] = str(round(prob, 2))+"%"
#       Probable_diseases.append(diseases_dict)
#       j += 1
#     print(Probable_diseases)
#     return {'prediction': Probable_diseases}
#     # return {'data':'data'}