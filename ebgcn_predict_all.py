import sys, os
import argparse
import json
import csv
import pprint
from datetime import datetime
from time import time
sys.path.append(os.getcwd())
import torch
import torch.nn.functional as F
import numpy as np
from tools.earlystopping import EarlyStopping
from torch_geometric.data import DataLoader
from torch_geometric.data import Data
from model.EBGCN import EBGCN
from transformers import BertTokenizer, BertModel

"""
This file is for creating an output csv file for a visualisation that is not available in this repository.
It also REQUIRES the pheme dataset's files within the directory to run.
"""


model_load_name = "ebgcnPHEME fold_4_iter_0.m"
def process_thread(somethread,returnalt=False):
        """
        Process a Thread in Shaun's Format to their required format so we can use a similar loading func like they do.
        """
        threadtextlist,tree,rootlabel,source_id = somethread
        threaddict = {}
        pointerdict = {}
        counter = 0 
        for text in threadtextlist:
            threaddict[text[1]] = text[0]
            pointerdict[text[1]] = counter #tweet id
            counter+=1

        fromrow = []
        torow = []
        for sender in tree:
            for target in tree[sender]:
                fromrow.append(pointerdict[sender])
                torow.append(pointerdict[target])
        invertedpointer = dict(map(reversed, pointerdict.items()))
        allinputs_list = [] 
        for numbered in range(len(list(pointerdict.keys()))):
            allinputs_list.append(threaddict[invertedpointer[numbered]])

        data ={}
        with torch.no_grad():
                allinputs_list = Bert_Tokeniser(list(allinputs_list), padding="max_length", max_length=256, truncation=True, return_tensors="pt")
                allinputs_list.to(device)
                allinputs_list = Bert_Embed(**allinputs_list)
                allinputs_list = allinputs_list[0].cpu()#.last_hidden_state
                allinputs_list = allinputs_list.cpu()

        with torch.no_grad():
            data['root'] = Bert_Embed(**Bert_Tokeniser(threaddict[source_id], padding="max_length", max_length=256, truncation=True, return_tensors="pt").to(device))[0].cpu()#.last_hidden_state.cpu()
            

        data["rootindex"] = pointerdict[source_id]
        data["x"] = allinputs_list # you also need to convert this to a tensor.
        data["y"] = rootlabel[0]
        data["edgeindex"] = np.array([fromrow,torow]) # imitating their dataloading method.

        return data




def nodrop_get_td_bu_edges(edge_index_matrix):
    new_edgeindex = edge_index_matrix
    burow = list(edge_index_matrix[1])
    bucol = list(edge_index_matrix[0])
    bunew_edgeindex = [burow,bucol]
    return new_edgeindex, bunew_edgeindex
    
    
def get_td_bu_edges(tddroprate,budroprate,edge_index_matrix):
    if tddroprate > 0:
        row = list(edge_index_matrix[0])
        col = list(edge_index_matrix[1])
        length = len(row)
        poslist = random.sample(range(length), int(length * (1 - tddroprate)))
        # poslist = random.sample(range(length), int(length * (1)))
        poslist = sorted(poslist)
        row = list(np.array(row)[poslist])
        col = list(np.array(col)[poslist])
        new_edgeindex = [row, col]
    else:
        new_edgeindex = edge_index_matrix

    burow = list(edge_index_matrix[1])
    bucol = list(edge_index_matrix[0])
    if budroprate > 0:
        length = len(burow)
        poslist = random.sample(range(length), int(length * (1 - budroprate)))
        # poslist = random.sample(range(length), int(length * (1)))
        poslist = sorted(poslist)
        row = list(np.array(burow)[poslist])
        col = list(np.array(bucol)[poslist])
        bunew_edgeindex = [row, col]
    else:
        bunew_edgeindex = [burow,bucol]
    return new_edgeindex, bunew_edgeindex

def run_model(tree,threadtextlist, source_id, rootlabel, model):
    """
    tree dictionary: {0:[1,2] 1: [3], 2:[], 3:[]}   0 -> 1 -> 3 and 0 -> 2
    list of string of tweets
    """

    data = process_thread((threadtextlist,tree,rootlabel,source_id))
    edgeindex = data['edgeindex']
    new_edgeindex, bunew_edgeindex = get_td_bu_edges(0,0,edgeindex)

    output_data =Data(x=data['x'].reshape(data["x"].shape[0],-1),
                edge_index=torch.LongTensor(new_edgeindex),BU_edge_index=torch.LongTensor(bunew_edgeindex),
                y=torch.LongTensor([int(data['y'])]), root=data['root'].reshape(data["root"].shape[0],-1),
                rootindex=torch.LongTensor([int(data['rootindex'])]))
    output_data.num_nodes = output_data.x.size(0) # prevents a torch scatter error.
    output_data.batch = torch.tensor([0]*output_data.x.shape[0]) # imitate torch geometric batch   
    output_data.to(device)


    with torch.no_grad():
        val_out, _, _ = model(output_data)
    _, pred = val_out.max(dim=-1)

    # print(val_out)
    # print(pred)
    # if pred==0:
        # print("rumour")
    # else:
        # print("non-rumour")
        

    return val_out,pred



if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--datasetname', type=str, default="Twitter16", metavar='dataname',
                        help='dataset name')
    parser.add_argument('--modelname', type=str, default="BiGCN", metavar='modeltype',
                        help='model type, option: BiGCN/EBGCN')
    parser.add_argument('--input_features', type=int, default=5000, metavar='inputF',
                        help='dimension of input features (TF-IDF)')
    parser.add_argument('--hidden_features', type=int, default=64, metavar='graph_hidden',
                        help='dimension of graph hidden state')
    parser.add_argument('--output_features', type=int, default=64, metavar='output_features',
                        help='dimension of output features')
    parser.add_argument('--num_class', type=int, default=4, metavar='numclass',
                        help='number of classes')
    parser.add_argument('--num_workers', type=int, default=0, metavar='num_workers',
                        help='number of workers for training')

    # Parameters for training the model
    parser.add_argument('--seed', type=int, default=2020, help='random state seed')
    parser.add_argument('--no_cuda', action='store_true',
                        help='does not use GPU')
    parser.add_argument('--num_cuda', type=int, default=0,
                        help='index of GPU 0/1')

    parser.add_argument('--lr', type=float, default=0.0005, metavar='LR',
                        help='learning rate')
    parser.add_argument('--lr_scale_bu', type=int, default=5, metavar='LRSB',
                        help='learning rate scale for bottom-up direction')
    parser.add_argument('--lr_scale_td', type=int, default=1, metavar='LRST',
                        help='learning rate scale for top-down direction')
    parser.add_argument('--l2', type=float, default=1e-4, metavar='L2',
                        help='L2 regularization weight')

    parser.add_argument('--dropout', type=float, default=0.5, metavar='dropout',
                        help='dropout rate')
    parser.add_argument('--patience', type=int, default=10, metavar='patience',
                        help='patience for early stop')
    parser.add_argument('--batchsize', type=int, default=128, metavar='BS',
                        help='batch size')
    parser.add_argument('--n_epochs', type=int, default=200, metavar='E',
                        help='number of max epochs')
    parser.add_argument('--iterations', type=int, default=50, metavar='F',
                        help='number of iterations for 5-fold cross-validation')

    # Parameters for the proposed model
    parser.add_argument('--TDdroprate', type=float, default=0.2, metavar='TDdroprate',
                        help='drop rate for edges in the top-down propagation graph')
    parser.add_argument('--BUdroprate', type=float, default=0.2, metavar='BUdroprate',
                        help='drop rate for edges in the bottom-up dispersion graph')
    parser.add_argument('--edge_infer_td', action='store_true', #default=False,
                        help='edge inference in the top-down graph')
    parser.add_argument('--edge_infer_bu', action='store_true', #default=True,
                        help='edge inference in the bottom-up graph')
    parser.add_argument('--edge_loss_td', type=float, default=0.2, metavar='edge_loss_td',
                        help='a hyperparameter gamma to weight the unsupervised relation learning loss in the top-down propagation graph')
    parser.add_argument('--edge_loss_bu', type=float, default=0.2, metavar='edge_loss_bu',
                        help='a hyperparameter gamma to weight the unsupervised relation learning loss in the bottom-up dispersion graph')
    parser.add_argument('--edge_num', type=int, default=2, metavar='edgenum', help='latent relation types T in the edge inference')
    global args
    args = parser.parse_args()
    args.datasetname = "pheme"
    if "pheme" in args.datasetname.lower():
        args.input_features=768*256 # bert shape.
        args.batchsize=1 # explosion prevention
        args.iterations=1 # just not do that iterations thing.

    if not args.no_cuda:
        print('Running on GPU:{}'.format(args.num_cuda))
        args.device = torch.device('cuda:{}'.format(args.num_cuda) if torch.cuda.is_available() else 'cpu')
    else:
        print('Running on CPU')
        args.device = torch.device('cpu')
    print(args)    
    
    global device
    device = "cuda:"+str(args.num_cuda) if torch.cuda.is_available() else "cpu"
    global Bert_Tokeniser 
    Bert_Tokeniser =  BertTokenizer.from_pretrained("bert-base-multilingual-uncased")
    global Bert_Embed 
    Bert_Embed = BertModel.from_pretrained("bert-base-multilingual-uncased").to(device)
    Bert_Embed.resize_token_embeddings(len(Bert_Tokeniser))

    if not args.no_cuda:
        print('Running on GPU:{}'.format(args.num_cuda))
        args.device = torch.device('cuda:{}'.format(args.num_cuda) if torch.cuda.is_available() else 'cpu')
    else:
        print('Running on CPU')
        args.device = torch.device('cpu')
    model = EBGCN(args).to(args.device)
    
    model.load_state_dict(torch.load(model_load_name))
    model.eval()
    # run_model({0:[1,2],1:[3],2:[],3:[]},[("First Tweet was about how we did stuff",0),("Second Tweet was reminiscing about how we did stuff",1),("Third Tweet was on something he missed on the thing.",2),("Fourth Tweet was about how second tweet could also think about something else.",3)],0,[0],model)

    with open("phemethreaddump.json","rb") as dumpfile:
        loaded_threads = json.load(dumpfile)
    allthreads = []

    for thread in loaded_threads:
        threadtextlist,tree,rootlabel,source_id = thread
        # print(tree)
        # print(threadtextlist)
        # input()
        val_out,pred = run_model(tree,threadtextlist,source_id,rootlabel,model)
        prediction = "rumour" if pred==0 else "non-rumour"
        # print(rootlabel)
        actual_label = "rumour" if rootlabel[0]==0 else "non-rumour"
        allthreads.append([source_id,prediction,actual_label])

        

    mainpath = os.path.join("all-rnr-annotated-threads")
    path_reference_dict = {}
    for eventwrap in os.listdir(mainpath):
        if eventwrap[0] == ".":
            continue
        for item in os.listdir(os.path.join(mainpath,eventwrap,"rumours")):
            if item[0]==".":
                continue
            path_reference_dict[item] = os.path.join(mainpath,eventwrap,"rumours",item)
        for item in os.listdir(os.path.join(mainpath,eventwrap,"non-rumours")):
            if item[0]==".":
                continue
            path_reference_dict[item] = os.path.join(mainpath,eventwrap,"non-rumours",item)

    treelist = []
    for i in allthreads:
        treeid = i[0]
        predicted = i[1]
        actual = i[2]
        
        readable = ['false', 'true', 'unverified']
        tree_path = path_reference_dict[str(treeid)]
        list_of_reactions = os.listdir(os.path.join(tree_path,"reactions"))
        tree_dict = {}
        with open(os.path.join(tree_path,"source-tweets",str(treeid)+".json"),"r",encoding="utf-8") as opened_source:
            loaded_source = json.load(opened_source)
            text = loaded_source["text"]
            source_id = loaded_source["id"]
            links = []
            tree_dict[source_id] = [text,source_id,links,predicted,actual,loaded_source["created_at"],loaded_source["user"]["screen_name"]]
            
        for item in list_of_reactions:
            if item[0] == ".":
                continue
            with open(os.path.join(tree_path,"reactions",item),"r",encoding="utf-8") as opened_reaction:
                
                reaction_dict = json.load(opened_reaction)
                reactiontext = reaction_dict["text"]
                reactionid = reaction_dict["id"]
                links = []
                reaction_target = reaction_dict["in_reply_to_status_id"]
                retweetedornot = reaction_dict["retweeted"]
                
                if not reactionid in tree_dict:
                    tree_dict[reactionid] = [reactiontext,reactionid,links,predicted,actual,reaction_dict["created_at"],reaction_dict["user"]["screen_name"]]
                else:
                    tree_dict[reactionid] = [reactiontext,reactionid,tree_dict[reactionid][2],predicted,actual,reaction_dict["created_at"],reaction_dict["user"]["screen_name"]]
                
                if reaction_target!="null":
                    if not reaction_target in tree_dict:
                        tree_dict[reaction_target] = [None,reaction_target,[[reactionid,reaction_target,"Reply"]],None,None,None,None]
                    else:
                        tree_dict[reaction_target][2].append([reactionid,reaction_target,"Reply"])
                    tree_dict[reactionid][2].append([reactionid,reaction_target,"Reply"])
                        
                        
                if retweetedornot:
                    if not reaction_target in tree_dict:
                        tree_dict[reaction_target] = [None,reaction_target,[[reactionid,reaction_target,"Retweet"]],None,None,None,None]
                    else:
                        tree_dict[reaction_target][2].append([reactionid,reaction_target,"Retweet"])
                    tree_dict[reactionid][2].append([reactionid,reaction_target,"Retweet"])
                    

        treelist.append(tree_dict)
    
    with open(model_load_name+"_ebgcn_all_predictions_dump.json","w",encoding="utf-8") as treedumpfile:
        # csvwriter = csv.writer(treedumpfile)
        fieldnames = ["Text","ID","Links","predicted","actual","timestamp","handle"]
        csvwriter = csv.DictWriter(treedumpfile, fieldnames=fieldnames)
        csvwriter.writeheader()
        for treeid in treelist:
            for node in treeid:

                timestampval = str(treeid[node][5])
                if timestampval!="None":
                # "Wed Jan 07 11:11:33 +0000 2015" -> 2012-02-23 09:15:26 +00:00
                    date = datetime.strptime(timestampval,"%a %b %d %H:%M:%S %z %Y").strftime("%Y-%m-%d %H:%M:%S %z")
                else:
                    date = "None"
                
                csvwriter.writerow({"Text":treeid[node][0], "ID":treeid[node][1], "Links":treeid[node][2],"predicted":treeid[node][3],"actual":treeid[node][4],"timestamp":treeid[node][5],"handle":treeid[node][6]})
    print("Dumped:",model_load_name+"_ebgcn_all_predictions_dump.json")
