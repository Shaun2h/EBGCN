import os
from Process.dataset import BiGraphDataset, bigraph_dataset_PHEME
cwd=os.getcwd()


################################### load tree#####################################
def loadTree(dataname):
    if 'Twitter' in dataname:
        treePath = os.path.join(cwd,'data/'+dataname+'/data.TD_RvNN.vol_5000.txt')
        print("reading twitter tree")
        treeDic = {}
        for line in open(treePath):
            line = line.rstrip()
            eid, indexP, indexC = line.split('\t')[0], line.split('\t')[1], int(line.split('\t')[2])
            max_degree, maxL, Vec = int(line.split('\t')[3]), int(line.split('\t')[4]), line.split('\t')[5]
            if not treeDic.__contains__(eid):
                treeDic[eid] = {}
            treeDic[eid][indexC] = {'parent': indexP, 'max_degree': max_degree, 'maxL': maxL, 'vec': Vec}
        print('tree no:', len(treeDic))

    if dataname == "Weibo":
        treePath = os.path.join(cwd,'data/Weibo/weibotree.txt')
        print("reading Weibo tree")
        treeDic = {}
        for line in open(treePath):
            line = line.rstrip()
            eid, indexP, indexC,Vec = line.split('\t')[0], line.split('\t')[1], int(line.split('\t')[2]),line.split('\t')[3]
            if not treeDic.__contains__(eid):
                treeDic[eid] = {}
            treeDic[eid][indexC] = {'parent': indexP, 'vec': Vec}
        print('tree no:', len(treeDic))
    return treeDic

################################# load data ###################################

def loadBiData(dataname, treeDic, fold_x_train, fold_x_test, TDdroprate,BUdroprate,picklefear=True):
    ispheme="PHEME" in dataname
    data_path = os.path.join(cwd,'data', dataname + 'graph')
    print("loading train set", )
    if ispheme:
        traindata_list = bigraph_dataset_PHEME(fold_x_train, treeDic, tddroprate=TDdroprate, budroprate=BUdroprate, picklefear=picklefear)
    else:
        traindata_list = BiGraphDataset(fold_x_train, treeDic, tddroprate=TDdroprate, budroprate=BUdroprate, data_path=data_path)
    print("train no:", len(traindata_list))
    print("loading test set", )
    if ispheme:
        testdata_list = bigraph_dataset_PHEME(fold_x_test, treeDic, picklefear=picklefear)
    else:
        testdata_list = BiGraphDataset(fold_x_test, treeDic, data_path=data_path)
    print("test no:", len(testdata_list))
    return traindata_list, testdata_list



