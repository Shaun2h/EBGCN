import os
from Process.dataset import BiGraphDataset
cwd=os.getcwd()


################################### load tree#####################################
def loadTree(dataname):
    if 'Twitter' in dataname:
        treePath = os.path.join(cwd,'data/'+dataname+'/data.TD_RvNN.vol_5000.txt')
        print("reading twitter tree")
        treeDic = {}
        for line in open(treePath):
            # '656955120626880512	None	1	2	9	1:1 3:1 164:1 5:1 2282:1 11:1 431:1 473:1 729:1'
            #  eid, indexP, index_C, max_degree maxL Vec
            line = line.rstrip()
            eid, indexP, indexC = line.split('\t')[0], line.split('\t')[1], int(line.split('\t')[2])
            max_degree, maxL, Vec = int(line.split('\t')[3]), int(line.split('\t')[4]), line.split('\t')[5]
            if not treeDic.__contains__(eid):
                treeDic[eid] = {}
            treeDic[eid][indexC] = {'parent': indexP, 'max_degree': max_degree, 'maxL': maxL, 'vec': Vec}
        print('tree no:', len(treeDic))
    return treeDic

################################# load data ###################################
def loadData(dataname, treeDic, fold_x_train, fold_x_test, TDdroprate, BUdroprate):
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
        testdata_list = bigraph_dataset_PHEME(fold_x_test, treeDic,picklefear=picklefear)
    else:
        testdata_list = BiGraphDataset(fold_x_test, treeDic, data_path=data_path)
    print("test no:", len(testdata_list))
    return traindata_list, testdata_list


