import pickle
import datetime
import numpy as np
from scipy.sparse import csr_matrix, coo_matrix, dok_matrix
from params import args
import scipy.sparse as sp
import dgl
from Utils.TimeLogger import log
import torch as t
import torch
import torch.utils.data as data
import torch.utils.data as dataloader
#device = "cuda" if t.cuda.is_available() else "cpu"
device = t.device('cuda:3' if t.cuda.is_available() else 'cpu')
class DataHandler:
    def __init__(self):
        
        if args.data == 'ijcai_15':
            self.predir = './data/ijcai_15/'
            self.behaviors = ['click', 'fav', 'cart', 'buy']
            self.beh_meta_path = ['buy', 'click_buy', 'click_fav_buy', 'click_fav_cart_buy']

        elif args.data  == 'tmall':
            self.predir = './data/tmall/'
            self.behaviors = ['pv', 'fav', 'cart', 'buy']
            self.beh_meta_path = ['buy', 'pv_buy', 'pv_fav_buy', 'pv_fav_cart_buy']
        elif args.data == 'retail_rocket':
            self.predir = './data/retail_rocket/'
            self.behaviors = ['view', 'cart', 'buy']
            self.beh_meta_path = ['buy', 'view_buy', 'view_cart_buy']

        self.train_file = self.predir + 'train_mat_'
        self.val_file = self.predir + 'test_mat.pkl'
        self.test_file = self.predir + 'test_mat.pkl'

    def _load_data(self):
        test_mode = 'lightgc'
        self.t_max = -1
        self.t_min = 0x7FFFFFFF
        self.time_number = -1

        self.user_num = -1
        self.item_num = -1
        self.behavior_mats = {}
        self.behaviors_data = {}


        all_behaviors = ['buy', 'fav', 'click', 'cart', 'view', 'pv']




        for behavior_name in self.behaviors:

            file_path = self.train_file + behavior_name + '.pkl'
            try:
                with open(file_path, 'rb') as fs:
                    data = pickle.load(fs)

                    if behavior_name == 'buy':
                        self.train_mat = data
                        self.trainLabel = 1 * (self.train_mat != 0)
                        self.labelP = np.squeeze(np.array(np.sum(self.trainLabel, axis=0)))
                        continue


                    self.behaviors_data[behavior_name] = 1 * (data != 0)


                    if data.get_shape()[0] > self.user_num:
                        self.user_num = data.get_shape()[0]
                    if data.get_shape()[1] > self.item_num:
                        self.item_num = data.get_shape()[1]


                    if hasattr(data, 'data'):
                        if data.data.max() > self.t_max:
                            self.t_max = data.data.max()
                        if data.data.min() < self.t_min:
                            self.t_min = data.data.min()

            except FileNotFoundError:
                print(f"Warning: {behavior_name} data file not found, skipping...")
                continue


        self.test_mat = pickle.load(open(self.test_file, 'rb'))

        self.userNum = self.user_num
        self.itemNum = self.item_num


        self._data2mat()

        if test_mode == 'muti':
            self.target_adj = self._dataTargetmat()
        elif test_mode == 'lightgcn':
            self.target_adj = self._make_bitorch_adj(self.train_mat)
        else:

            self.target_adj = self.makeBiAdj(self.train_mat, self.userNum, self.itemNum).to(device)


            for behavior_name in self.behaviors_data:
                mat = self.behaviors_data[behavior_name]
                self.behavior_mats[behavior_name] = self.makeBiAdj(mat, self.userNum, self.itemNum).to(device)


        self.beh_degree_list = {}
        for behavior_name in self.behaviors_data:
            degree_tensor = torch.tensor((self.behaviors_data[behavior_name] != 0).sum(axis=-1)).cuda()
            self.beh_degree_list[behavior_name] = degree_tensor




    def _data2mat(self):
        time = datetime.datetime.now()
        print("Start building: ", time)


        for behavior in self.behaviors_data:
            self.behaviors_data[behavior] = 1 * (self.behaviors_data[behavior] != 0)


        self.behavior_mats = {}
        for behavior in self.behaviors_data:
            mat = self.behaviors_data[behavior]
            self.behavior_mats[behavior] = self.makeBiAdj(mat, self.userNum, self.itemNum).to(device)

        time = datetime.datetime.now()
        print("End building: ", time)


    def _dataTargetmat(self):
         target_adj = 1*(self.train_mat!=0)
         target_adj = self._get_use(target_adj)
         return target_adj


    def _get_use(self, behaviors_data):
        behavior_mats = {}
        behaviors_data = (behaviors_data != 0) * 1

        behavior_mats['A'] = self._matrix_to_tensor(self._normalize_adj(behaviors_data))
        behavior_mats['AT'] = self._matrix_to_tensor(self._normalize_adj(behaviors_data.T))
        behavior_mats['A_ori'] = None
        return behavior_mats

    def _normalize_adj(self, adj):

        adj = sp.coo_matrix(adj)
        rowsum = np.array(adj.sum(1))
        rowsum_diag = sp.diags(np.power(rowsum+1e-8, -0.5).flatten())
        colsum = np.array(adj.sum(0))
        colsum_diag = sp.diags(np.power(colsum+1e-8, -0.5).flatten())
        return rowsum_diag*adj*colsum_diag


    def _matrix_to_tensor(self, cur_matrix):
        if type(cur_matrix) != sp.coo_matrix:
            cur_matrix = cur_matrix.tocoo()
        indices = torch.from_numpy(np.vstack((cur_matrix.row, cur_matrix.col)).astype(np.int64))
        values = torch.from_numpy(cur_matrix.data)
        shape = torch.Size(cur_matrix.shape)
        return torch.sparse.FloatTensor(indices, values, shape).to(torch.float32).cuda()

    def makeBiAdj(self, mat,n_user,n_item):


        a = sp.csr_matrix((n_user, n_user))
        b = sp.csr_matrix((n_item, n_item))

        mat = sp.vstack([sp.hstack([a, mat]), sp.hstack([mat.transpose(), b])])
        mat = (mat != 0) * 1.0
        mat = mat.tocoo()

        edge_src,edge_dst = mat.nonzero()
        ui_graph = dgl.graph(data=(edge_src, edge_dst),
                            idtype=torch.int32,
                             num_nodes=mat.shape[0]
                             )

        return ui_graph

    def load_data(self):
        
        self._load_data()
        args.user_num, args.item_num = self.train_mat.shape

        test_data = AllRankTestData(self.test_mat, self.train_mat)    #
        self.test_dataloader = dataloader.DataLoader(test_data, batch_size=args.tstBat, shuffle=False, num_workers=0)

        train_dataset = PairwiseTrnData(self.trainLabel.tocoo())
        self.train_dataloader = dataloader.DataLoader(train_dataset, batch_size=args.batch, shuffle=True, num_workers=4, pin_memory=True)


    def _normalize_biadj(self, mat):

        degree = np.array(mat.sum(axis=-1)) + 1e-10

        d_inv_sqrt = np.reshape(np.power(degree, -0.5), [-1])
        d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.0


        d_inv_sqrt_mat = sp.diags(d_inv_sqrt)

        return mat.dot(d_inv_sqrt_mat).transpose().dot(d_inv_sqrt_mat).tocoo()


    def _make_bitorch_adj(self, mat):

        if type(mat) != sp.csr_matrix:
            mat = mat.tocsr()

        a = csr_matrix((self.userNum,  self.userNum))
        b = csr_matrix((self.itemNum, self.itemNum))

        mat = sp.vstack([sp.hstack([a, mat]), sp.hstack([mat.transpose(), b])])

        mat = (mat != 0) * 1.0

        mat = self._normalize_biadj(mat)


        idxs = t.from_numpy(np.vstack([mat.row, mat.col]).astype(np.int64))
        vals = t.from_numpy(mat.data.astype(np.float32))
        shape = t.Size(mat.shape)
        return t.sparse.FloatTensor(idxs, vals, shape).cuda()
        



class TrnData(data.Dataset):
    def __init__(self, coomat):
        coomat = coomat.tocoo()
        self.rows = coomat.row
        self.cols = coomat.col
        self.dokmat = coomat.todok()
        self.negs = np.zeros(len(self.rows)).astype(np.int32)

    def negSampling(self):
        for i in range(len(self.rows)):
            u = self.rows[i]
            while True:
                iNeg = np.random.randint(args.item_num)
                if (u, iNeg) not in self.dokmat:
                    break
            self.negs[i] = iNeg

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        return self.rows[idx], self.cols[idx], self.negs[idx]

class TstData(data.Dataset):
    def __init__(self, coomat, trnMat):
        coomat = coomat.tocoo()
        self.csrmat = (trnMat.tocsr() != 0) * 1.0

        tstLocs = [None] * coomat.shape[0]
        tstUsrs = set()
        for i in range(len(coomat.data)):
            row = coomat.row[i]
            col = coomat.col[i]
            if tstLocs[row] is None:
                tstLocs[row] = list()
            tstLocs[row].append(col)
            tstUsrs.add(row)
        tstUsrs = np.array(list(tstUsrs))
        self.tstUsrs = tstUsrs
        self.user_pos_lists = tstLocs

    def __len__(self):
        return len(self.tstUsrs)

    def __getitem__(self, idx):
        return self.tstUsrs[idx], np.reshape(self.csrmat[self.tstUsrs[idx]].toarray(), [-1])


class AllRankTestData(data.Dataset):
    def __init__(self, coomat, trn_mat):
        self.csrmat = (trn_mat.tocsr() != 0) * 1.0

        user_pos_lists = [list() for i in range(coomat.shape[0])]

        test_users = set()
        for i in range(len(coomat.data)):
            row = coomat.row[i]
            col = coomat.col[i]
            user_pos_lists[row].append(col)
            test_users.add(row)
        self.test_users = np.array(list(test_users))
        self.user_pos_lists = user_pos_lists

    def __len__(self):
        return len(self.test_users)

    def __getitem__(self, idx):
        pck_user = self.test_users[idx]
        pck_mask = self.csrmat[pck_user].toarray()
        pck_mask = np.reshape(pck_mask, [-1])
        return pck_user, pck_mask


class PairwiseTrnData(data.Dataset):
	def __init__(self, coomat):
		self.rows = coomat.row
		self.cols = coomat.col
		self.dokmat = coomat.todok()
		self.negs = np.zeros(len(self.rows)).astype(np.int32)
	
	def negSampling(self):
		for i in range(len(self.rows)):
			u = self.rows[i]
			while True:
				iNeg = np.random.randint(args.item_num)
				if (u, iNeg) not in self.dokmat:
					break
			self.negs[i] = iNeg
	
	def __len__(self):
		return len(self.rows)

	def __getitem__(self, idx):
		return self.rows[idx], self.cols[idx], self.negs[idx]
     

class DiffusionData(data.Dataset):
    def __init__(self,y_data):
        self.y_data = y_data
        self.x_data = np.arange(0,len(y_data))
    def __len__(self):
        return len(self.x_data)

    def __getitem__(self, idx):
        return self.x_data[idx],self.y_data[idx]



