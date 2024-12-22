import os
import json
import sys
import pickle
import numpy as np


def rule_convert_np(data_dir, num_nodes):
    file_names = os.listdir(data_dir)
    snap = dict()
    min_double = sys.float_info.min
    for file_name in file_names:
        snap[int(file_name.split("_")[1])] = json.load(open(os.path.join(data_dir, file_name), 'r', encoding='utf-8'))
    snap_inver = dict()
    snap_zheng = dict()
    for key in snap.keys():
        # if int(key) != 12 and int(key) != 13:
        #     continue
        if len(snap[key]) % 2 != 0:
            print(key)
        idx = int(len(snap[key]) / 2)
        value = list(snap[key].values())
        snap_zheng[key] = value[:idx]
        snap_inver[key] = value[idx:]
    snap_zheng_np = dict()
    snap_inver_np = dict()
    for key in snap_zheng.keys():
        # 正序
        snap_zheng_np[key] = np.zeros((len(snap_zheng[key]), num_nodes))
        snap_zheng_np[key][:, :] = min_double
        for idx, i in enumerate(snap_zheng[key]):
            query_key = [int(idx) for idx in i.keys()]
            query_value = list(i.values())
            snap_zheng_np[key][idx, query_key] = query_value
        # inverse
        snap_inver_np[key] = np.zeros((len(snap_inver[key]), num_nodes))
        snap_inver_np[key][:, :] = min_double
        for idx, i in enumerate(snap_inver[key]):
            query_key = [int(idx) for idx in i.keys()]
            query_value = list(i.values())
            snap_inver_np[key][idx, query_key] = query_value
        s = np.concatenate([snap_zheng_np[key], snap_inver_np[key]], axis=1)
        out_dir = "D:/时序/Temporal_Memory/data/ICEWS18/test"
        file = out_dir + "/" + str(key) + ".pkl"
        with open(file, 'wb') as file:
            pickle.dump(s, file)
    # snap_np = dict()
    # for key in snap_zheng_np.keys():
    #     snap_np[key] = np.concatenate([snap_zheng_np[key], snap_inver_np[key]], axis=1)
    # return snap_np

data_dir = "D:/时序/rule_candidate/icews18_new/test"
rule_convert_np(data_dir,23033)
#
# out_dir = "D:/时序/Temporal_Memory/data/ICEWS18/test"
# for key in tet.keys():
#     file = out_dir+"/"+str(key)+".pkl"
#     with open(file,'wb') as file:
#         pickle.dump(tet[key],file)









#数据检查
# candidate_data_dir = "D:/时序/rule_candidate/icews15_new/snap"
# candidate_file_names = os.listdir(candidate_data_dir)
# snap_data_dir = "D:/时序/TLogic-main/data/icews0515_new/snap"
# snap_file_names = os.listdir(snap_data_dir)
# can_dict={}
# for file_name in candidate_file_names:
#     idx = file_name.split("_")[1]
#     can_dict[int(idx)] = file_name
# for file_name in snap_file_names:
#     idx = file_name.split(".")[0]
#     snap_content = None
#     if int(idx) in can_dict.keys():
#         json_content = json.load(open(os.path.join(candidate_data_dir,can_dict[int(idx)])))
#         with open(os.path.join(snap_data_dir,file_name),'r') as f:
#             snap_content = f.readlines()
#         if len(snap_content) != (len(json_content)/2):
#             print("匹配错误", file_name, can_dict[int(idx)], len(snap_content), (len(json_content)/2))
#     else:
#         print("少数了",idx)