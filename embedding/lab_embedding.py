import numpy as np
import torch
from bert_embedding import *

# Lab usage
if __name__ == "__main__":
    # sample time series data:
    # each entry is a tuple: (timestamp, [(test_name, test_value), ...])
    data = [
            (-248, [('-basos', 1.1286022590538405), ('-eos', 1.6826683504341247), ('-lymphs', 0.004852290106795553), ('-monos', 0.26289545719770113), ('-polys', -0.8213480804934596), ('ALT (SGPT)', -0.19319455378142533), ('AST (SGOT)', -0.15740436350408996), ('BUN', -0.5149892893503268), ('Hct', 1.4819347595566128), ('Hgb', 1.6883815831234443), ('MCH', 0.5237478387751789), ('MCHC', 0.7994241644888418), ('MCV', -0.050577762284479806), ('MPV', 0.9692198555809464), ('RBC', 1.3510371235582574), ('RDW', -1.1695990879938063), ('WBC x 1000', -0.31283045063638126), ('alkaline phos.', -0.4564236030171427), ('amylase', -0.1260756188786068), ('anion gap', -0.7500168051424104), ('bicarbonate', 1.264937687026683), ('calcium', 1.5809586979457797), ('chloride', 0.4642425320159119), ('creatinine', -0.46012349399992253), ('glucose', -0.5564013760807064), ('lipase', -0.14335523073272916), ('platelets x 1000', 0.6472300255022235), ('potassium', 0.8749304493695654), ('sodium', 1.2647888499927997), ('total bilirubin', -0.3229632701340325)]), 
    (-90, [('Base Excess', -0.48396401537505584), ('Carboxyhemoglobin', 0.05384145681722124), ('HCO3', -0.42504962635442817), ('Methemoglobin', -0.7581220701554726), ('O2 Content', 0.17384482438606189), ('O2 Sat (%)', -0.05322575684175308), ('pH', -0.0016136682173541338), ('paCO2', -0.3398275987912352), ('paO2', -0.5975567325652753)]), 
    (-8, [('Base Excess', -0.32186131532033174), ('Carboxyhemoglobin', -0.03253657596243764), ('FiO2', 0.25776678544535203), ('HCO3', 0.07173248121073801), ('Methemoglobin', -0.7581220701554726), ('O2 Content', 0.1805669280212859), ('O2 Sat (%)', -0.07896717106654727), ('PEEP', -0.030005996998062338), ('Respiratory Rate', 0.03320496237017513), ('TV', -0.03276774148522367), ('pH', -0.007050079263907476), ('paCO2', 0.5572825332529243), ('paO2', -0.5706195681934281)]), 
    (28, [('bedside glucose', 0.23972517512973662)]), 
    (58, [('Base Excess', -0.23344166074502767), ('Carboxyhemoglobin', -0.03253657596243764), ('HCO3', 0.026570471432086445), ('Methemoglobin', -0.7581220701554726), ('O2 Content', 0.18728903165651017), ('O2 Sat (%)', 0.4616025276541233), ('pH', -0.0042887276212137), ('paCO2', 0.2676643016481167), ('paO2', -0.1613818786980549)]), 
    (127, [('CPK', -0.12937706005238897), ('CPK-MB', -0.252267293313654), ('TSH', -0.28335498382671803), ('free T4', 0.10926507321230804), ('magnesium', -0.46096518071309295), ('phosphate', -0.10943485442764046), ('troponin - I', -0.14889978280361713)]), 
    (259, [('Base Excess', -0.38080775170386777), ('Carboxyhemoglobin', -0.7235608381997096), ('FiO2', 0.25776678544535203), ('HCO3', -0.18418557420162016), ('Methemoglobin', -0.4487042294810336), ('O2 Content', -0.2630919119035086), ('O2 Sat (%)', -3.4897045558517283), ('PEEP', -0.030005996998062338), ('Respiratory Rate', 0.03320496237017513), ('TV', -0.03276774148522367), ('pH', -0.003857266427042807), ('paCO2', 0.041620803810218494), ('paO2', -1.005758377277116)]), 
    (472, [('CPK', -0.12897463553139968), ('CPK-MB', -0.2089375821499389), ('troponin - I', -0.09707963713249954)])
    ]
    # for this kind of data we have to remove torch.cat in windows encoder and remove test_values used in the main code
    # data = [
    #         (-248, [('-basos'), ('-eos'), ('-lymphs'), ('-monos'), ('-polys'), ('ALT (SGPT)'), ('AST (SGOT)'), ('BUN'), ('Hct'), ('Hgb'), ('MCH'), ('MCHC'), ('MCV'), ('MPV'), ('RBC'), ('RDW'), ('WBC x 1000'), ('alkaline phos.'), ('amylase'), ('anion gap'), ('bicarbonate'), ('calcium'), ('chloride'), ('creatinine'), ('glucose'), ('lipase'), ('platelets x 1000'), ('potassium'), ('sodium'), ('total bilirubin')]), 
    # (-90, [('Base Excess'), ('Carboxyhemoglobin'), ('HCO3'), ('Methemoglobin'), ('O2 Content'), ('O2 Sat (%)'), ('pH'), ('paCO2'), ('paO2')]), 
    # (-8, [('Base Excess'), ('Carboxyhemoglobin'), ('FiO2'), ('HCO3'), ('Methemoglobin'), ('O2 Content'), ('O2 Sat (%)'), ('PEEP'), ('Respiratory Rate'), ('TV'), ('pH'), ('paCO2'), ('paO2')]), 
    # (28, [('bedside glucose')]), 
    # (58, [('Base Excess'), ('Carboxyhemoglobin'), ('HCO3'), ('Methemoglobin'), ('O2 Content'), ('O2 Sat (%)'), ('pH'), ('paCO2'), ('paO2')]), 
    # (127, [('CPK'), ('CPK-MB'), ('TSH'), ('free T4'), ('magnesium'), ('phosphate'), ('troponin - I')]), 
    # (259, [('Base Excess'), ('Carboxyhemoglobin'), ('FiO2'), ('HCO3'), ('Methemoglobin'), ('O2 Content'), ('O2 Sat (%)'), ('PEEP'), ('Respiratory Rate'), ('TV'), ('pH'), ('paCO2'), ('paO2')]), 
    # (472, [('CPK'), ('CPK-MB'), ('troponin - I')])
    # ]
    window_size = 60  # group data points that occur within the same hour
    
    # segment the data into windows
    windows = segment_time_series(data, window_size)
    print("Segmented Windows:")
    for i, window in enumerate(windows):
        print(f"Window {i}: {window}")
    
    # build a vocabulary for test names
    # we should precompute this from your full dataset
    test_names_set = set()
    for _, tests in data:
        for test_name, _ in tests:
            test_names_set.add(test_name)
    test_names_list = sorted(list(test_names_set))
    name2idx = {name: idx for idx, name in enumerate(test_names_list)}
    vocab_size = len(name2idx)
    
    # prepare the input for the model:
    #   for each window, flatten the tests into two tensors: one for test names, one for test values
    windows_batch = []
    for window in windows:
        names = []
        values = []
        for _, tests in window:
            for test_name, test_value in tests:
                names.append(name2idx[test_name])
                values.append(test_value)
        test_names_tensor = torch.tensor(names, dtype=torch.long)
        test_values_tensor = torch.tensor(values, dtype=torch.float)
        windows_batch.append((test_names_tensor, test_values_tensor))
    
    # define model hyperparameters
    name_embed_dim = 8
    value_dim = 4
    hidden_dim = 16
    num_layers = 2
    num_heads = 2
    max_seq_len = len(windows_batch) + 1  # ensure this is large enough for your sequences
    
    # instantiate the model
    model = TimeSeriesBERTCLS(vocab_size, name_embed_dim, value_dim, hidden_dim, 
                           num_layers, num_heads, max_seq_len)
    
    # perform a forward pass.
    output = model(windows_batch)  # expected shape: (seq_len, hidden_dim)
    # TODO: store in a file
    print("\nTransformer Output:")
    print(output)

