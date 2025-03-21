# End-to-end\_graph\_missing\_modalities
Lab project on end-to-end graph missing modalities by Timothée Strouk and Xuecheng Wu

# Usage
1. Edit the path in `src/utils.py` to your local path.
2. Obtain the eICU dataset and place it under `{raw_data_path}`.
3. Run the following notebooks under `src/preprocess` in the specified order to prepare the data:
   1. eICU:
      1. Run `src/preprocess/parse_eicu_remote.ipynb`
      2. Run `cd ../../`
      3. Run `preprocess_eicu.py`
      4. Run `cd src/preprocess/`
      5. Run `build_vocab_eicu.py`
      6. Run `data_split_eicu.py`
   2. Run `get_code_embeddings.py`
4. Check that icu\_stay\_dict.pkl, big\_icu\_stay\_dict.pkl, small\_icu\_stay\_dict.pkl, embeddings.txt and the lab\_embedding\*.pkl are in src/processed\_data/eicu
5. Create the output directory
6. Run impute.py: 
    + The data is saved in the `{output\_path}/\<split\>` folder. {output\_path} is defined in `src/utils.py`. \<split\> being train, test and validation.
    + Each file's name is the observation id.
    + Each file contains a list of tensors. The last element is the label.
    + Each file can be loaded with pickle.load().

> We increased the k of the feat-Prop to 3, for ApacheApsvar because only 4% of elements were complete
