# End-to-end_graph_missing_modalities
Lab project on end-ot-end graph missing modalitiesby Timothée Strouk and Xuecheng Wu

# Usage
1. Edit the path in `src/utils.py` to your local path.
2. Obtain the eICU dataset and place it under `{raw_data_path}`.
3. Run the following notebooks under `src/preprocess` in the specified order to prepare the data:
   1. eICU:
      1. Run `parse_eicu_remote.ipynb`
      2. Run `preprocess_eicu.py`
      3. Run `build_vocab_eicu.py`
      4. Run `data_split_eicu.py`
   2. Run `get_code_embeddings.py`
4. Run preprocess\_eicu.py
5. Check that icu\_stay\_dict.pkl, big\_icu\_stay\_dict.pkl, small\_icu\_stay\_dict.pkl and the lab\embedding\*.pkl are in src/processed\_data/eicu.
6. Run impute\_apacheapsvar.py The data before and after imputation will be printed on the console. 
