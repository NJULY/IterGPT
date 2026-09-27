# IterGPT

## requirements
python==3.8.19
numpy==1.24.3
requests==2.32.3
levenshtein==0.25.1

we only use the basic functions of the above packages, so the version requirements of these packages may be not strict

## run
before running, remember to put your api key in chatgpt_api.key
### inference with two retrievers
`python {fb_itergpt.py|wn_itergpt.py} --structure_model TransE --text_model SimKGC --gpt_model gpt-4o-mini`

### inference with one retriever
`python {fb_itergpt_one.py|wn_itergpt_one.py} --base_model SimKGC --gpt_model gpt-4o-mini`

## compact data
The uploaded datasets use compact indexed files. Each dataset directory contains
`entity_id_map.json` and `relation_id_map.json`; other compact files store entity
and relation indexes. The code maps those indexes back to the original ids in
memory when loading.
