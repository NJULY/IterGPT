import json
import os


TASK_TO_ID = {
    "head prediction": 0,
    "tail prediction": 1,
}
ID_TO_TASK = {value: key for key, value in TASK_TO_ID.items()}


def _load_json(path):
    with open(path, "r", encoding="utf-8") as fin:
        return json.load(fin)


def _load_id_list(data_dir, kind):
    map_path = os.path.join(data_dir, f"{kind}_id_map.json")
    if os.path.exists(map_path):
        return _load_json(map_path)

    legacy_path = os.path.join(data_dir, f"{kind}_ids.json")
    if os.path.exists(legacy_path):
        return _load_json(legacy_path)

    compact_path = os.path.join(data_dir, f"{kind}.compact.json")
    if os.path.exists(compact_path):
        payload = _load_json(compact_path)
        if "ids" in payload:
            return payload["ids"]
    return None


def load_entity_text(data_dir):
    compact_path = os.path.join(data_dir, "entity.compact.json")
    if os.path.exists(compact_path):
        payload = _load_json(compact_path)
        entity_ids = payload.get("ids") or _load_id_list(data_dir, "entity")
        return {
            ent: {"name": text[0], "desc": text[1]}
            for ent, text in zip(entity_ids, payload["texts"])
        }
    return _load_json(os.path.join(data_dir, "entity.json"))


def load_relation_text(data_dir):
    compact_path = os.path.join(data_dir, "relation.compact.json")
    if os.path.exists(compact_path):
        payload = _load_json(compact_path)
        relation_ids = payload.get("ids") or _load_id_list(data_dir, "relation")
        return {
            rel: {"name": text}
            for rel, text in zip(relation_ids, payload["names"])
        }
    return _load_json(os.path.join(data_dir, "relation.json"))


def load_triples(data_dir, split):
    compact_path = os.path.join(data_dir, f"{split}.compact.tsv")
    entity_ids = _load_id_list(data_dir, "entity")
    relation_ids = _load_id_list(data_dir, "relation")
    if os.path.exists(compact_path) and entity_ids is not None and relation_ids is not None:
        triples = []
        with open(compact_path, "r", encoding="utf-8") as fin:
            for line in fin:
                h, r, t = line.rstrip("\n").split("\t")
                triples.append((entity_ids[int(h)], relation_ids[int(r)], entity_ids[int(t)]))
        return triples

    triples = []
    with open(os.path.join(data_dir, f"{split}.txt"), "r", encoding="utf-8") as fin:
        for line in fin:
            h, r, t = line.strip().split("\t")
            triples.append((h, r, t))
    return triples


def load_query_rows(data_dir):
    compact_path = os.path.join(data_dir, "query.compact.tsv")
    entity_ids = _load_id_list(data_dir, "entity")
    relation_ids = _load_id_list(data_dir, "relation")
    if os.path.exists(compact_path) and entity_ids is not None and relation_ids is not None:
        rows = []
        with open(compact_path, "r", encoding="utf-8") as fin:
            for line in fin:
                h, r, t, task = line.rstrip("\n").split("\t")
                rows.append((entity_ids[int(h)], relation_ids[int(r)], entity_ids[int(t)], ID_TO_TASK[int(task)]))
        return rows

    rows = []
    with open(os.path.join(data_dir, "query.txt"), "r", encoding="utf-8") as fin:
        for line in fin:
            rows.append(tuple(line.strip().split("\t")))
    return rows


def load_base_model(data_dir, model_name):
    compact_path = os.path.join(data_dir, "base_model", f"{model_name}.compact.json")
    entity_ids = _load_id_list(data_dir, "entity")
    relation_ids = _load_id_list(data_dir, "relation")
    if os.path.exists(compact_path) and entity_ids is not None and relation_ids is not None:
        data = {}
        for h, r, t, task, topk, rank in _load_json(compact_path):
            triple = [entity_ids[h], relation_ids[r], entity_ids[t]]
            key = "\t".join(triple + [ID_TO_TASK[task]])
            data[key] = {
                "triple": triple,
                "task": ID_TO_TASK[task],
                "topk": [entity_ids[ent] for ent in topk],
                "base rank": rank,
            }
        return data

    raw_path = os.path.join(data_dir, "base_model", f"{model_name}.json")
    raw_data = _load_json(raw_path)
    return {
        "\t".join(data_dict["triple"] + [data_dict["task"]]): data_dict
        for data_dict in raw_data
    }
