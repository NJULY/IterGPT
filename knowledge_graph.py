import random
from collections import defaultdict

from compact_data import load_entity_text, load_relation_text, load_triples


class KnowledgeGraph:
    def __init__(self, args) -> None:
        self.args = args

        self.entity2text = load_entity_text(args.data_dir)
        self.relation2text = load_relation_text(args.data_dir)
        self.ent2name = {ent: self.entity2text[ent]["name"] for ent in self.entity2text}
        if args.data_dir.endswith("WN18RR"):
            self.clean_wordnet_postag()
        
        self.ent2desc = {ent: self.entity2text[ent]["desc"] for ent in self.entity2text}
        self.rel2name = {rel: self.relation2text[rel]["name"] for rel in self.relation2text}

        self.train_triples = load_triples(args.data_dir, 'train')
        self.valid_triples = load_triples(args.data_dir, 'valid')
        self.test_triples = load_triples(args.data_dir, 'test')
        
        triples = self.train_triples + self.valid_triples + self.test_triples
        self.entity_list = sorted(list(set([h for h, _, _ in triples] + [t for _, _, t in triples])))
        self.relation_list = sorted(list(set([r for _, r, _ in triples])))
        print(f'entity num: {len(self.entity_list)}; relation num: {len(self.relation_list)}')
        
        self.entity2triples = defaultdict(list)
        self.relation2triples = defaultdict(list)
        for h, r, t in self.train_triples:
            self.entity2triples[h].append((h, r, t))
            self.entity2triples[t].append((h, r, t))
            self.relation2triples[r].append((h, r, t))

    def clean_wordnet_postag(self):
        for ent, name in self.ent2name.items():
            name_tokens = name.split("_")
            pos_tag = name_tokens[-2]
            if pos_tag == "NN":
                pos_tag = "noun"
            elif pos_tag == "VB":
                pos_tag = "verb"
            elif pos_tag == "JJ":
                pos_tag = "adjective"
            elif pos_tag == "RB":
                pos_tag = "adverb"
            else:
                assert 0, pos_tag

            name = " ".join(name_tokens[:-2])+f" ({pos_tag}, {name_tokens[-1]})"
            self.ent2name[ent] = name
    
    def sample_triples(self, entity, relation, task, num):
        entity_triples = self.entity2triples[entity]
        if len(entity_triples) > num:
            entity_triples = random.sample(entity_triples, num)
        
        relation_triples = self.relation2triples[relation]
        if len(relation_triples) > num:
            relation_triples = random.sample(relation_triples, num)
        
        return entity_triples, relation_triples
