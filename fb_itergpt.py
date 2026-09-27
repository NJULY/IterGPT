import os
import json
import random
import numpy as np
from tqdm import tqdm
from argparse import ArgumentParser
from typing import Dict
from Levenshtein import distance

from knowledge_graph import KnowledgeGraph
from chatgpt_api import chatgpt
from compact_data import load_base_model, load_query_rows
from utils import compute_metrics


def load_data():
    structure_data = load_base_model(args.data_dir, args.structure_model)
    text_data = load_base_model(args.data_dir, args.text_model)

    data = []
    for h, r, t, task in load_query_rows(args.data_dir):
        key = "\t".join([h, r, t, task])
        data.append({
            "triple": [h, r, t],
            "task": task,
            "structure topk": structure_data[key]["topk"][: args.topk],
            "text topk": text_data[key]["topk"][: args.topk],
            "structure rank": structure_data[key]["base rank"],
            "text rank": text_data[key]["base rank"]
        })
    return data


def score(prompts):
    responses = []
    with open(output_path, "r", encoding="utf-8") as fin:
        for line in fin.readlines():
            answer_entity, is_correct = line.strip().split("\t")
            if is_correct == "true":
                responses.append(answer_entity)
            else:
                responses.append("None")
    
    assert len(prompts) == len(responses)

    ranks = []
    for i in range(len(prompts)):        
        answer_entity = responses[i]
        structure_topk = prompts[i].structure_topk
        text_topk = prompts[i].text_topk
        structure_rank = prompts[i].structure_rank
        text_rank = prompts[i].text_rank
        
        if answer_entity == "None":
            topk_names = None
        else:
            if answer_entity in structure_topk and answer_entity in text_topk:
                idx1 = structure_topk.index(answer_entity)
                idx2 = text_topk.index(answer_entity)
                if idx1 < idx2:
                    topk_names = structure_topk
                    base_rank = structure_rank
                else:
                    topk_names = text_topk
                    base_rank = text_rank
            elif answer_entity in structure_topk and answer_entity not in text_topk:
                topk_names = structure_topk
                base_rank = structure_rank
            elif answer_entity not in structure_topk and answer_entity in text_topk:
                topk_names = text_topk
                base_rank = text_rank
            else:
                topk_names = None

        if topk_names:
            response_rank = topk_names.index(answer_entity)+1
            if response_rank == base_rank:
                rank = 1
            elif response_rank < base_rank:
                rank = base_rank
            else:
                rank = base_rank + 1
        else:
            rank = text_rank
        ranks.append(rank)

    print(compute_metrics(ranks))


class Prompt:
    def __init__(self, data_dict: Dict) -> None:
        self.head, self.relation, self.tail = data_dict["triple"]
        self.task = data_dict["task"]
        self.structure_topk = data_dict["structure topk"]
        self.text_topk = data_dict["text topk"]
        self.structure_rank = data_dict["structure rank"]
        self.text_rank = data_dict["text rank"]
        self.structure_topk_names = [ent2name[ent] for ent in self.structure_topk]
        self.text_topk_names = [ent2name[ent] for ent in self.text_topk]

        if self.task == "head prediction":
            self.entity = self.tail
            self.query_str = f"([mask], {self.relation}, {ent2name[self.tail]})"
        elif self.task == "tail prediction":
            self.entity = self.head
            self.query_str = f"({ent2name[self.head]}, {self.relation}, [mask])"
        else:
            assert 0, f"unknown task: {self.task}"

        self.entity_triples, self.relation_triples = kg.sample_triples(self.entity, self.relation, self.task, args.support_triple_num)
        self.entity_demos = "; ".join([f"({ent2name[h]}, {rel2name[r]}, {ent2name[t]})" for h, r, t in self.entity_triples])
        self.relation_demos = "; ".join([f"({ent2name[h]}, {rel2name[r]}, {ent2name[t]})" for h, r, t in self.relation_triples])

        self.structure_candidates = "[" + "; ".join([f"{ent2name[ent]}" for idx, ent in enumerate(self.structure_topk)]) + "]"
        self.text_candidates = "[" + "; ".join([f"{ent2name[ent]}" for idx, ent in enumerate(self.text_topk)]) + "]"

    def init_prompt(self):
        prompt = """Let us do the knowledge graph completion task that predicts the missing entity in an incomplete triple. The entity to be predicted is denoted by "[mask]". You are provided with the relevant descriptions, relational facts and two candiate entity ranking list."""
        # 1. the query
        prompt += f"\n\nThe incomplete triple:\n{self.query_str}."
        # 2. background knowledge
        # entity knowledge
        prompt += f"\n\nDescription of entity {ent2name[self.entity]}:\n{ent2desc[self.entity]}"
        # relation knowledge
        if args.support_triple_num > 0 and len(self.relation_triples) > 0:
            prompt += f"\n\nRelational facts of relation {self.relation}:\n{self.relation_demos}"
        # 3. context from structural retriever
        prompt += f"\n\nA graph structure-based ranking list of top-{args.topk} candidates for the unknown entity [mask]:\n{self.structure_candidates}"
        # 4. context from textual retriever
        prompt += f"\n\nA text-based ranking list of top-{args.topk} candidates for the unknown entity [mask]:\n{self.text_candidates}"
        # 5. task requirements
        prompt += f"\n\nBoth ranking lists are ranked in descending order of confidence. Taking both candidate lists into consideration at the same time, your task is to select the most plausible candidate from the provided ranking lists, denote the picked candidate as \"C\"."
        prompt += f"""\nYou can return the first candidate in the graph structure-based ranking list, if (1) you do not know anything about the input incomplete triple, or (2) the candidates are difficult for you to distinguish."""
        prompt += f"""\n\nUsing the format "The answer is: <candidate>! The reason is: <reason>." for output. "<candidate>" is the selected candidate "C" mentioned above. "<reason>" is the explanation why you choose this candidate as your answer. Do not output anything except the given format."""

        return prompt

    @classmethod
    def parse_init_response(cls, response):
        response = response[len("The answer is: "): ]
        answer, reason = response.split("! The reason is: ")
        return answer, reason
    
    def find_entity_by_name(self, answer_name):
        if answer_name in self.structure_topk_names:
            idx = self.structure_topk_names.index(answer_name)
            answer_entity = self.structure_topk[idx]
        elif answer_name in self.text_topk_names:
            idx = self.text_topk_names.index(answer_name)
            answer_entity = self.text_topk[idx]
        else:
            entity_list = self.structure_topk + self.text_topk
            name_list = self.structure_topk_names + self.text_topk_names
            
            distances = np.array([distance(answer_name, name) for name in name_list])
            sorted_idxs = np.argsort(distances)
            idx = sorted_idxs[0]
            answer_entity = entity_list[idx]
        return answer_entity

    def check_prompt(self, answer_entity, reason):
        answer_name = ent2name[answer_entity]

        candidate_triple = self.query_str.replace("[mask]", answer_name)
        prompt = f"""Based on your answer, "{answer_name}" is the most plausible candidate for the missing entity in the incomplete triple."""
        prompt += f"""\nThe reason why "{answer_name}" is the most plausible candidate:\n{reason}"""
        
        prompt += f"""\nThe description of the answer {ent2name[answer_entity]}:\n{ent2desc[answer_entity]}"""
        prompt += f"""\n\nBased on the reason and the descriptions of "{ent2name[self.entity]}" and "{answer_name}" mentioned above, answer the question:\n"Whether the triple {candidate_triple} is correct?"\n"""
        
        prompt += f"""Using the format "The triple is correct!" or "The triple is incorrect!" for output. Do not output anything except the given format."""
        return prompt

    def parse_check_response(self, response):
        if response == "The triple is correct!":
            return True
        elif response == "The triple is incorrect!":
            return False
        else:
            assert 0, response

    def retry_prompt(self, answer_entity):
        answer_name = ent2name[answer_entity]
        prompt = f"""Based on your decision, {answer_name} is not the desired candidate."""
        prompt += f"""\nSo let us select another candidate from the two provided ranking list as the new answer, denote the picked candidate as "C"."""
        prompt += f"""\nUsing the format "The answer is: <candidate>! The reason is: <reason>." for output. "<candidate>" is the selected candidate "C" mentioned above. "<reason>" is the explanation why you choose this candidate as your answer. Do not output anything except the given format."""
        return prompt

    def loop(self):
        context = {"answer": None, "correct": False}
        message = [{"role": "user", "content": self.init_prompt()}]
        for _ in range(args.retry_num):
            response = chatgpt(message, args.gpt_model)
            message.append({"role": "assistant", "content": response})
            answer_name, reason = Prompt.parse_init_response(response)
            answer_entity = self.find_entity_by_name(answer_name)
            
            message.append({"role": "user", "content": self.check_prompt(answer_entity, reason)})
            response = chatgpt(message, args.gpt_model)
            message.append({"role": "assistant", "content": response})
            is_correct = self.parse_check_response(response)

            context["answer"] = answer_entity
            if is_correct:
                context["correct"] = True
                break
            
            context["correct"] = False
            message.append({"role": "user", "content": self.retry_prompt(answer_entity)})
        
        if context["correct"]:
            output = context["answer"] + "\t" + "true"
        else:
            output = context["answer"] + "\t" + "false"

        if message[-1]["role"] == "user":
            message = message[: -1]

        return message, output


def get_args():
    parser = ArgumentParser()
    parser.add_argument('--data_dir', default="dataset/FB15K237", type=str)
    parser.add_argument('--output_dir', default="output/FB15K237/itergpt", type=str)
    
    parser.add_argument('--structure_model', default="TransE", type=str)
    parser.add_argument('--text_model', default="SimKGC", type=str)
    parser.add_argument('--gpt_model', default="gpt-4o-mini", type=str)

    parser.add_argument('--retry_num', default=4, type=int)
    parser.add_argument('--support_triple_num', default=10, type=int)
    parser.add_argument('--topk', default=20, type=int)
    parser.add_argument('--seed', default=42, type=int)
    args = parser.parse_args()
    return args


args = get_args()
random.seed(args.seed)

message_path = os.path.join(args.output_dir, "message.jsonl")
output_path = os.path.join(args.output_dir, "output.txt")
if not os.path.exists(args.output_dir):
    os.makedirs(args.output_dir)
    json.dump(vars(args), open(os.path.join(args.output_dir, "args.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=4)
    with open(message_path, "w", encoding="utf-8") as fout:
        print(f"create message_path file: {message_path}")
    with open(output_path, "w", encoding="utf-8") as fout:
        print(f"create output file: {output_path}")

kg = KnowledgeGraph(args)
ent2name, ent2desc, rel2name = kg.ent2name, kg.ent2desc, kg.rel2name

data = load_data()
prompts = [Prompt(data_dict) for data_dict in data]


offset = 0
with open(output_path, "r", encoding="utf-8") as fin:
    offset += len(fin.readlines())

print("call API:", args.gpt_model)
for i in tqdm(range(offset, len(prompts))):
    message, output = prompts[i].loop()
    with open(message_path, "a", encoding="utf-8") as fout:
        fout.write(json.dumps(message, ensure_ascii=False) + "\n")
    with open(output_path, "a", encoding="utf-8") as fout:
        fout.write(output + "\n")

score(prompts)
