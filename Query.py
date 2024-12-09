import json

import numpy as np
import pandas as pd
import editdistance
import csv
from sklearn.metrics import pairwise_distances

crowd_data = pd.read_csv('Dataset/crowd_data/crowd_data.tsv', sep='\t')
image_file_path = 'Dataset/images.json'
# Multimedia dataset
with open(image_file_path, 'r') as f:
    images_data = json.load(f)

class QueryExecutor:
    def __init__(self, graph, all_label_dict, entity_label_dict, relation_label_dict,label_entity_dict,label_relation_dict):
        self.graph = graph

        self.all_label_dict = all_label_dict
        self.entity_label_dict = entity_label_dict
        self.relation_label_dict = relation_label_dict
        self.label_entity_dict = label_entity_dict
        self.label_relation_dict = label_relation_dict

        # Build embedding dictionary to process embedding questions.
        entity_embedding = np.load("Dataset/entity_embeds.npy")
        self.entity_embedding = entity_embedding
        relation_embedding = np.load("Dataset/relation_embeds.npy")
        self.relation_embedding = relation_embedding
        idx_entity_dict = {}
        idx_relation_dict = {}
        entity_embedding_dict = {}
        relation_embedding_dict = {}
        with open("Dataset/entity_ids.del", "r", encoding="UTF-8") as f:
            for row in csv.reader(f, delimiter="\t"):
                idx_entity_dict[int(row[0])] = row[1]
                entity_embedding_dict[row[1]] = entity_embedding[int(row[0])]
        with open("Dataset/relation_ids.del", "r", encoding="UTF-8") as f:
            for row in csv.reader(f, delimiter="\t"):
                idx_relation_dict[int(row[0])] = row[1]
                relation_embedding_dict[row[1]] = relation_embedding[int(row[0])]
        self.idx_entity_dict = idx_entity_dict
        self.idx_relation_dict = idx_relation_dict
        self.entity_embedding_dict = entity_embedding_dict
        self.relation_embedding_dict = relation_embedding_dict

        # Load crowdsource dataset
        self.crowdsource_data = crowd_data
        self.crowdsource_subject_list = self.crowdsource_data['Input1ID'].unique()
        self.crowdsource_predicate_list = self.crowdsource_data['Input2ID'].unique()
        self.images_data = images_data

    def querySPARQL(self, sparql):
        return str([str(s) for s, in self.graph.query(sparql)])


    def queryFactualQuestions(self, entity, predicate):
        # Get entity and predicate from previous step
        input_entity = entity
        input_predicate = predicate
        print(f"User's input entity : {input_entity}, relation : {input_predicate}")
        entity_uri = None
        relation_uri = None

        entity_uri, relation_uri = self.__get_URL_By_label(input_entity, input_predicate)

        # If relation belong to these word, then just return the object from triple
        if input_predicate in ["cost", "box office", "IMDb ID", "publication date", "node description", "image"]:
            query = f"""
                        PREFIX ddis: <http://ddis.ch/atai/>   
                        PREFIX wd: <http://www.wikidata.org/entity/>   
                        PREFIX wdt: <http://www.wikidata.org/prop/direct/>   
                        PREFIX schema: <http://schema.org/>
                        PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>  

                        SELECT ?object WHERE {{
                            <{entity_uri}> <{relation_uri}> ?object .
                        }}
                        """
        else:
            # Other relations should find label of object
            query = f"""
                        PREFIX ddis: <http://ddis.ch/atai/>   
                        PREFIX wd: <http://www.wikidata.org/entity/>   
                        PREFIX wdt: <http://www.wikidata.org/prop/direct/>   
                        PREFIX schema: <http://schema.org/>
                        PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>  

                        SELECT ?label WHERE {{
                            ?object rdfs:label ?label .
                            <{entity_uri}> <{relation_uri}> ?object .
                            FILTER(LANG(?label) = "en")
                        }}
                        """
        print("SPARQL STATEMENT : "+ query)
        return [str(s) for s, in self.graph.query(query)]

    def queryEmbeddingQuestions(self, entity, predicate):
        # Get entity and predicate from previous step
        input_entity = entity
        input_predicate = predicate
        print(f"User's input entity : {input_entity}, relation : {input_predicate}")
        entity_uri = None
        relation_uri = None

        # Get URI of entity and relation
        entity_uri, relation_uri = self.__get_URL_By_label(input_entity, input_predicate)
        input_entity_embedding = self.entity_embedding_dict[entity_uri]
        input_relation_embedding = self.relation_embedding_dict[relation_uri]
        final_embedding = (input_entity_embedding + input_relation_embedding).reshape((1, -1))
        # Find the closest embedding entity - Use pairwise distance to measure similarity
        distance = pairwise_distances(final_embedding, self.entity_embedding).flatten()
        highest_similarity_entity_idx = distance.argsort()[0]
        highest_similarity_entity_URI = self.idx_entity_dict[highest_similarity_entity_idx]
        # return the label of the entity based on its URI
        return self.entity_label_dict[highest_similarity_entity_URI]


    def __get_URL_By_label(self, input_entity, input_predicate):
        entity_uri = ""
        relation_uri = ""
        # If the entity and predicate already exist in dict, then just get its URI
        if self.label_entity_dict.get(input_entity):
            entity_uri = str(self.label_entity_dict[input_entity][0])
        else:
            # Max distance between two words
            distance = 1000
            for key in self.label_entity_dict.keys():
                n = editdistance.eval(input_entity, key)
                if n < distance:
                    distance = n
                    entity_uri = self.label_entity_dict[key][0]
        # Same as relations
        if self.label_relation_dict.get(input_predicate):
            relation_uri = self.label_relation_dict[input_predicate][0]
        else:
            # Max distance between two words
            distance = 1000
            for key in self.label_relation_dict.keys():
                n = editdistance.eval(input_predicate, key)
                if n < distance:
                    distance = n
                    relation_uri = self.label_relation_dict[key][0]
        return entity_uri, relation_uri

    def queryCrowdsourceQuestions(self, entity_idx, relation_idx):
        if "wd:"+entity_idx in self.crowdsource_subject_list:
            if "wdt:" + relation_idx in self.crowdsource_predicate_list:
                selected_answers = self.crowdsource_data[
                    (self.crowdsource_data['Input1ID'] == "wd:" + entity_idx) & (
                                    self.crowdsource_data['Input2ID'] == "wdt:" + relation_idx)]
            else:
                selected_answers = self.crowdsource_data[self.crowdsource_data['Input1ID'] == "wd:"+ entity_idx]
            support = (selected_answers['AnswerLabel'] == 'CORRECT').sum()
            reject = (selected_answers['AnswerLabel'] == 'INCORRECT').sum()
            ans = selected_answers['Input3ID'].unique()[0]
            if "wd:" in ans:
                ans = self.all_label_dict["http://www.wikidata.org/entity/"+ans.split(":")[-1]]
            batch_idx = selected_answers['HITTypeId'].unique()[0]
            selected_batch = self.crowdsource_data[self.crowdsource_data['HITTypeId'] == batch_idx]
            Pj = [round(((selected_batch['AnswerLabel'] == 'CORRECT').sum() / selected_batch.shape[0]), 3),
                      round(((selected_batch['AnswerLabel'] == 'INCORRECT').sum() / selected_batch.shape[0]), 3)]
            question_idx = selected_batch['HITId'].unique().tolist()
            Pi = []
            for idx in question_idx:
                n = (selected_batch['HITId'] == idx).sum()
                pos = ((selected_batch['HITId'] == idx) & (selected_batch['AnswerLabel'] == 'CORRECT')).sum()
                neg = ((selected_batch['HITId'] == idx) & (selected_batch['AnswerLabel'] == 'INCORRECT')).sum()
                # print(f'{n,pos,neg}')
                Pi.append(round((pos * (pos - 1) + neg * (neg - 1)) / (n * (n - 1)), 3))
            Po = round((sum(Pi) / len(question_idx)), 3)
            Pe = sum(x ** 2 for x in Pj)
            inter_rater = round(((Po-Pe) / (1-Pe)),3)
            return inter_rater,support,reject,ans
        else:
            return None

    def get_image(self, entity):

        entity_uri = None

        entity_uri, _ = self.__get_URL_By_label(entity, "IMDb ID")

        query = f"""
            PREFIX wdt: <http://www.wikidata.org/prop/direct/>
            SELECT ?object WHERE {{
                <{entity_uri}> wdt:P345 ?object .
                }}
            """

        imdb_id = [str(s) for s, in self.graph.query(query)]

        if not imdb_id:
            return f"Sorry, we couldn't find the item you mentioned."

        imdb_id = imdb_id[0]

        entity_type = "actor" if imdb_id.startswith("nm") else "movie" if imdb_id.startswith("tt") else None

        matching_images = [
            img for img in self.images_data
            if imdb_id in img.get("cast" if entity_type == "actor" else "movie", [])
        ]

        if not matching_images:
            return f"Sorry, no image found for this {entity_type}."

        # priority
        if entity_type == "actor":
            type_priority = ["publicity"]
            matching_images = [img for img in matching_images
                                    if len(img.get("cast", [])) == 1
                                  ] or matching_images
        elif entity_type == "movie":
            type_priority = ["poster", "still_frame", "event", "behind_the_scenes"]
        else:
            type_priority = ["publicity", "poster", "event", "still_frame", "behind_the_scenes"]

        matching_images.sort(
                key=lambda x: type_priority.index(x["type"]) if x["type"] in type_priority else len(type_priority))

        return f"Here's the picture:image:{matching_images[0]['img'].rstrip('.jpg')}"
