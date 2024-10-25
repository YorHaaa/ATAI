from typing import final

import numpy as np
import rdflib
import editdistance
from rdflib import URIRef, RDFS
import csv
from sklearn.metrics import pairwise_distances

# Read the nt file to generate knowledge graph
graph = rdflib.Graph()
graph.parse('Dataset/14_graph.nt', format='turtle')
print("----KG load succeed----")

class QueryExecutor:
    def __init__(self):
        self.graph = graph

        # Namespace
        self.WD = rdflib.Namespace('http://www.wikidata.org/entity/')
        self.WDT = rdflib.Namespace('http://www.wikidata.org/prop/direct/')
        self.DDIS = rdflib.Namespace('http://ddis.ch/atai/')
        self.RDFS = rdflib.namespace.RDFS
        self.SCHEMA = rdflib.Namespace('http://schema.org/')

        # Build entity_label and relation_label dictionary
        self.all_label_dict = {str(sub): str(obj) for sub, pre, obj in graph.triples((None,RDFS.label,None))}
        self.entity_label_dict = {entity:label for entity, label in self.all_label_dict.items() if
                                    self.__is_entity(entity)}
        self.relation_label_dict = {entity:label for entity, label in self.all_label_dict.items() if
                                    self.__is_relation(entity)}
        # Reverse following dictionaries, and store same label's URIs as an array
        label_entity_dict = {}
        for key, value in self.entity_label_dict.items():
            if value in label_entity_dict:
                label_entity_dict[value].append(key)
            else:
                label_entity_dict[value] = [key]
        self.label_entity_dict = label_entity_dict

        label_relation_dict = {}
        for key, value in self.relation_label_dict.items():
            if value in label_relation_dict:
                label_relation_dict[value].append(key)
            else:
                label_relation_dict[value] = [key]
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




    def __get_entity_index(self, uri):
        return str(uri).split('/')[-1]

    def __is_relation(self, uri):
        label = self.__get_entity_index(uri)
        return label[0] == 'P'

    def __is_entity(self, uri):
        label = self.__get_entity_index(uri)
        return label[0] == 'Q'

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




if __name__ == '__main__':
    Q = QueryExecutor()