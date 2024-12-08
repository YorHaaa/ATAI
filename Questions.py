"""
Core idea:
    1. Use spacy to extract the relation and entities from user's input
    2. According to the relation extracted from previous steps, generating and executing SPARQL
    3. Return output to user based on templates.
"""
import rdflib
import spacy
from rdflib import RDFS
from Query import QueryExecutor
from Recommender import Recommender

model = spacy.load("en_core_web_trf")
print("----Spacy model load succeed----")
graph = rdflib.Graph()
graph.parse('Dataset/14_graph.nt', format='turtle')
print("----KG load succeed----")
recommender = Recommender(graph)

class Question:

    def __init__(self, question_text):
        self.entities = None
        self.question_text = question_text
        self.doc = None
        self.question_type = None
        self.question_type_dict = {
            "factual question" : 0,
            "embedding question" : 1,
            "recommendation question" : 2,
            "crowdsource question" : 3
        }
        self.predicate = None
        self.answer = None
        self.recommender = recommender
        # Synonyms words dictionary
        self.synonyms_dict = {
            'cast member': ['actor', 'actress', 'cast'],
            'genre': ['type', 'kind'],
            'publication date': ['release', 'date', 'airdate', 'publication', 'launch', 'broadcast', 'released',
                                 'launched'],
            'executive producer': ['showrunner'],
            'screenwriter': ['scriptwriter', 'screenplay', 'teleplay', 'writer', 'script', 'scenarist', 'story'],
            'director of photography': ['cinematographer', 'DOP', 'dop'],
            'film editor': ['editor'],
            'production designer': ['designer'],
            'box office': ['box', 'office', 'funding'],
            'cost': ['budget', 'cost'],
            'nominated for': ['nomination', 'award', 'finalist', 'shortlist', 'selection'],
            'costume designer': ['costume'],
            'official website': ['website', 'site'],
            'filming location': ['flocation'],
            'narrative website': ['nlocation'],
            'production company': ['company'],
            'country of origin': ['origin', 'country'],
            'director': ['directed', 'directs'],
            'IMDb ID': ['IMDb', 'IMDB', 'imdb']
        }
        # Answer templates
        self.answer_templates = {
            "factual question default" : """According to my records in my knowledge graph, 
            the search results of {entity}'s {predicate} are as follows : {query_result}""",
            "publication date" : "The {entity} was released in {query_result} based on my knowledge graph",
            "box office" : "The box office of {entity} is {query_result}",
            "genre" : "Here are the genres of {entity} : {query_result}",
            "IMDb ID" : "The IMDb ID of movie {entity} is {query_result}",
            "embedding question" : "I can't answer this question from my KG. "
                                   "Here is the answer compute by embedding: {embedding_result}",
            "Recommendation question" : "Here are the top 5 recommendations: {rec_result}"
        }
        self.graph = graph

        # Namespace
        self.WD = rdflib.Namespace('http://www.wikidata.org/entity/')
        self.WDT = rdflib.Namespace('http://www.wikidata.org/prop/direct/')
        self.DDIS = rdflib.Namespace('http://ddis.ch/atai/')
        self.RDFS = rdflib.namespace.RDFS
        self.SCHEMA = rdflib.Namespace('http://schema.org/')

        # Build entity_label and relation_label dictionary
        self.all_label_dict = {str(sub): str(obj) for sub, pre, obj in graph.triples((None, RDFS.label, None))}
        self.entity_label_dict = {entity: label for entity, label in self.all_label_dict.items() if
                                  self.__is_entity(entity)}
        self.relation_label_dict = {entity: label for entity, label in self.all_label_dict.items() if
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
            if "http://www.wikidata.org/entity/" in key:
                key = key.replace("http://www.wikidata.org/entity/", "http://www.wikidata.org/prop/direct/")
            if value in label_relation_dict:
                label_relation_dict[value].append(key)
            else:
                label_relation_dict[value] = [key]
        self.label_relation_dict = label_relation_dict
        self.queryExecutor = QueryExecutor(graph,self.all_label_dict,self.entity_label_dict,self.relation_label_dict
                                           ,self.label_entity_dict,self.label_relation_dict)

    def __get_entity_index(self, uri):
        return str(uri).split('/')[-1]

    def __is_relation(self, uri):
        label = self.__get_entity_index(uri)
        return label[0] == 'P'

    def __is_entity(self, uri):
        label = self.__get_entity_index(uri)
        return label[0] == 'Q'

    def parseQuestion(self):
        # If the user's input is a SPARQL query, then just returns the executing result.
        if "SELECT" in self.question_text or "select" in self.question_text :
            return self.queryExecutor.querySPARQL(self.question_text)

        self.doc = model(self.question_text)

        # NER
        entities = []
        if self.__extractEntities():
            entities = self.__extractEntities()
        else:
            for entity, label in self.entity_label_dict.items():
                if label.lower() in self.question_text.lower() and len(label)>= 6:
                    entities.append(label)
        self.entities = entities

        # Get predicates from the input sentences
        predicate = None
        for relation, label in self.relation_label_dict.items():
            if label.lower() in self.question_text.lower():
                predicate = label

        if predicate is None:
            if self.__extractRelations():
                predicate = self.__extractRelations()[0]
            else:
                return "Sorry, I can't find any relations in your input, please try another way."

        # Deal with synonyms word
        for relation, synonyms_word in self.synonyms_dict.items():
            if predicate in synonyms_word:
                predicate = relation

        self.predicate = predicate

        # Processing recommendation Questions
        if any(word in self.question_text.lower() for word in ["recommend", "recommended", "advise", "suggest",
                                                               "similar", "like"]):
            print("rec_mode")
            print(entities)
            return self.recommender.recommend_by(entities)

        # Processing CrowdSourcing questions
        entity_idx = self.__get_entity_index(self.label_entity_dict.get(max(self.entities, key=len, default=''))[0])
        relation_idx = self.__get_entity_index(self.label_relation_dict.get(self.predicate)[0])
        crowd_result = self.queryExecutor.queryCrowdsourceQuestions(entity_idx,relation_idx)
        if crowd_result:
            inter_rater, support, reject, ans = crowd_result
            if len(self.predicate) > 0:
                ans = f"The {self.predicate} of {self.entities[0]} is {ans}\nInter-rater agreement: {inter_rater} Support votes: {support} Reject votes: {reject}"
            else:
                movie_entity = self.entities[0].capitalize()
                ans = f'{movie_entity} is the subclass of {ans}\nInter-rater agreement: {inter_rater} Support votes: {support} Reject votes: {reject}'

            return ans

        # Processing factual questions
        query_result = self.queryExecutor.queryFactualQuestions(max(self.entities, key=len, default=''), self.predicate)
        if len(query_result) != 0:
            self.question_type = 0
            return self.__generateAnswer(query_result)
        else:
            # Precessing embedding questions
            embedding_result = self.queryExecutor.queryEmbeddingQuestions(max(self.entities, key=len, default=''), self.predicate)
            if embedding_result:
                return self.answer_templates["embedding question"].format(embedding_result=embedding_result)
            else:
                return "Sorry, I can't answer your question now because of my limited knowledge ~ 🤣"


    def __extractEntities(self):
        return [ent.text for ent in self.doc.ents if ent.label_ == "WORK_OF_ART"]

    def __extractRelations(self):
        predicates = []
        for token in self.doc:
            if token.dep_ in ("attr", "nsubj", "ROOT") and token.pos_ in ("NOUN", "VERB"):
                predicates.append(token.text)
        return predicates

    def __generateAnswer(self, query_result):
        query_result = ", ".join(query_result)
        # Factual questions
        if self.question_type == 0 :
            if self.predicate == "publication date":
                return self.answer_templates["publication date"].format(entity=max(self.entities, key=len, default=''), query_result=query_result)
            elif self.predicate == "genre":
                return self.answer_templates["genre"].format(entity=max(self.entities, key=len, default=''), query_result=query_result)
            elif self.predicate == "box office":
                return self.answer_templates["box office"].format(entity=max(self.entities, key=len, default=''), query_result=query_result)
            elif self.predicate == "IMDb ID":
                return self.answer_templates["IMDb ID"].format(entity=max(self.entities, key=len, default=''), query_result=query_result)
            else:
                return self.answer_templates["factual question default"].format(entity=max(self.entities, key=len, default=''), predicate=self.predicate, query_result=query_result)


