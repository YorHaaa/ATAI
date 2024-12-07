import editdistance
from rdflib import URIRef, RDFS,Namespace
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class Recommender:
    def __init__(self,graph):
        self.graph = graph
        
        # Define namespaces for Wikidata entities and properties
        self.WD = Namespace('http://www.wikidata.org/entity/')
        self.WDT = Namespace('http://www.wikidata.org/prop/direct/')

        # Define a dictionary to filter specific properties of movies
        self.TARGET_PROPERTIES = {
            'director': self.WDT['P57'],      # Property for director
            'producer': self.WDT['P272'],     # Property for producer
            'genre': self.WDT['P136'],        # Property for genre
            'language': self.WDT['P364'],     # Property for language
            'release_date': self.WDT['P577']  # Property for release date
        }

        # Preload movie features and name-to-URI mapping
        self.movies, self.name_to_uri, self.uri_to_name = self.preload_movies_with_features(graph)

    def preload_movies_with_features(self,graph):
        movies = {}  # Dictionary to hold movie URIs and their features
        name_to_uri = {}  # Dictionary to map movie names to their URIs
        uri_to_name = {}

        def get_subclasses(entity, graph):
            subclasses = set()
            for subclass in graph.subjects(self.WDT['P279'], entity):
                subclasses.add(subclass)
                subclasses.update(get_subclasses(subclass, graph))
            return subclasses

        film_types = get_subclasses(self.WD['Q11424'], graph)
        film_types.add(self.WD['Q11424'])
        # Iterate over all movie entities in the graph
        for film_type in film_types:
            for movie_uri, _, _ in graph.triples((None, self.WDT['P31'], film_type)):
                features = []  # List to hold features of the current movie
                # Iterate over the target properties to collect features
                for prop_name, prop_uri in self.TARGET_PROPERTIES.items():
                    if prop_name == "release_date":
                        for _, _, value in graph.triples((movie_uri, prop_uri, None)):
                            date = str(value)[0:3]
                            features.append(f"{prop_name}::{date}|")  # Append feature in the format "property::value"
                    else:
                        for _, _, value in graph.triples((movie_uri, prop_uri, None)):
                            # {prop_name}::
                            features.append(f"{prop_name}::{value}|")  # Append feature in the format "property::value"
                feature_string = " ".join(features)  # Create a single string of features
                movies[str(movie_uri)] = feature_string  # Store the feature string in the movies dictionary
            
        # Retrieve the movie's label (name) and map it to its URI
                for _, _, label in graph.triples((movie_uri, RDFS.label, None)):
                    label_str = str(label)
                    uri_to_name[str(movie_uri)] = label_str
                    if label_str not in name_to_uri:
                        name_to_uri[label_str] = []  # Initialize list if not present
                    name_to_uri[label_str].append(movie_uri)  # Add the movie URI to the list
        print("movie feature dic ok")
        return movies, name_to_uri, uri_to_name  # Return the movies dictionary and name-to-URI mapping



    def recommend_by(self, movie_names, top_k=5):
        user_movie_uris = []  # List to hold URIs of user-provided movies
        # Retrieve URIs for each movie name provided by the user
        for name in movie_names:
            # uris = self.name_to_uri.get(name)
            # if uris:
            #     user_movie_uris.extend(uris)  # Add all matching URIs
            # else:
            #     user_movie_not_found.append(name)
            if self.name_to_uri.get(name):
                user_movie_uris.append(str(self.name_to_uri[name][0]))
            else:
                # Max distance between two words
                distance = 1000
                entity_uri = ""
                for key in self.name_to_uri.keys():
                    n = editdistance.eval(name, key)
                    if n < distance:
                        distance = n
                        entity_uri = str(self.name_to_uri[key][0])
                user_movie_uris.append(entity_uri)

        # Build a combined feature vector for the user's input movies
        user_features = []
        for uri in user_movie_uris:
            user_features.append(self.movies[str(uri)])  # Append features of each movie
        user_text = " ".join(user_features)  # Create a single string of user features

        # Vectorize the text data
        movie_texts = list(self.movies.values())  # Get all movie feature strings
        vectorizer = TfidfVectorizer(token_pattern=r"[^|]+").fit_transform([user_text] + movie_texts)  # Fit and transform the data
        vectors = vectorizer.toarray()  # Convert to array format

        # Calculate cosine similarity between the user's features and all movie features
        similarities = cosine_similarity(vectors[0].reshape(1, -1), vectors[1:]).flatten()
        similar_indices = similarities.argsort()[::-1]  # Get indices of movies sorted by similarity

        filtered_indices = [i for i in similar_indices if list(self.movies.keys())[i] not in user_movie_uris]

        # Collect recommended movies, excluding those already provided by the user
        recommended_movies = []
        for i in filtered_indices:
            movie_uri = list(self.movies.keys())[i]  # Get the movie URI
            if URIRef(movie_uri) not in user_movie_uris:  # Check if it's not a user movie
                recommended_movies.append(movie_uri)  # Add to recommendations
            if len(recommended_movies) == top_k:  # Stop if we have enough recommendations
                break

        # Retrieve the names of the recommended movies
        movie_names = []
        for movie in recommended_movies:
            movie_names.append(self.uri_to_name[str(movie)])  # Append the movie name to the list
        return f"Here're our recommdations:{movie_names}."  # Return the list of recommended movie names