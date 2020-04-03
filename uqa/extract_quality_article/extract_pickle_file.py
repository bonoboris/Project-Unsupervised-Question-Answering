import pickle

pickle_in = open("list_articles.pickle", 'rb')

list_article = pickle.load(pickle_in)

# We need to split the con

if __name__ == '__main__':
    article = list_article[0]
    print(article)