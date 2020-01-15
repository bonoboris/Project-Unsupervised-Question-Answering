import gensim

def word2vec(model_path, context_file):

    w2v_model = gensim.models.Word2Vec.load(model_path)

    with open(context_file) as cf:
        line = cf.readlines()
    for i, word in enumerate(w2v_model.wv.vocab):
        if i == 10:
            break
        print(word)


if __name__ == '__main__':
    model_path = 'chirac.gensim'
    context_file = './resource/context.txt'
    vecs = word2vec(model_path,context_file)
    print(vecs)