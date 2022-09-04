from aiogram import types
import string
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.corpus.reader.plaintext import PlaintextCorpusReader
import natasha as nt


welcome_input = ["привет", "ку", "прив", "добрый день", "доброго времени суток","здравствуйте", "приветствую"]
goodbye_input = ["пока", "стоп", "выход", "конец", "до свидания"]

corpus = PlaintextCorpusReader('corpus/', r'.*\.txt')
sentences = nltk.sent_tokenize(corpus.raw())

segmenter = nt.Segmenter()
morph_vocab = nt.MorphVocab()
emb = nt.NewsEmbedding()
morph_tagger = nt.NewsMorphTagger(emb)
ner_tagger = nt.NewsNERTagger(emb)


def normalize(text: str) -> list[str]:
    """
    Tokenize text and normalize tokens.
    """
    # remove punctuation
    word_token = text.translate(str.maketrans("", "", string.punctuation)).replace("—", "")

    # tokenize and normalize
    doc = nt.Doc(word_token)
    doc.segment(segmenter)
    doc.tag_morph(morph_tagger)
    doc.tag_ner(ner_tagger)
    for token in doc.tokens:
        token.lemmatize(morph_vocab)

    return [token.lemma for token in doc.tokens]



def process_question(question: str) -> str:
    """
    Generate response to user's question
    """
    # add question to sentences
    question = question.lower()
    sentences.append(question)

    # vectorize sentences using TF-IDF
    tfidf = TfidfVectorizer(tokenizer=normalize).fit_transform(sentences)
    # find cosine similarity between the question and other sentences
    vals = cosine_similarity(tfidf[-1], tfidf)
    # index of best match
    idx=vals.argsort()[0][-2]

    # check if result was not found
    flat = vals.flatten()
    flat.sort()
    req_tfidf = flat[-2]
    sentences.remove(question)
    if(req_tfidf==0):
        return "Извините, я не нашел ответа..."
 
    return sentences[idx]



async def question_handler(event: types.Message):
    if event.text.lower() in welcome_input:
        await event.answer('Привет!')
    elif event.text.lower() in goodbye_input:
        await event.answer('Буду ждать вас!')
    else:
        await event.answer('Дайте подумать...')
        await event.answer(process_question(event.text))
