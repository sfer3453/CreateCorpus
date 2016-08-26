from Utils import get_article_titles
import json
from CreateCorpus import CreateCorpus


if __name__ == '__main__':
    saved_file = 'raw_articles.json'
    download = False
    corpus = CreateCorpus(download=download,
                          url_file='article_urls.txt',
                          saved_file=saved_file)

    # Save articles to JSON file
    if download:
        with open(saved_file, 'w') as json_file:
            json.dump(corpus.articles, json_file, sort_keys=True, indent=4)

    print([title for title in corpus.articles.keys()])

    semi_cleaned_articles = {}
    for title, article in corpus.articles.items():
        semi_cleaned_articles[title] = {
            'pageid': article['pageid'],
            'text': corpus.strip_mediawiki_markup(article['text'])
        }

    # limit = 0
    article_sentences = {}
    for title, article in semi_cleaned_articles.items():
        # print(title)
        article_sentences[title] = {
            'pageid': article['pageid'],
            'sentences': corpus.get_sentences_and_citations(article['text'])
        }
        # for sentence in article_sentences[title]['sentences']:
        #     # if sentence['heading_level'] != 0:
        #     print('\tcits: {}, hl: {}, text: {}'.format(sentence['num_citations'], sentence['heading_level'], repr(sentence['text'][:120])))
        # limit += 1
        # if limit == 3:
        #     break

    with open('article_sentences.json', 'w') as json_file:
        json.dump(article_sentences, json_file, sort_keys=True, indent=4)

