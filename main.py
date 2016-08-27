import json

from utils import save_as_json, get_corpus_stats, save_stats_to_md
from create_corpus import CreateCorpus
from article import Article


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
    all_articles = []
    for title, article in semi_cleaned_articles.items():
        all_articles.append(Article(title, article['pageid'], corpus.get_sentences_and_citations(article['text'])))
        # limit += 1
        # if limit == 3:
        #     break

    train, dev, test = corpus.get_corpus_splits(all_articles)

    # print(get_corpus_stats(all_articles))
    # print(get_corpus_stats(train))
    # print(get_corpus_stats(dev))
    # print(get_corpus_stats(test))

    save_as_json(all_articles, 'all_articles.json')
    save_as_json(train, 'train.json')
    save_as_json(dev, 'dev.json')
    save_as_json(test, 'test.json')

    save_stats_to_md(all_articles, train, dev, test)

    # with open('article_sentences.json', 'w') as json_file:
    #     json.dump(article_sentences, json_file, sort_keys=True, indent=4)

