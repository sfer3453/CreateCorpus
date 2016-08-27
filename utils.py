import os
import json

def get_article_titles(file_path):
    """
    Read the titles of the required articles from the given file of WikiPedia URLs
    :param path: the location of the file
    :return: a list of article titles
    """
    article_titles = []
    with open(file_path, 'r') as url_file:
        for line in url_file:
            url = line[30:].strip()  # parse.unquote(line[30:].strip()) #.decode('utf-8')
            article_titles.append(url)
    return article_titles

def save_as_json(articles, file_name):
    os.makedirs('corpus', exist_ok=True)

    with open('corpus/{}'.format(file_name), 'w') as json_file:
        json.dump({article.title: article.to_dict() for article in articles}, json_file, sort_keys=True, indent=4)

def get_corpus_stats(articles):
    av_length = int(sum([article.get_length() for article in articles]) / len(articles))
    av_num_sentences = int(sum([article.get_num_sentences() for article in articles]) / len(articles))
    av_num_citations = sum([article.get_num_citations() for article in articles]) / len(articles)

    return 'Averages ({} articles):\n' \
           '\tnum characters: {}\n' \
           '\tnum sentences: {}\n' \
           '\tnum citations: {}\n'.format(len(articles),
                                          av_length,
                                          av_num_sentences,
                                          av_num_citations)