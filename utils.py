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
            url = line[30:].strip()
            article_titles.append(url)
    return article_titles


def save_as_json(articles, file_name):
    """
    Saves a given list of Articles to a JSON-formatted file
    :param articles: the articles to save
    :param file_name: the filename for the JSON output
    """
    os.makedirs('corpus', exist_ok=True)
    with open('corpus/{}'.format(file_name), 'w') as json_file:
        json.dump({article.title: article.to_dict() for article in articles},
                  json_file,
                  sort_keys=True,
                  indent=4)


def get_corpus_stats(articles, set_name='All articles'):
    """
    Calculates various statistics about the given corpus (or corpus split)
    and formats them as a string
    :param articles: the articles to calculate statistics for
    :param set_name: The name of the set (eg 'Training')
    :return: A string (Markdown formatted) containing statistics for the given set
    """
    av_length = int(sum([article.get_length() for article in articles]) / len(articles))
    av_num_sentences = sum([article.get_num_sentences() for article in articles]) / len(articles)
    av_num_citations = sum([article.get_num_citations() for article in articles]) / len(articles)

    return '**{}** ({} articles):\n\n' \
           '- Average length (characters): {}\n' \
           '- Average length (sentences): {:.2f}\n' \
           '- Average number of citations: {:.2f}\n'.format(set_name,
                                                            len(articles),
                                                            av_length,
                                                            av_num_sentences,
                                                            av_num_citations)


def save_stats_to_md(all_articles, train, dev, test):
    """

    :param all_articles:
    :param train:
    :param dev:
    :param test:
    """
    with open('corpus/stats.md', 'w') as stats:
        stats.write('# Corpus #\n\n## Stats ##\n\n')
        stats.write('{}\n'.format(get_corpus_stats(all_articles)))
        stats.write('{}\n'.format(get_corpus_stats(train, 'Training set')))
        stats.write('{}\n'.format(get_corpus_stats(dev, 'Dev set')))
        stats.write('{}\n'.format(get_corpus_stats(test, 'Test set')))
