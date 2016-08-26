

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