

class Article:

    def __init__(self, title, pageid, sentences):
        self.title = title
        self.pageid = pageid
        self.sentences = sentences
        self._length = sum([sent.get_length() for sent in sentences])
        self._num_citations = sum([sent.num_citations for sent in sentences])
        self._num_sentences = len(sentences)

    def get_length(self):
        return self._length

    def get_num_citations(self):
        return self._num_citations

    def get_num_sentences(self):
        return self._num_sentences

    def to_dict(self):
        return {
            # 'title': self.title,
            'pageid': self.pageid,
            'sentences': [sent.to_dict() for sent in self.sentences]
        }


class Sentence:

    def __init__(self, text, num_citations, heading_level):
        self.text = text
        self.num_citations = num_citations
        self.heading_level = heading_level
        self._length = len(text)

    def get_length(self):
        return self._length

    def to_dict(self):
        return {
            'num_citations': self.num_citations,
            'heading_level': self.heading_level,
            'text': self.text
        }
