import requests
import re
import html
import random
import json
from urllib import parse

import spacy

from utils import get_article_titles
from article import Article, Sentence


class CreateCorpus:

    def __init__(self, url_file, saved_file, download=False):
        self.article_titles = get_article_titles(url_file)
        if download:
            self.articles = self.download_articles(self.article_titles)
        else:
            self.articles = self.open_articles(saved_file)
        self.nlp = None

    def download_articles(self, article_titles):
        wikipedia_endpoint = 'https://en.wikipedia.org/w/api.php?action=query&titles={}&prop=revisions&rvprop=content&format=json'

        start_idx = 0
        articles = {}
        missing_articles = set()
        while start_idx < len(article_titles):
            end_idx = min(start_idx + 20, len(article_titles))
            titles = '|'.join(re.sub(' ', '_', url) for url in article_titles[start_idx:end_idx])
            r = requests.get(wikipedia_endpoint.format(titles))

            if r.status_code >= 300:
                print('Something went wrong')
                raise requests.exceptions.RequestException()

            pages = r.json()['query']['pages']

            if len(pages) != end_idx - start_idx:
                missing_articles.update(([page['title'] for page in pages.values() if
                                          page['title'] not in article_titles]))

            for page in pages.values():
                articles[parse.unquote(page['title'])] = {
                    'pageid': page['pageid'],
                    'text': page['revisions'][0]['*']
                }

            start_idx += 20

        print('Number of article urls given: {}'.format(len(articles)))
        print('Number of articles missed: {}'.format(len(missing_articles)))
        print('Missed articles: {}'.format(missing_articles))
        return articles

    def open_articles(self, file_path):
        with open(file_path, 'r') as json_file:
            articles = json.load(json_file)
            return articles

    def strip_mediawiki_markup(self, text):
        """
        Strips all WikiMedia markup from a given document apart from section headings and
        reference links
        :param text: the text to clean
        :return: the cleaned text
        """
        rx_comment     = re.compile(r'<!--.*?-->', re.DOTALL | re.UNICODE)
        rx_end_matter  = re.compile(r'\n==( )?(Notes|References|See [Aa]lso|External [Ll]inks|Further [Rr]eading)( )?==\n.*$', re.DOTALL | re.UNICODE)

        rx_link        = re.compile(r'\[\[(?!([Ii]mage:|[Ff]ile:))(?P<innertext>(\[[^\[\]]+\]|[^\[\]\|]+){0,3})\]\]', re.DOTALL | re.UNICODE)
        rx_link_alt    = re.compile(r'\[\[(?!([Ii]mage:|[Ff]ile:))([^\[\]\|]*)\|(?P<innertext>(\[[^\[\]]+\]|[^\[\]\|]+){0,3})\]\]', re.DOTALL | re.UNICODE)
        rx_link_ext    = re.compile(r'(?!<=\[)\[[^\[\] ]*( (?P<innertext>[^\[\]]{0,200}))?\](?!\])', re.DOTALL | re.UNICODE)
        rx_anchor      = re.compile(r'{{[Vv]isible [Aa]nchor( )?\|(?P<innertext>[^{}]*)}}')
        rx_template    = re.compile(r'{{(?!([Cc]itation [Nn]eeded|cn))[^{}]*}}', re.DOTALL | re.UNICODE)
        rx_file_image  = re.compile(r'\[\[([Ii]mage:|[Ff]ile:)[^\[\]]*(\[\[[^\[\]]*\]\]|\[[^\[\]]*\]|[^\[\]]*){0,10}\]\]', re.DOTALL | re.UNICODE)  # may have nested [[...]]
        rx_gallery     = re.compile(r'<[Gg]allery(.*?)</[Gg]allery>', re.DOTALL | re.UNICODE)
        rx_table_inner = re.compile(r'{\|[^{}]*\|}((\s)*?<ref([^>]*/>|[^<]*</ref>))?', re.DOTALL | re.UNICODE)
        rx_table       = re.compile(r'{\|.*?\|}((\s)*?<ref([^>]*/>|[^<]*</ref>))?', re.DOTALL | re.UNICODE)
        rx_bold_italic = re.compile(r"'{2,6}(?P<innertext>.*?)'{2,6}", re.DOTALL | re.UNICODE)

        rx_tag_empty   = re.compile(r'<(?![Rr]ef)[^>]*/>|<[Bb][Rr]>|<p>', re.DOTALL | re.UNICODE)
        rx_tag_no_keep = re.compile(r'<(?![Rr]ef)(?P<tag>([Gg]allery|[Tt]imeline|[Ss]yntaxhighlight))( [^>]*)?>.*?</(?P=tag)>', re.DOTALL | re.UNICODE)
        rx_tag_keep    = re.compile(r'<(?![Rr]ef)(?P<tag>[a-zA-Z]*)( [^>]*)?>(?P<innertext>.*?)</(?P=tag)>', re.DOTALL | re.UNICODE)
        rx_infobox     = re.compile(r'{{( )?[Ii]nfobox[^{}]*({{[^{}]*({{[^{}]*}}|[^{}]*)*}}|{[^{}]*}|[^{}]*)*}}', re.DOTALL | re.UNICODE)

        rx_html_esc    = re.compile(r'(?P<esc>&[\w]{4,6};)')
        rx_spaces      = re.compile(r'^[\s]+')

        text = rx_end_matter.sub('', text)      # Remove end-matter
        text = rx_comment.sub('', text)         # Remove comments
        text = rx_anchor.sub('\g<innertext>', text) # Remove "visible anchors"

        # Remove links and non-"citation needed" templates (loop gets rid of nested ones)
        n_subs = 1
        while n_subs != 0:
            text, i = rx_link.subn('\g<innertext>', text)
            text, j = rx_link_alt.subn('\g<innertext>', text)
            text, k = rx_template.subn('', text)
            n_subs = i + j + k

        text = rx_table_inner.sub('', text)     # Remove inner tables
        text = rx_table.sub('', text)           # Remove remaining outer tables

        text = rx_file_image.sub('', text)      # Remove images/captions
        text = rx_link_ext.sub('\g<innertext>', text)       # Remove external links
        text = rx_gallery.sub('', text)         # Remove galleries of multiple images
        text = rx_bold_italic.sub('\g<innertext>', text)    # Remove bold/italic markup
        text = rx_tag_no_keep.sub('', text)     # Remove HTML tags and unneeded content
        text = rx_tag_keep.sub('\g<innertext>', text)       # Remove tags, leave content
        text = rx_tag_empty.sub('', text)       # Remove empty (self-closing eg <br />) tags
        text = rx_infobox.sub('', text)         # Remove infoboxes (MUST be done after other template removals)

        text = rx_html_esc.sub(lambda m: html.unescape(m.group('esc')), text)   # Replace HTML-escaped characters
        text = rx_spaces.sub('', text)          # Remove spaces/newlines at start of article
        
        return text

    def get_sentences_and_citations(self, text):
        """
        Takes text which has been stripped of all WikiMedia markup other than heading markup or
        markup for reference locations/citation needed locations
        :param text: the semi-cleaned text
        :return: a list of Sentences, each containing information about whether there are citations
        relating to it and whether it is a heading (level 2, 3 etc)
        """
        rx_heading  = re.compile(r'(?<=\n)(?P<hlevel>={2,5})( ){0,2}(?P<htext>[^=\n]{0,200}([^ ]|<[Rr]ef[^>]*/>))( ){0,2}(?P=hlevel)( )?(?=\n)', re.DOTALL | re.UNICODE)
        rx_citation = re.compile(r'<[Rr]ef([^>]*[^/])?>.*?</[Rr]ef>|<[Rr]ef[^>]*/>|{{([Cc]itation [Nn]eeded|cn)[^{}]*}}', re.DOTALL | re.UNICODE)

        if self.nlp is None:
            print('Loading spaCy (only required first time method is called, can take ~20s)')
            self.nlp = spacy.load('en')
            print('Done loading spaCy')

        heading_offsets = []    # List of (offset, heading level) eg 2 for H2
        citation_offsets = []
        heading_match = rx_heading.search(text)
        citation_match = rx_citation.search(text)
        from_idx = 0
        while heading_match or citation_match is not None:
            if citation_match is None or (heading_match is not None and heading_match.start() < citation_match.start()):
                heading_offsets.append((heading_match.start(), len(heading_match.group('hlevel'))))
                text = rx_heading.sub('\g<htext>', text, count=1)
                from_idx = heading_match.start()
                # print('\t', text[heading_match.start():heading_match.start() + len(heading_match.group('htext'))])
            else:
                citation_offsets.append(citation_match.start())
                text = rx_citation.sub('', text, count=1)
                from_idx = citation_match.start()
                # print('\t', text[from_idx - 20:from_idx])

            heading_match = rx_heading.search(text, from_idx)
            citation_match = rx_citation.search(text, from_idx)

        # print('\theading offsets ({}): {}'.format(len(heading_offsets), heading_offsets))
        # print('\tcitation offsets ({}): {}'.format(len(citation_offsets), citation_offsets))

        sp_doc = self.nlp(text)
        sentences = []
        start_offset = 0
        end_offset = 0
        curr_cit_idx = 0
        curr_hdg_idx = 0

        for i, sent in enumerate(sp_doc.sents):
            # print('start offset: {}\n\t{}'.format(start_offset, sent.text))
            # Spacy drops spaces after sentences (but not newlines)
            end_offset = start_offset + len(sent.text)
            while end_offset < len(text) and text[end_offset] == ' ':
                end_offset += 1

            # The current "sentence" may actually be several sentences (including headings)
            # Count the number of headings in this "sentence"
            tmp_hdg_idx = curr_hdg_idx
            while tmp_hdg_idx < len(heading_offsets) and heading_offsets[tmp_hdg_idx][0] < end_offset:
                tmp_hdg_idx += 1

            spans = []  # (span_start_offset, span_end_offset, heading_level)
            if tmp_hdg_idx == curr_hdg_idx:
                spans.append((start_offset, end_offset, 0))
            else:
                for idx in range(curr_hdg_idx, tmp_hdg_idx):
                    # Sentence has at least one heading (may not be split by spaCy but we want to
                    #  split it at the newline after each heading)
                    if idx == curr_hdg_idx and heading_offsets[idx][0] != start_offset:
                        spans.append((start_offset, heading_offsets[idx][0], 0))
                    span_start = heading_offsets[idx][0]
                    span_end = heading_offsets[idx + 1][0] if idx != tmp_hdg_idx - 1 else end_offset

                    # Check there is no newline contained in the span (after a heading)
                    sub_span_start = span_start - start_offset
                    sub_span_end = span_end - start_offset
                    found_newline = False
                    for i, c in enumerate(sent.text[sub_span_start:sub_span_end]):
                        if c == '\n' and i + sub_span_start + 1 < len(sent.text) and sent.text[i + sub_span_start + 1] != '\n':
                            found_newline = True
                            spans.append((sub_span_start + start_offset, sub_span_start + i + start_offset + 1, heading_offsets[idx][1]))
                            sub_span_start = sub_span_start + i + 1
                            break

                    if span_end > sub_span_start + start_offset:
                        spans.append((sub_span_start + start_offset, span_end, 0 if found_newline else heading_offsets[idx][1]))

            curr_hdg_idx = tmp_hdg_idx

            for span_start, span_end, heading_level in spans:
                # Find citations which relate to this sentence - headings don't have citations so even if the
                # sentence has been split above
                n_cits = 0
                while curr_cit_idx < len(citation_offsets) and citation_offsets[curr_cit_idx] <= span_end:
                    n_cits += 1
                    curr_cit_idx += 1

                sentences.append(Sentence(sent.text[span_start - start_offset:span_end - start_offset],
                                          n_cits,
                                          heading_level))
            start_offset = end_offset

        sentences = self._fix_headings(sentences)
        return sentences

    def _fix_headings(self, sentences):
        """
        Some headings are split into two sentences by spaCy when a colon is present...
        :param sentences:
        :return:
        """
        fixed = []
        skip_next = False
        for i, sent in enumerate(sentences):
            if skip_next:
                skip_next = False
                continue

            if sent.heading_level != 0 and sent.text[-1] == ':' and i < len(sentences) - 1:
                next_text = sentences[i + 1].text
                for j, c in enumerate(next_text):
                    if c == '\n':
                        if j < len(next_text) - 1 and next_text[j + 1] != '\n':
                            # Part of next sentence belongs to heading
                            new_text = '{} {}'.format(sent.text, next_text[:j + 1])
                            next_text = next_text[j + 1:]
                            fixed.append(Sentence(new_text, sent.num_citations, sent.heading_level))
                            fixed.append(Sentence(next_text, sentences[i + 1].num_citations, sentences[i + 1].heading_level))
                            skip_next = True
                            break
                        elif j == len(next_text) - 1:
                            # Entire next sentence belongs to heading
                            new_text = '{} {}'.format(sent.text, next_text)
                            fixed.append(Sentence(new_text, sent.num_citations, sent.heading_level))
                            skip_next = True
                            break
                    if j == 60:
                        break
            else:
                fixed.append(sent)
        return fixed

    def get_corpus_splits(self, all_articles):
        """
        Splits the given articles into training, dev and test sets with ratio roughly 3:1:1 in size
        :param all_articles: the articles to split into training, dev and test sets
        :return: tuple of training, dev and test sets
        """
        done = False

        while not done:
            indices = [i for i in range(len(all_articles))]
            random.shuffle(indices)

            train_indices = indices[:int(len(indices) * 3 / 5)]
            dev_indices = indices[int(len(indices) * 3 / 5):int(len(indices) * 4 / 5)]
            test_indices = indices[int(len(indices) * 4 / 5):]

            train = [article for i, article in filter(lambda i_art: i_art[0] in train_indices, enumerate(all_articles))]
            dev = [article for i, article in filter(lambda i_art: i_art[0] in dev_indices, enumerate(all_articles))]
            test = [article for i, article in filter(lambda i_art: i_art[0] in test_indices, enumerate(all_articles))]

            done = self._corpus_is_ok(all_articles, train, dev, test)

        return train, dev, test

    def _corpus_is_ok(self, all_articles, train, dev, test,
                      max_av_len_diff=200, max_av_sents_diff=2, max_av_cits_diff=0.3):
        """
        Tests that each set in the given corpus split is similar enough to the overall average
        length (in characters), number of sentences and number of citations
        :param all_articles:
        :param train:
        :param dev:
        :param test:
        :param max_av_len_diff:
        :param max_av_sents_diff:
        :param max_av_cits_diff:
        :return:
        """

        all_av_length = self._av_len(all_articles)
        all_av_sents = self._av_num_sentences(all_articles)
        all_av_cits = self._av_num_citations(all_articles)

        min_av_len = all_av_length - max_av_len_diff
        max_av_len = all_av_length + max_av_len_diff
        min_av_sents = all_av_sents - max_av_sents_diff
        max_av_sents = all_av_sents + max_av_sents_diff
        min_av_cits = all_av_cits - max_av_cits_diff
        max_av_cits = all_av_cits + max_av_cits_diff

        return  min_av_len   <= self._av_len(train)           <= max_av_len \
            and min_av_len   <= self._av_len(dev)             <= max_av_len \
            and min_av_len   <= self._av_len(test)            <= max_av_len \
            and min_av_sents <= self._av_num_sentences(train) <= max_av_sents \
            and min_av_sents <= self._av_num_sentences(dev)   <= max_av_sents \
            and min_av_sents <= self._av_num_sentences(test)  <= max_av_sents \
            and min_av_cits  <= self._av_num_citations(train) <= max_av_cits \
            and min_av_cits  <= self._av_num_citations(dev)   <= max_av_cits \
            and min_av_cits  <= self._av_num_citations(test)  <= max_av_cits

    def _av_len(self, articles):
        return int(sum([article.get_length() for article in articles]) / len(articles))

    def _av_num_sentences(self, articles):
        return int(sum([article.get_num_sentences() for article in articles]) / len(articles))

    def _av_num_citations(self, articles):
        return sum([article.get_num_citations() for article in articles]) / len(articles)
