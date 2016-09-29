import requests
import re
import html
import random
import json
from urllib import parse
from collections import namedtuple
from string import whitespace

# import spacy
import nltk
from nltk.tokenize.punkt import PunktSentenceTokenizer, PunktLanguageVars

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
            # Download 20 articles per request
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
        rx_lang        = re.compile(r'{{[Ll]ang[^{}]*\|(?P<innertext>[^{}|]+)}}', re.DOTALL | re.UNICODE)
        rx_jap         = re.compile(r"{{[Nn]ihongo\|(?P<innertext>([^{}\|\[\]]*\[\[[^\[\]]*\]\])?[^{}\|\[\]]*)\|[^{}]*?}}", re.DOTALL | re.UNICODE)
        rx_ill         = re.compile(r'{{(Interlanguage link|Ill)(\|[^{}]*)?\|(?P<innertext>[^{}\|]*)}}', re.DOTALL | re.UNICODE | re.IGNORECASE)
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
        rx_indent      = re.compile(r'\n[\s]*:[\s]*', re.DOTALL | re.UNICODE)
        rx_bullet      = re.compile(r'(?<=\n)[*\u2022]+[\s]*', re.DOTALL | re.UNICODE)
        rx_spaces      = re.compile(r'^[\s]+')

        text = rx_end_matter.sub('', text)           # Remove end-matter
        text = rx_comment.sub('', text)              # Remove comments
        text = rx_anchor.sub('\g<innertext>', text)  # Remove "visible anchors"
        text = rx_lang.sub('\g<innertext>', text)    # Replace transliteration templates
        text = rx_jap.sub('\g<innertext>', text)     # Replace Japanese transliterations
        text = rx_ill.sub('\g<innertext>', text)     # Replace interlanguage links
        
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
        text = rx_indent.sub('\n', text)        # Remove indents (colon at start of line)
        text = rx_bullet.sub('• ', text)        # Convert * to • (replace one/many with single), normalize spaces
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
        SentSpan = namedtuple('SentSpan', ['start_offset', 'end_offset', 'h_level'])
        
        # if self.nlp is None:
        #     print('Loading spaCy (only required first time method is called, can take ~20s)')
        #     self.nlp = spacy.load('en')
        #     print('Done loading spaCy')
        
        # Remove heading markup and citation markup from the text
        #     headings: keep track of the spans (start offset/end offset)
        #     citation: keep track of offsets
        rx_heading  = re.compile(r'(?<=\n)(?P<hlevel>={2,5})( ){0,2}(?P<htext>[^=\n]{0,200}([^ ]|<[Rr]ef[^>]*/>))( ){0,2}(?P=hlevel)( )?(?=\n)', re.DOTALL | re.UNICODE)
        rx_citation = re.compile(r'<[Rr]ef([^>]*[^/])?>.*?</[Rr]ef>|<[Rr]ef[^>]*/>|{{([Cc]itation [Nn]eeded|cn)[^{}]*}}', re.DOTALL | re.UNICODE)
        h_spans = []
        c_offsets = []
        h_match = rx_heading.search(text)
        c_match = rx_citation.search(text)
        from_idx = 0
        while h_match is not None or c_match is not None:
            if c_match is None or (h_match is not None and h_match.start() < c_match.start()):
                
                # Check that no citation has been captured in the heading span
                repl_text = h_match.group('htext')
                m = rx_citation.search(repl_text)
                if m is not None:
                    repl_text = rx_citation.sub('', repl_text, count=1)
                
                h_start = h_match.start()
                h_end = h_start + len(repl_text)
                h_level = len(h_match.group('hlevel'))
                h_span = SentSpan(h_start, h_end, h_level)
                h_spans.append(h_span)
                # text = rx_heading.sub('\g<htext>', text, count=1)
                text = rx_heading.sub(repl_text, text, count=1)
                from_idx = h_match.start()
            else:
                c_offsets.append(c_match.start())
                text = rx_citation.sub('', text, count=1)
                from_idx = c_match.start()

            h_match = rx_heading.search(text, from_idx)
            c_match = rx_citation.search(text, from_idx)
        
        class BulletPointLangVars(PunktLanguageVars):
            sent_end_chars = PunktLanguageVars.sent_end_chars + ('•', )
        sent_tokenizer = PunktSentenceTokenizer(lang_vars=BulletPointLangVars())
        # sents = nltk.sent_tokenize(text)    # TODO: use nltk sentence spans method
        sents = sent_tokenizer.span_tokenize(text)
        rx_bullet = re.compile(r'(?<=\n)(?P<bul_and_sp>•[\s]*$)', re.DOTALL | re.UNICODE)
        for i, (sent_span, next_sent_span) in enumerate(zip(sents, sents[1:])):
            start, end = sent_span
            next_start, next_end = next_sent_span
            m = rx_bullet.search(text, start, next_start)
            if m is not None:
                # print('-' * 80)
                # print(repr(text[start:end]))
                # print(repr(text[next_start:next_end]), '\n')
                sents[i] = (start, end - len(m.group('bul_and_sp')))
                sents[i + 1] = (next_start - len(m.group('bul_and_sp')), next_end)
                # print(repr(text[start:end - len(m.group('bul_and_sp'))]))
                # print(repr(text[next_start - len(m.group('bul_and_sp')):next_end]))
                
        sents = [text[start:end] for start, end in sents if start < end]    # TODO: use spans below
        
        # sents = [sent.text for sent in self.nlp(text).sents]
        sentences = []
        start_offset = 0
        end_offset = 0
        curr_c_idx = 0
        curr_h_idx = 0
        next_sent_start = None
        spaces = whitespace + '\xa0'
        for sent in sents:
            # nltk.sent_tokenize() drops spaces/newlines
            end_offset = start_offset + len(sent)
            while end_offset < len(text) and text[end_offset] in spaces:
                end_offset += 1
                
            # May need to adjust if the previous sentence was a heading which
            # was split by the sentence tokenizer
            if next_sent_start is not None:
                start_offset = next_sent_start
                next_sent_start = None
                if start_offset == end_offset:
                    continue

            # The current "sentence" may actually be several sentences (including headings)
            # Count the number of headings in this "sentence"
            tmp_h_index = curr_h_idx
            while tmp_h_index < len(h_spans) and h_spans[tmp_h_index].start_offset < end_offset:
                tmp_h_index += 1
            n_headings = tmp_h_index - curr_h_idx
            # TODO: remove ^, remove if/else below
            sent_spans = []
            if n_headings == 0:
                sent_spans.append(SentSpan(start_offset, end_offset, 0))
            else:
                # "sentence" contains at least one heading (may not be split by
                # nltk so we need to split it before/after each heading)
                span_start = start_offset
                span_end = span_start
                while span_end < end_offset:
                    
                    # Next heading not at span_start
                    next_h_start = h_spans[curr_h_idx].start_offset if curr_h_idx < len(h_spans) else None
                    if next_h_start is None or next_h_start >= end_offset:
                        # This is the last span in this "sentence"
                        sent_span = SentSpan(span_start, end_offset, 0)
                        sent_spans.append(sent_span)
                        break
                    elif span_start < next_h_start:
                        # There is a heading later in the "sentence"
                        # Add the span before the heading
                        sent_span = SentSpan(span_start, next_h_start, 0)
                        sent_spans.append(sent_span)
                        span_start = next_h_start
                    
                    # Add heading and trailing spaces/newlines
                    span_end = h_spans[curr_h_idx].end_offset
                    while span_end < len(text) and text[span_end] in spaces:
                        span_end += 1
                    
                    # If the heading has been split into multiple sentences,
                    # need to adjust start offset of next sentence
                    if span_end > end_offset:
                        next_sent_start = span_end
                        
                    sent_spans.append(SentSpan(span_start, span_end, h_spans[curr_h_idx].h_level))
                    curr_h_idx += 1
                    span_start = span_end
            
            # Add sentence spans to list of Sentence objects
            for span_start, span_end, heading_level in sent_spans:
                # Find citations which relate to this "sentence"
                n_cits = 0
                while curr_c_idx < len(c_offsets) and c_offsets[curr_c_idx] < span_end:
                    n_cits += 1
                    curr_c_idx += 1
                sentences.append(Sentence(text[span_start:span_end],
                                          n_cits if heading_level == 0 else 0,
                                          heading_level))
            start_offset = end_offset

        return sentences

    def get_corpus_splits(self, all_articles):
        """
        Splits the given articles into training, dev and test sets with ratio
        roughly 3:1:1 in size
        :param all_articles: the articles to split into training/dev/test sets
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
