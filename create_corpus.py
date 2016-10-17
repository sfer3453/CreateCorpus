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
        
        # List items are often not split, make sure they are used to detect
        # the end of a sentence and then adjust 
        class BulletPointLangVars(PunktLanguageVars):
            sent_end_chars = PunktLanguageVars.sent_end_chars + ('•', '‣', '▪')
        self.sent_tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
        self.sent_tokenizer._lang_vars = BulletPointLangVars()
        
        # List of abbreviations provided by NLTK does not cover a lot of common
        # cases - add some more to avoid splitting sentences in the wrong place
        abbrevs = ['e.g', 'i.e', 'incl', 'md', 'n.a', 'dept', 'etc', 'fl',
                   'oz', 's.a.b', 'mi', 'b.c', 'govt', 's.l', 'a.k.a', 'p.l.c',
                   'f.c', 'u.a.e' 'al', 's.a.c', 'phd', 'c.e.o', 'i.t', 'llc',
                   'pty', 'ltd', 's.a', 'e.u', 'vol']
        self.sent_tokenizer._params.abbrev_types.update(abbrevs)

    def download_articles(self, article_titles):
        """
        Download articles using the MediaWiki API
        NOTE: if article is moved, there is no error message but only a
        '#REDIRECT [[New Location]]' is recieved in the content.
        Redirect is NOT followed (TODO)
        """
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
        rx_currency    = re.compile(r'{{(?P<currency>USD|GBP|AUD|CNY|JPY|yen|NOK)\|(?P<amount>[\d]+[^|}]*)(\|[^}]*)?}}', re.UNICODE)
        rx_as_of       = re.compile(r'{{(?P<as_of>[Aa]s [Oo]f)\|(?P<year>(\d){4})(\|[^{}]+)?}}', re.DOTALL | re.UNICODE)
        rx_as_of_alt   = re.compile(r'{{(?P<as_of>[Aa]s [Oo]f)\|(?P<date>[\w]* (\d){4})}}', re.DOTALL | re.UNICODE)
        rx_template    = re.compile(r'{{(?!([Cc]itation [Nn]eeded|cn))[^{}]*}}', re.DOTALL | re.UNICODE)
        rx_file_image  = re.compile(r'\[\[([Ii]mage:|[Ff]ile:)[^\[\]]*(\[\[[^\[\]]*\]\]|\[[^\[\]]*\]|[^\[\]]*){0,10}\]\]', re.DOTALL | re.UNICODE)  # may have nested [[...]]
        rx_gallery     = re.compile(r'<[Gg]allery(.*?)</[Gg]allery>', re.DOTALL | re.UNICODE)
        rx_table_inner = re.compile(r'{\|[^{}]*\|}((\s)*?<ref([^>]*/>|[^<]*</ref>))?', re.DOTALL | re.UNICODE)
        rx_table       = re.compile(r'{\|.*?\|}((\s)*?<ref([^>]*/>|[^<]*</ref>))?', re.DOTALL | re.UNICODE)
        rx_bold_italic = re.compile(r"'{2,6}(?P<innertext>.*?)'{2,6}", re.DOTALL | re.UNICODE)
        rx_ellipsis    = re.compile(r'(\.){3,}|\. \. \.')

        rx_tag_empty   = re.compile(r'<(?![Rr]ef)[^>]*/>|<[Bb][Rr]>|<p>', re.DOTALL | re.UNICODE)
        rx_tag_no_keep = re.compile(r'<(?![Rr]ef)(?P<tag>([Gg]allery|[Tt]imeline|[Ss]yntaxhighlight))( [^>]*)?>.*?</(?P=tag)>', re.DOTALL | re.UNICODE)
        rx_tag_keep    = re.compile(r'<(?![Rr]ef)(?P<tag>[a-zA-Z]*)( [^>]*)?>(?P<innertext>.*?)</(?P=tag)>', re.DOTALL | re.UNICODE)
        rx_infobox     = re.compile(r'{{( )?[Ii]nfobox[^{}]*({{[^{}]*({{[^{}]*}}|[^{}]*)*}}|{[^{}]*}|[^{}]*)*}}', re.DOTALL | re.UNICODE)

        rx_html_esc    = re.compile(r'(?P<esc>&[\w]{4,6};)')
        rx_indent      = re.compile(r'\n[\s]*:[\s]*', re.DOTALL | re.UNICODE)
        rx_bul_list    = re.compile(r'(?<=\n)[*\u2022]+[\s]*', re.DOTALL | re.UNICODE)
        rx_def_list    = re.compile(r'\n;.+?(?=\n\n)', re.DOTALL | re.UNICODE)
        rx_num_list    = re.compile(r'(?<=\n)#.+?(?=\n\n)', re.DOTALL | re.UNICODE)
        rx_num_list_item = re.compile(r'(?<=\n)#[ ]*', re.DOTALL | re.UNICODE)
        rx_spaces      = re.compile(r'^[\s]+')
        
        def currency_format(m):
            return '{}{}'.format('$' if m.group('currency') in ['USD', 'AUD']
                                 else '¥' if m.group('currency') in ['CNY', 'JPY', 'yen']
                                 else '£' if m.group('currency') == 'GBP' else '€',
                                 m.group('amount'))
        
        text = rx_end_matter.sub('\n', text)         # Remove end-matter
        text = rx_comment.sub('', text)              # Remove comments
        text = rx_anchor.sub('\g<innertext>', text)  # Remove "visible anchors"
        text = rx_lang.sub('\g<innertext>', text)    # Replace transliteration templates
        text = rx_jap.sub('\g<innertext>', text)     # Replace Japanese transliterations
        text = rx_ill.sub('\g<innertext>', text)     # Replace interlanguage links
        text = rx_currency.sub(currency_format, text)  # Replace currency templates
        text = rx_as_of.sub(lambda m: '{} {}'.format(m.group('as_of'), m.group('year')), text)
        text = rx_as_of_alt.sub(lambda m: '{} {}'.format(m.group('as_of'), m.group('date')), text)
        
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
        text = rx_ellipsis.sub('\u2026', text)  # Sub ellipsis char (helps with sent splits)
        text = rx_tag_no_keep.sub('', text)     # Remove HTML tags and unneeded content
        text = rx_tag_keep.sub('\g<innertext>', text)       # Remove tags, leave content
        text = rx_tag_empty.sub('', text)       # Remove empty (self-closing eg <br />) tags
        text = rx_infobox.sub('', text)         # Remove infoboxes (MUST be done after other template removals)

        text = rx_html_esc.sub(lambda m: html.unescape(m.group('esc')), text)   # Replace HTML-escaped characters
        text = rx_bul_list.sub('• ', text)      # Convert * to • (replace one/many with single), normalize spaces
        text = rx_spaces.sub('', text)          # Remove spaces/newlines at start of article
        
        # Convert numbered lists (starting with hashes) to numbers
        m = rx_num_list.search(text)
        while m is not None:
            num = 1
            li = rx_num_list_item.search(text, max(0, m.start() - 1), m.end())
            while li is not None:
                text = rx_num_list_item.sub('{}. '.format(num), text, 1)
                li = rx_num_list_item.search(text, li.start(), m.end())
                num += 1
            m = rx_num_list.search(text, m.end())
            
        # Convert definition lists.
        # Terms (dt) are replaced with '‣' (triangular bullet, \u2023)
        # Descriptions (dd) are replaced with '▪' (square bullet, \u25aa)
        rx_dt = re.compile(r'(?<=\n)(?P<dt>( )*;)(?P<term>.+?)(?=(?P<end_dt>\n|(?!<=http|https):(?!//)))', re.UNICODE)
        rx_dd = re.compile(r'(?P<pre_dd>[\n \w])(?P<dd>(?!<=http|https):(?!//)|\u2022)(?P<desc>.+?)(?=\n)', re.UNICODE)
        
        def clean_def_list(match):
            repl_text = match.string[match.start():match.end() + 1]
            repl_text = rx_dt.sub(lambda m: '\u2023 {}'.format(m.group('term').strip()), repl_text)
            repl_text = rx_dd.sub(lambda m: '{}\n\u25aa {}'.format(m.group('pre_dd') if m.group('pre_dd') not in ' \n' else '',
                                                                   m.group('desc').strip()), repl_text)
            return repl_text
        text = rx_def_list.sub(clean_def_list, text)
        
        text = rx_indent.sub('\n', text)  # Remove indents (colon at start of line)
        
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
        
        sents = self.sent_tokenizer.span_tokenize(text)
        rx_list_item = re.compile(r'(?<=\n)(?P<li_and_sp>(•|‣|▪|[\d]+\.)[ \t]*$)', re.DOTALL | re.UNICODE)
        for i, (sent_span, next_sent_span) in enumerate(zip(sents, sents[1:])):
            start, end = sent_span
            next_start, next_end = next_sent_span
            m = rx_list_item.search(text, start, next_start)
            if m is not None:   # FIXME: check length of m.group('li_and_sp')?
                sents[i] = (start, end - len(m.group('li_and_sp')))
                sents[i + 1] = (next_start - len(m.group('li_and_sp')), next_end)
                
        # sents = [sent.text for sent in self.nlp(text).sents]
        sentences = []
        curr_c_idx = 0
        curr_h_idx = 0
        next_sent_start = None
        spaces = whitespace + '\xa0'
        rx_nn = re.compile(r'( )*(\n){2,}( )*(?=[\w\d]+)', re.DOTALL | re.UNICODE)
        for i, (start_offset, end_offset) in enumerate(sents):
            # Sometimes there is a double newline in the middle of the sentence...
            m = rx_nn.search(text, start_offset, end_offset)
            if m is not None:
                # Split span: add as next in list, adjust current end_offset
                sents.insert(i + 1, (m.end(), end_offset))
                # sents[i] = (start_offset, m.end())
                end_offset = m.end()
            
            # nltk.sent_tokenize() drops spaces/newlines
            while end_offset < len(text) and text[end_offset] in spaces:
                end_offset += 1
                
            # May need to adjust if the previous sentence was a heading which
            # was split by the sentence tokenizer
            if next_sent_start is not None:
                start_offset = next_sent_start
                next_sent_start = None
                if start_offset == end_offset:
                    continue

            # The current "sentence" may actually be several sentences
            # including headings (may not be split by nltk so we need to
            # split it before/after each heading)
            sent_spans = []
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
                if span_end - span_start > 1:
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
