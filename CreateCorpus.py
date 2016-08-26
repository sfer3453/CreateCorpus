import requests
import re
import html
from urllib import parse
import json
from Utils import get_article_titles
import spacy


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
        Strips all markup from a given document apart from section headings and reference links
        :param text: the document to clean
        :return: the cleaned document
        """
        rx_comment      = re.compile(r'<!--.*?-->', re.DOTALL | re.UNICODE)
        rx_end_matter   = re.compile(r'\n==( )?(Notes|References|See [Aa]lso|External [Ll]inks|Further [Rr]eading)( )?==\n.*$', re.DOTALL | re.UNICODE)

        rx_link         = re.compile(r'\[\[(?!([Ii]mage:|[Ff]ile:))(?P<innertext>(\[[^\[\]]+\]|[^\[\]\|]+){0,3})\]\]', re.DOTALL | re.UNICODE)
        rx_link_alt     = re.compile(r'\[\[(?!([Ii]mage:|[Ff]ile:))([^\[\]\|]*)\|(?P<innertext>(\[[^\[\]]+\]|[^\[\]\|]+){0,3})\]\]', re.DOTALL | re.UNICODE)
        rx_link_ext     = re.compile(r'(?!<=\[)\[[^\[\] ]*( (?P<innertext>[^\[\]]{0,200}))?\](?!\])', re.DOTALL | re.UNICODE)
        rx_anchor       = re.compile(r'{{[Vv]isible [Aa]nchor( )?\|(?P<innertext>[^{}]*)}}')
        rx_template     = re.compile(r'{{(?!([Cc]itation [Nn]eeded|cn))[^{}]*}}', re.DOTALL | re.UNICODE)
        rx_file_image   = re.compile(r'\[\[([Ii]mage:|[Ff]ile:)[^\[\]]*(\[\[[^\[\]]*\]\]|\[[^\[\]]*\]|[^\[\]]*){0,10}\]\]', re.DOTALL | re.UNICODE)  # may have nested [[...]]
        rx_gallery      = re.compile(r'<[Gg]allery(.*?)</[Gg]allery>', re.DOTALL | re.UNICODE)
        rx_table_inner  = re.compile(r'{\|[^{}]*\|}((\s)*?<ref([^>]*/>|[^<]*</ref>))?', re.DOTALL | re.UNICODE)
        rx_table        = re.compile(r'{\|.*?\|}((\s)*?<ref([^>]*/>|[^<]*</ref>))?', re.DOTALL | re.UNICODE)
        rx_bold_italic  = re.compile(r"'{2,6}(?P<innertext>.*?)'{2,6}", re.DOTALL | re.UNICODE)

        rx_tag_empty    = re.compile(r'<(?![Rr]ef)[^>]*/>|<[Bb][Rr]>|<p>', re.DOTALL | re.UNICODE)
        rx_tag_no_keep  = re.compile(r'<(?![Rr]ef)(?P<tag>([Gg]allery|[Tt]imeline|[Ss]yntaxhighlight))( [^>]*)?>.*?</(?P=tag)>', re.DOTALL | re.UNICODE)
        rx_tag_keep     = re.compile(r'<(?![Rr]ef)(?P<tag>[a-zA-Z]*)( [^>]*)?>(?P<innertext>.*?)</(?P=tag)>', re.DOTALL | re.UNICODE)
        rx_infobox      = re.compile(r'{{( )?[Ii]nfobox[^{}]*({{[^{}]*({{[^{}]*}}|[^{}]*)*}}|{[^{}]*}|[^{}]*)*}}', re.DOTALL | re.UNICODE)

        rx_html_esc     = re.compile(r'(?P<esc>&[\s]{4,6};)')

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

        return text

    def get_sentences_and_citations(self, text):
        """

        :param text:
        :return:
        """
        rx_heading          = re.compile(r'(?<=\n)(?P<hlevel>={2,5})( ){0,2}(?P<htext>[^=\n]{0,200}([^ ]|<[Rr]ef[^>]*/>))( ){0,2}(?P=hlevel)( )?(?=\n)', re.DOTALL | re.UNICODE)
        rx_citation         = re.compile(r'<[Rr]ef([^>]*[^/])?>.*?</[Rr]ef>|<[Rr]ef[^>]*/>|{{([Cc]itation [Nn]eeded|cn)[^{}]*}}', re.DOTALL | re.UNICODE)

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

                sentences.append({
                    "text": sent.text[span_start - start_offset:span_end - start_offset],
                    "num_citations": n_cits,
                    "heading_level": heading_level
                })

            start_offset = end_offset

        return sentences
