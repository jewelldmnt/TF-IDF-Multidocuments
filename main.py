import os
import numpy as np
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import nltk
import math
from string import punctuation


def load_documents(folder_path):
    documents = {}
    filenames = []
    for filename in os.listdir(folder_path):
        filenames.append(filename)
        with open(os.path.join(folder_path, filename), 'r') as file:
            text = file.read()
            documents[filename] = text
    return documents, filenames


def tokenize_words(documents):
    for filename, sentence in documents.items():
        words = [word.lower() for word in word_tokenize(sentence) if word.isalpha() and word not in stopwords.words("english")]
        documents[filename] = words
        
        global word_set
        word_set.update(words)
        
    return documents


def compute_term_occurrences(documents_contents):
    term_occurrences = []
    total_term_occurrences = {word: 0 for word in word_set}  # Initialize total term occurrences

    for content in documents_contents:
        term_occurrences_dict = {word: 0 for word in sorted_word_set}
        for word in content:
            term_occurrences_dict[word] += 1
            total_term_occurrences[word] += 1  # Increment total term occurrences
        term_occurrences.append(term_occurrences_dict)
    return term_occurrences, total_term_occurrences


def display_term_occurrences(term_occurrences):
    global sorted_word_set
    term_occurrences_df = pd.DataFrame(index=filenames, columns=sorted_word_set)
    for index, filename in enumerate(filenames):        
        for word in sorted_word_set:
            term_occurrences_df.at[filename, word] = term_occurrences[index][word]
    print(term_occurrences_df)


def compute_term_frequency(documents):
    term_frequency = []
    for document in documents:
        term_frequency_dict = {}
        document_length = len(document)
        
        global sorted_word_set
        for word in sorted_word_set:
            occurence = document.count(word)
            term_frequency_dict[word] = occurence / document_length
        term_frequency.append(term_frequency_dict)
    return term_frequency


def display_term_frequency(term_frequency):
    global sorted_word_set
    term_frequency_df = pd.DataFrame(index=filenames, columns=sorted_word_set)
    for index, filename in enumerate(filenames):              
        for word in sorted_word_set:
            term_frequency_df.at[filename, word] = term_frequency[index][word]
    print(term_frequency_df)


def compute_inverse_document_frequency(total_documents, total_term_occurrences):
    idf_dict = {}
    
    global sorted_word_set
    for word in sorted_word_set:
        idf_dict[word] = math.log10(total_documents / float(total_term_occurrences[word]))
    return idf_dict


def display_idf(idf):
    idf_df = pd.DataFrame.from_dict(idf, orient='index', columns=['idf'])
    print(idf_df)


def compute_tfidf(term_frequency, idf):
    tfidf = []
    for index, dict in enumerate(term_frequency):
        tfidf_temp = {}
        for word in sorted_word_set:
            tfidf_temp[word] = dict[word] * idf[word]
        tfidf.append(tfidf_temp)
    return tfidf


def display_tfidf(tfidf):
    tfidf_df = pd.DataFrame(index=filenames, columns=sorted_word_set)
    for index, filename in enumerate(filenames):
        for word in sorted_word_set:
            tfidf_df.at[filename, word] = tfidf[index][word]
    print(tfidf_df)


def get_top_keywords(tfidf, num_keywords):
    top_keywords = {}
    for index, filename in enumerate(filenames):
        sorted_tfidf = sorted(tfidf[index].items(), key=lambda x: x[1], reverse=True)
        top_keywords[filename] = {word: str(round(score, 4)) for word, score in sorted_tfidf[:num_keywords]}
    return top_keywords

def generate_html_summary(filename, top_keywords):
    output_folder = 'html_summary'
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    html_content = '<!DOCTYPE html>\n'
    html_content += '<html lang="en">\n'
    html_content += '<head>\n' \
                    '\t<meta charset="UTF-8">\n' \
                    '\t<meta http-equiv="X-UA-Compatible" content="IE=edge">\n' \
                    '\t<meta name="viewport" content="width=device-width, initial-scale=1.0">\n' \
                    f'\t<title>{filename}</title>\n' \
                    '\t<link rel="stylesheet" href="/CSS/stylesheet.css">\n' \
                    '</head>\n'
    html_content += '<body>\n' \
                    '\t<header class="header">\n' \
                    '\t\t<h1 class="header__title"><span>TFIDF</span> Summary</h1>\n' \
                    '\t</header>\n'
    html_content += '<main class="main">\n' \
                    '\t<section class="document__keywords_section section">\n' \
                    f'\t\t<h2 class="section__title">Document Title: {filename}</h2>\n' \
                    f'\t\t<h3 class="section__subtitle">TF-IDF Top {len(top_keywords.values())} keywords</h3>\n' \
                    '\t\t<div class="words__occurrences container grid">\n' \
                    '\t\t\t<table class="words__occurrences-table">\n' \
                    '\t\t\t\t<thead>\n' \
                    '\t\t\t\t\t<tr>\n' \
                    '\t\t\t\t\t\t<th>Rank</th>\n' \
                    '\t\t\t\t\t\t<th>Keyword</th>\n' \
                    '\t\t\t\t\t\t<th>TF-IDF Score</th>\n' \
                    '\t\t\t\t\t</tr>\n' \
                    '\t\t\t\t</thead>\n' \
                    '\t\t\t\t<tbody>\n'
    rank = 1
    for keyword, score in top_keywords.items():
        html_content += f'\t\t\t\t<tr>\n' \
                        f'\t\t\t\t\t<td>{rank}</td>\n' \
                        f'\t\t\t\t\t<td>{keyword}</td>\n' \
                        f'\t\t\t\t\t<td>{score}</td>\n' \
                        f'\t\t\t\t</tr>\n'
        rank += 1
    html_content += '\t\t\t\t</tbody>\n' \
                    '\t\t\t</table>\n' \
                    '\t\t</div>\n' \
                    '\t</section>\n' \
                    '\t<section class="document_text_section section">\n' \
                    '\t\t<h3 class="section__subtitle">Document Text</h3>\n' \
                    '\t\t<div class="document_text_container container grid">\n' \
                    '\t\t\t<p class="document__text">\n' \
                    f'\t\t\t\t{documents_text[filename]}\n' \
                    '\t\t\t</p>\n' \
                    '\t\t</div>\n' \
                    '\t</section>\n' \
                    '</main>\n' \
                    '</body>\n' \
                    '</html>'
                    
    # Save HTML file
    html_filename = f'{filename}_summary_document.html'
    output_path = os.path.join(output_folder, html_filename)

    with open(output_path, 'w', encoding='utf-8') as file:
        file.write(html_content)




folder_path = './sample_documents'
documents, filenames = load_documents(folder_path)
documents_text = documents.copy()
total_documents = len(documents)
# Total documents in our corpus
print(f"Total documents: {total_documents}")

word_set = set()
documents = tokenize_words(documents)
print(f"Documents: {documents}")
documents_list = list(documents.values())
print(f"Document lists: {documents_list}")

word_index = {word: i for i, word in enumerate(word_set)}
sorted_word_set = sorted(word_set, key=lambda x: word_index[x])
print(f"\nword_index: {word_index}")
print(f"\nsorted words: {sorted_word_set}\n\n")

for index, filename in enumerate(filenames):
    print(f"{filename} length = {len(documents_list[index])}")

term_occurrences, total_term_occurrences = compute_term_occurrences(documents_list)
print("\nTerm Occurrences:")
display_term_occurrences(term_occurrences)
print("\n")

term_frequency = compute_term_frequency(documents_list)
print(f"Term frequency:")
display_term_frequency(term_frequency)
print("\n")

idf = compute_inverse_document_frequency(total_documents, total_term_occurrences)
print(f"Inverse Document Frequency:")
display_idf(idf)
print("\n")

tfidf = compute_tfidf(term_frequency, idf)
print(f"TF-IDF scores:")
display_tfidf(tfidf)
print("\n")

print(f"Total words in the documents: {len(sorted_word_set)}")
num_keywords = int(input("Enter the number of keywords to generate: "))
top_keywords = get_top_keywords(tfidf, num_keywords)

print("\nTop Keywords:")
for filename, keywords in top_keywords.items():
    generate_html_summary(filename, keywords)
    print(f"HTML file for {filename} was generated")
    print(f"{filename}:")    
    for word, score in keywords.items():
        print(f"{word}: {score}")
    print()


