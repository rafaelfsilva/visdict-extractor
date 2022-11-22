#!/bin/python3
# -*- coding: utf-8 -*-
#
# Copyright (c) 2022 The VisDict team.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

from nltk.stem.wordnet import WordNetLemmatizer
import nltk
import pathlib
import PyPDF2
import sys
import yake
import pandas as pd

# install NLTK dependencies
nltk.download("wordnet")
nltk.download("omw-1.4")

# required argument: folder to look for PDF files
folder = pathlib.Path(sys.argv[1])
files = folder.glob("*.pdf")
print(f"Processing PDF files")

# general configurations
sanity_check_words = ["doi", "fig.", "key", "number", "university",
                      "figure", "notre", "dame", "total", "result", "notre dame", "les"]
kw_dict = {}
f_count = 0
kw_extractor = yake.KeywordExtractor()
language = "en"
max_ngram_size = 3
deduplication_threshold = 0.9
numOfKeywords = 100
custom_kw_extractor = yake.KeywordExtractor(
    lan=language, n=max_ngram_size, dedupLim=deduplication_threshold, top=numOfKeywords, features=None)
lem = WordNetLemmatizer()

# processing PDF files
for filename in files:
    print(f"  Processing: {filename}")
    # read PDF
    pdfFileObj = open(filename, "rb")
    pdfReader = PyPDF2.PdfFileReader(pdfFileObj, strict=False)
    num_pages = pdfReader.numPages

    count = 0
    text = ""
    should_break = False

    while count < num_pages:
        pageObj = pdfReader.getPage(count)
        count += 1
        tmp_text = pageObj.extractText()

        if count > num_pages * 0.75:
            for r in ["References", "REFERENCES"]:
                if r in tmp_text:
                    # ignoring references
                    tmp_text = tmp_text[:tmp_text.index(r)]
                    should_break = True
        text += tmp_text
        if should_break:
            # do not continue to examine text if references
            break

    # extract keywords
    text = text.encode("ascii", "ignore").lower().decode('utf-8')
    keywords = custom_kw_extractor.extract_keywords(text)
    
    for kw in keywords:
        if kw[0] not in sanity_check_words:
            key = kw[0]
            # sanity checks
            if "workow" in key:
                key = key.replace("workow", "workflow")
            if "scientic" in kw[0]:
                key = key.replace("scientic", "scientific")
            
            # lemmatizing keywords
            key = lem.lemmatize(key)
            
            if key not in kw_dict:
                kw_dict[key] = {
                    "values": [],
                    "count": 0
                }
            kw_dict[key]["values"].append(kw[1])
            kw_dict[key]["count"] += 1

    # code below for debug purposes
    # f_count += 1
    # if f_count == 20:
    #     break

data = {
    "keyword": [],
    "score": [],
    "count": []
}

for key in kw_dict:
    sum = 0.0
    for v in kw_dict[key]["values"]:
        sum += v
    data["keyword"].append(key)
    data["score"].append(sum / kw_dict[key]["count"])
    data["count"].append(kw_dict[key]["count"])

# create data frame and sort by count and score
df = pd.DataFrame(data)
df = df.sort_values(by=["count", "score"], ascending=[False, True])

# write the top 100 keywords to CSV file
df.head(100).to_csv("keywords.csv")
