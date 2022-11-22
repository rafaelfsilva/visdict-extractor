#!/bin/python3

import pathlib
import PyPDF2
import sys
import yake
import pandas as pd

folder = pathlib.Path(sys.argv[1])
files = folder.glob("*.pdf")
print(f"Processing PDF files")

kw_dict = {}
f_count = 0

sanity_check_words = ["doi", "fig.", "key", "number", "university", "figure", "notre", "dame", "total", "result", "notre dame", "les"]
kw_extractor = yake.KeywordExtractor()
language = "en"
max_ngram_size = 3
deduplication_threshold = 0.9
numOfKeywords = 100
custom_kw_extractor = yake.KeywordExtractor(
    lan=language, n=max_ngram_size, dedupLim=deduplication_threshold, top=numOfKeywords, features=None)

for filename in files:
    print(f"  Processing: {filename}")
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
                if r in tmp_text :
                    tmp_text = tmp_text[:tmp_text.index(r)]
                    should_break = True
        text += tmp_text
        if should_break:
            break

    text = text.encode("ascii", "ignore").lower().decode('utf-8')
    keywords = custom_kw_extractor.extract_keywords(text)
    for kw in keywords:
        if kw[0] not in sanity_check_words:
            key = kw[0]
            if "workow" in key:
                key = key.replace("workow", "workflow")
            if "scientic" in kw[0]:
                key = key.replace("scientic", "scientific")
            if key not in kw_dict:
                kw_dict[key] = {
                    "values": [],
                    "count": 0
                }
            kw_dict[key]["values"].append(kw[1])
            kw_dict[key]["count"] += 1

    f_count += 1

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

df = pd.DataFrame(data)
df = df.sort_values(by=["count", "score"], ascending=[False, True])

df.head(100).to_csv("keywords.csv")
