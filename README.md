# THOTH 

This is an attempt to create a chatbot that would help explore the contents of the released JFK files. It is incomplete, 
mostly because the OCR of the documents resulted in very low quality text. The documents themselves are low quality, many
of them are impossible to read as a human.

The basic pipeline is 

- Gather the files (py/sensors.py)
- Run OCR (optical character recognition) on the pdfs (py/ocr.py)
- Embed the text into a vector space (py/embed.py)
- Map queries into the vector space and collect neighboring documents (incomplete, py/main.py)

The model for the embedding comes from the sentence_transformer library. 

--- 

## File Collection

If you would like to simply download the files you can do `python3 py/sensors.py`. 
It takes the links in meta/deduped_file_links.json and downloads them all. This list was
created from the meta/*.xlsx files, which were collected from the 
[National Archive Website](https://www.archives.gov/research/jfk)

As more files are released, the new xlsx files will have to be collected and 
deduped_file_links.json will have to be regenerated.

## Install

`pip3 install -r py/requirements.txt`